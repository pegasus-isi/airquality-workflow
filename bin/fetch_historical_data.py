#!/usr/bin/env python3

"""
Fetch historical air quality data from OpenAQ API for LSTM training.

This script fetches 90 days (or custom range) of historical air quality measurements
from OpenAQ API v3 for use in training AQI forecasting models.

Usage:
    python fetch_historical_data.py --location-id 2178 --days 90 --output historical/location_2178_historical.csv
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_api_key() -> str:
    """
    Get OpenAQ API key from environment variable.

    Returns:
        API key string

    Raises:
        ValueError if API key not found
    """
    api_key = os.environ.get('OPENAQ_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAQ API key not found. Please set OPENAQ_API_KEY environment variable.\n"
            "Get your API key at: https://explore.openaq.org/register"
        )
    return api_key


def fetch_historical_data(
    location_id: int,
    start_date: datetime,
    end_date: datetime,
    parameters: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical air quality measurements from OpenAQ API v3.

    Args:
        location_id: OpenAQ location ID
        start_date: Start date for historical data
        end_date: End date for historical data
        parameters: List of parameters to fetch (pm25, pm10, o3, no2, so2, co)
        api_key: OpenAQ API key

    Returns:
        DataFrame with historical measurements
    """
    if api_key is None:
        api_key = get_api_key()

    if parameters is None:
        parameters = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

    base_url = "https://api.openaq.org/v3"
    headers = {"X-API-Key": api_key}
    all_measurements = []

    logger.info(f"Fetching historical data for location {location_id}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    try:
        # Get location details
        loc_response = requests.get(
            f"{base_url}/locations/{location_id}",
            headers=headers,
            timeout=30
        )
        loc_response.raise_for_status()
        loc_response_json = loc_response.json()

        # Handle different response formats
        if 'results' in loc_response_json and isinstance(loc_response_json['results'], list):
            results = loc_response_json['results']
            loc_data = results[0] if results else {}
        elif 'results' in loc_response_json and isinstance(loc_response_json['results'], dict):
            loc_data = loc_response_json['results']
        elif isinstance(loc_response_json, dict):
            loc_data = loc_response_json
        else:
            logger.warning("Unexpected location response format")
            loc_data = {}

        location_name = (
            loc_data.get('name') or
            loc_data.get('displayName') or
            loc_data.get('label') or
            f"Location_{location_id}"
        )

        logger.info(f"Location: {location_name}")

        # Get sensors for this location
        sensors_response = requests.get(
            f"{base_url}/locations/{location_id}/sensors",
            headers=headers,
            params={'limit': 1000},
            timeout=30
        )
        sensors_response.raise_for_status()
        sensors_data = sensors_response.json()

        # Fetch measurements from each sensor
        for sensor in sensors_data.get('results', []):
            sensor_id = sensor.get('id')
            param_info = sensor.get('parameter', {})
            param_name = param_info.get('name', '').lower()

            # Filter by requested parameters
            if parameters and param_name not in parameters:
                continue

            logger.info(f"  Fetching sensor {sensor_id} ({param_name})...")

            # Use daily aggregates endpoint (use hours endpoint for hourly data)
            # For historical training data, we'll use hourly data for better resolution
            measurements_url = f"{base_url}/sensors/{sensor_id}/hours"
            params = {
                'limit': 1000,
                'date_from': start_date.strftime('%Y-%m-%d'),
                'date_to': end_date.strftime('%Y-%m-%d')
            }

            # Paginate through results
            page = 1
            while True:
                params['page'] = page
                meas_response = requests.get(
                    measurements_url,
                    headers=headers,
                    params=params,
                    timeout=30
                )
                meas_response.raise_for_status()
                meas_data = meas_response.json()

                results = meas_data.get('results', [])
                if not results:
                    break

                for measurement in results:
                    # Extract datetime from period object
                    period = measurement.get('period', {})
                    datetime_from = period.get('datetimeFrom', {})
                    datetime_str = datetime_from.get('utc') or datetime_from.get('local')

                    # Extract summary statistics
                    summary = measurement.get('summary', {})

                    all_measurements.append({
                        'location': location_name,
                        'location_id': location_id,
                        'sensor_id': sensor_id,
                        'city': loc_data.get('city'),
                        'country': loc_data.get('country', {}).get('code') if isinstance(loc_data.get('country'), dict) else loc_data.get('country'),
                        'parameter': param_name,
                        'value': measurement.get('value'),  # Average value
                        'min_value': summary.get('min'),
                        'max_value': summary.get('max'),
                        'median_value': summary.get('median'),
                        'sd_value': summary.get('sd'),
                        'unit': param_info.get('units'),
                        'datetime': datetime_str,
                        'coordinates_lat': loc_data.get('coordinates', {}).get('latitude'),
                        'coordinates_lon': loc_data.get('coordinates', {}).get('longitude'),
                    })

                # Check if there are more pages
                meta = meas_data.get('meta', {})
                if page >= meta.get('pages', 1):
                    break
                page += 1

                logger.info(f"    Fetched page {page-1}/{meta.get('pages', 1)}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for location {location_id}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"  Response status: {e.response.status_code}")
            logger.error(f"  Response body: {e.response.text[:500]}")
        raise

    # Convert to DataFrame
    if not all_measurements:
        logger.warning("No measurements found for the specified criteria")
        return pd.DataFrame()

    df = pd.DataFrame(all_measurements)

    # Convert datetime to timestamp
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)

    # Remove rows with invalid dates
    invalid_dates = df['datetime'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"{invalid_dates} measurements had invalid dates and were removed")
        df = df.dropna(subset=['datetime'])

    if df.empty:
        logger.error("No valid measurements after date parsing")
        return pd.DataFrame()

    df['timestamp'] = df['datetime'].astype(int) // 10**9

    # Group by hour
    df['hour_bucket'] = df['datetime'].dt.floor('h')

    logger.info(f"Fetched {len(df)} measurements")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Parameters: {df['parameter'].unique().tolist()}")

    return df


def validate_data_completeness(
    df: pd.DataFrame,
    expected_hours: int,
    min_coverage: float = 0.7
) -> dict:
    """
    Validate that the historical data has sufficient coverage for training.

    Args:
        df: DataFrame with historical measurements
        expected_hours: Expected number of hours in the date range
        min_coverage: Minimum required data coverage (0.0 to 1.0)

    Returns:
        Dictionary with validation results
    """
    if df.empty:
        return {
            'valid': False,
            'coverage': 0.0,
            'message': 'No data available'
        }

    # Count unique hours with data for each parameter
    param_coverage = {}
    for param in df['parameter'].unique():
        param_df = df[df['parameter'] == param]
        unique_hours = param_df['hour_bucket'].nunique()
        coverage = unique_hours / expected_hours if expected_hours > 0 else 0
        param_coverage[param] = {
            'hours': unique_hours,
            'expected': expected_hours,
            'coverage': coverage
        }

    # Overall coverage (average across parameters)
    avg_coverage = sum(p['coverage'] for p in param_coverage.values()) / len(param_coverage) if param_coverage else 0

    is_valid = avg_coverage >= min_coverage

    return {
        'valid': is_valid,
        'coverage': avg_coverage,
        'min_coverage': min_coverage,
        'parameter_coverage': param_coverage,
        'message': f"Coverage: {avg_coverage:.1%} ({'valid' if is_valid else 'insufficient'})"
    }


def save_historical_catalog(
    df: pd.DataFrame,
    output_file: str,
    coverage_info: dict
):
    """
    Save historical catalog to CSV file.

    Args:
        df: DataFrame with historical measurements
        output_file: Path to output CSV file
        coverage_info: Data coverage validation results
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save measurements
    df.to_csv(output_file, index=False)
    logger.info(f"Historical data saved to {output_file}")

    # Save coverage information
    coverage_file = output_path.parent / f"{output_path.stem}_coverage.json"
    with open(coverage_file, 'w') as f:
        json.dump(coverage_info, f, indent=2)
    logger.info(f"Coverage info saved to {coverage_file}")

    # Log summary
    logger.info(f"Total measurements: {len(df)}")
    logger.info(f"Parameters: {df['parameter'].unique().tolist()}")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Data coverage: {coverage_info['coverage']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical air quality data for LSTM training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--location-id",
        type=int,
        required=True,
        help="OpenAQ location ID"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of historical data to fetch (default: 90)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for historical data (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=['pm25', 'pm10', 'o3', 'no2', 'so2', 'co'],
        default=None,
        help="Parameters to fetch (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.7,
        help="Minimum required data coverage (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAQ API key (or set OPENAQ_API_KEY env var)"
    )

    args = parser.parse_args()

    # Set API key from argument if provided
    if args.api_key:
        os.environ['OPENAQ_API_KEY'] = args.api_key

    # Calculate date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=args.days)

    # Fetch historical data
    try:
        df = fetch_historical_data(
            location_id=args.location_id,
            start_date=start_date,
            end_date=end_date,
            parameters=args.parameters
        )

        if df.empty:
            logger.error("No data fetched. Exiting.")
            sys.exit(1)

        # Validate data completeness
        expected_hours = args.days * 24
        coverage_info = validate_data_completeness(df, expected_hours, args.min_coverage)

        logger.info(coverage_info['message'])
        for param, info in coverage_info.get('parameter_coverage', {}).items():
            logger.info(f"  {param}: {info['hours']}/{info['expected']} hours ({info['coverage']:.1%})")

        if not coverage_info['valid']:
            logger.warning(
                f"Data coverage ({coverage_info['coverage']:.1%}) is below "
                f"minimum threshold ({args.min_coverage:.1%})"
            )
            logger.warning("Model training may be less accurate with insufficient data")

        # Save data
        save_historical_catalog(df, args.output, coverage_info)

        logger.info("Historical data fetch completed successfully")

    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
