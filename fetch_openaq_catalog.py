#!/usr/bin/env python3

import pandas as pd
import requests
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional

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


def search_locations(
    country: Optional[str] = None,
    city: Optional[str] = None,
    location_name: Optional[str] = None,
    bbox: Optional[tuple] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Search for locations in OpenAQ v3 API.

    Args:
        country: Country code (e.g., 'US', 'GB')
        city: City name
        location_name: Location name to search for
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        api_key: OpenAQ API key

    Returns:
        DataFrame with location information
    """
    if api_key is None:
        api_key = get_api_key()

    base_url = "https://api.openaq.org/v3/locations"
    headers = {"X-API-Key": api_key}

    params = {"limit": 1000}
    if country:
        params['countries'] = country
    if city:
        params['city'] = city
    if bbox:
        params['bbox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        locations = []
        for loc in data.get('results', []):
            if location_name and location_name.lower() not in loc.get('name', '').lower():
                continue

            locations.append({
                'id': loc.get('id'),
                'name': loc.get('name'),
                'city': loc.get('city'),
                'country': loc.get('country'),
                'latitude': loc.get('coordinates', {}).get('latitude'),
                'longitude': loc.get('coordinates', {}).get('longitude'),
                'parameters': [p.get('name') for p in loc.get('parameters', [])]
            })

        return pd.DataFrame(locations)

    except requests.exceptions.RequestException as e:
        print(f"Error searching locations: {e}")
        return pd.DataFrame()


def fetch_openaq_catalog(
    location_ids: List[int],
    start_date: datetime,
    end_date: datetime,
    parameters: Optional[List[str]] = None,
    limit_per_page: int = 1000,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch air quality measurements from OpenAQ API v3.

    Args:
        location_ids: List of location IDs (integers) to fetch data for
        start_date: Start date for measurements
        end_date: End date for measurements
        parameters: List of parameters to fetch (pm25, pm10, o3, no2, so2, co)
                   If None, fetches all available parameters
        limit_per_page: Number of results per API page
        api_key: OpenAQ API key (if None, reads from OPENAQ_API_KEY env var)

    Returns:
        DataFrame with air quality measurements
    """
    if api_key is None:
        api_key = get_api_key()

    if parameters is None:
        parameters = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

    # Map common parameter names to OpenAQ v3 parameter IDs
    parameter_map = {
        'pm25': 2,    # PM2.5
        'pm10': 1,    # PM10
        'o3': 5,      # Ozone
        'no2': 3,     # Nitrogen Dioxide
        'so2': 4,     # Sulfur Dioxide
        'co': 7       # Carbon Monoxide
    }

    base_url = "https://api.openaq.org/v3"
    headers = {"X-API-Key": api_key}
    all_measurements = []

    # Fetch measurements for each location
    for location_id in location_ids:
        print(f"Fetching data for location ID: {location_id}")

        try:
            # Get location details first
            loc_response = requests.get(
                f"{base_url}/locations/{location_id}",
                headers=headers,
                timeout=30
            )
            loc_response.raise_for_status()
            loc_response_json = loc_response.json()

            # OpenAQ v3 API wraps location data in 'results' or returns it directly
            if 'results' in loc_response_json and isinstance(loc_response_json['results'], list):
                # If results is a list, take the first element
                results = loc_response_json['results']
                loc_data = results[0] if results else {}
            elif 'results' in loc_response_json and isinstance(loc_response_json['results'], dict):
                loc_data = loc_response_json['results']
            elif isinstance(loc_response_json, dict):
                loc_data = loc_response_json
            else:
                print(f"  Warning: Unexpected location response format")
                loc_data = {}

            # Debug: Show available fields (first location only)
            if location_id == location_ids[0] and loc_data:
                print(f"  Debug: Location API response keys: {list(loc_data.keys())}")

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

                print(f"  Fetching sensor {sensor_id} ({param_name})...")

                # Use daily aggregates endpoint for date range queries
                measurements_url = f"{base_url}/sensors/{sensor_id}/days"
                params = {
                    'limit': limit_per_page,
                    'date_from': start_date.strftime('%Y-%m-%d'),
                    'date_to': end_date.strftime('%Y-%m-%d')
                }

                meas_response = requests.get(measurements_url, headers=headers, params=params, timeout=30)
                meas_response.raise_for_status()
                meas_data = meas_response.json()

                # Debug: Show measurement structure (first sensor only)
                if location_id == location_ids[0] and sensor_id == sensors_data.get('results', [{}])[0].get('id'):
                    if meas_data.get('results'):
                        print(f"  Debug: Measurement keys: {list(meas_data['results'][0].keys())}")
                        period = meas_data['results'][0].get('period', {})
                        datetime_from = period.get('datetimeFrom', {})
                        print(f"  Debug: Sample datetime: {datetime_from.get('utc')}")

                for measurement in meas_data.get('results', []):
                    # Extract location name with fallbacks
                    location_name = (
                        loc_data.get('name') or
                        loc_data.get('displayName') or
                        loc_data.get('label') or
                        f"Location_{location_id}"
                    )

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
                        'value': measurement.get('value'),  # This is the average value
                        'min_value': summary.get('min'),
                        'max_value': summary.get('max'),
                        'median_value': summary.get('median'),
                        'sd_value': summary.get('sd'),
                        'unit': param_info.get('units'),
                        'datetime': datetime_str,
                        'coordinates_lat': loc_data.get('coordinates', {}).get('latitude'),
                        'coordinates_lon': loc_data.get('coordinates', {}).get('longitude'),
                    })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for location {location_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response status: {e.response.status_code}")
                print(f"  Response body: {e.response.text[:500]}")
            continue

    # Convert to DataFrame
    if not all_measurements:
        print("No measurements found for the specified criteria")
        return pd.DataFrame()

    df = pd.DataFrame(all_measurements)

    # Convert datetime to timestamp with error handling
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)

    # Remove rows with invalid dates
    invalid_dates = df['datetime'].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: {invalid_dates} measurements had invalid dates and were removed")
        df = df.dropna(subset=['datetime'])

    if df.empty:
        print("No valid measurements after date parsing")
        return pd.DataFrame()

    df['timestamp'] = df['datetime'].astype(int) // 10**9

    # Group by location and hour for workflow processing
    df['hour_bucket'] = df['datetime'].dt.floor('h')  # Use 'h' instead of deprecated 'H'

    return df


def save_catalog(df: pd.DataFrame, output_file: str = "openaq_catalog.csv"):
    """Save the catalog to CSV file."""
    df.to_csv(output_file, index=False)
    print(f"Catalog saved to {output_file}")
    print(f"Total measurements: {len(df)}")
    print(f"Locations: {df['location'].nunique()}")
    print(f"Parameters: {df['parameter'].unique().tolist()}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch OpenAQ air quality catalog (v3 API)",
        epilog="""
Examples:
  # Search for locations first
  %(prog)s --search --city "Los Angeles"

  # Fetch data using location IDs
  %(prog)s --location-ids 2178 4603 --start-date 2024-01-01

  # Use bounding box to find locations
  %(prog)s --search --bbox -118.668 33.704 -118.155 34.337
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search for locations (returns location IDs)"
    )

    # Search parameters
    parser.add_argument(
        "--city",
        type=str,
        help="City name for location search"
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Country code (e.g., US, GB, CN)"
    )
    parser.add_argument(
        "--location-name",
        type=str,
        help="Location name to search for (partial match)"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
        help="Bounding box for location search"
    )

    # Data fetching parameters
    parser.add_argument(
        "--location-ids",
        nargs="+",
        type=int,
        help="Location IDs to fetch data for (integers)"
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=None,
        help="End date (YYYY-MM-DD), defaults to start_date + 1 day"
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
        default="openaq_catalog.csv",
        help="Output CSV file (default: openaq_catalog.csv)"
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

    # Search mode
    if args.search:
        print("Searching for locations...")
        locations_df = search_locations(
            country=args.country,
            city=args.city,
            location_name=args.location_name,
            bbox=tuple(args.bbox) if args.bbox else None
        )

        if not locations_df.empty:
            print(f"\nFound {len(locations_df)} locations:")
            print(locations_df[['id', 'name', 'city', 'country']].to_string(index=False))
            print("\nUse --location-ids with the IDs above to fetch data")

            # Save locations to file
            locations_df.to_csv("openaq_locations.csv", index=False)
            print(f"\nLocations saved to openaq_locations.csv")
        else:
            print("No locations found matching criteria")
        exit(0)

    # Fetch mode
    if not args.location_ids:
        parser.error("--location-ids is required (unless using --search mode)")

    if not args.start_date:
        parser.error("--start-date is required for fetching data")

    if not args.end_date:
        args.end_date = args.start_date + timedelta(days=1)

    print(f"Fetching OpenAQ data from {args.start_date} to {args.end_date}")
    print(f"Location IDs: {args.location_ids}")

    df = fetch_openaq_catalog(
        location_ids=args.location_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        parameters=args.parameters
    )

    if not df.empty:
        save_catalog(df, args.output)
    else:
        print("No data fetched. Please check your parameters.")
        print("\nTip: Use --search to find valid location IDs first")
