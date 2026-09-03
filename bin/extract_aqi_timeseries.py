#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def calculate_aqi(parameter: str, concentration: float) -> dict:
    """
    Calculate Air Quality Index (AQI) based on EPA standards.

    Args:
        parameter: Pollutant type (pm25, pm10, o3, no2, so2, co)
        concentration: Concentration value in appropriate units

    Returns:
        Dictionary with AQI value and category
    """
    # AQI breakpoints (simplified EPA standard)
    breakpoints = {
        'pm25': [  # µg/m³, 24-hour average
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500),
        ],
        'pm10': [  # µg/m³, 24-hour average
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 604, 301, 500),
        ],
        'o3': [  # ppb, 8-hour average
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
        ],
        'no2': [  # ppb, 1-hour average
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 2049, 301, 500),
        ],
    }

    categories = [
        (0, 50, "Good"),
        (51, 100, "Moderate"),
        (101, 150, "Unhealthy for Sensitive Groups"),
        (151, 200, "Unhealthy"),
        (201, 300, "Very Unhealthy"),
        (301, 500, "Hazardous"),
    ]

    if parameter not in breakpoints:
        return {"aqi": None, "category": "Unknown", "raw_value": concentration}

    # Find the appropriate breakpoint
    aqi_value = None
    for c_low, c_high, i_low, i_high in breakpoints[parameter]:
        if c_low <= concentration <= c_high:
            # Linear interpolation
            aqi_value = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            break

    if aqi_value is None:
        # Handle values outside range
        if concentration > breakpoints[parameter][-1][1]:
            aqi_value = 500  # Maximum
        else:
            aqi_value = 0

    # Determine category
    category = "Unknown"
    for low, high, cat in categories:
        if low <= aqi_value <= high:
            category = cat
            break

    return {
        "aqi": round(aqi_value, 1),
        "category": category,
        "raw_value": concentration,
        "parameter": parameter
    }


def extract_aqi_timeseries(input_file: str, output_dir: str):
    """
    Extract AQI time series from raw measurement data.

    Args:
        input_file: Path to input CSV file with air quality measurements
        output_dir: Directory to save processed time series data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)

    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Group by location, parameter, and hour
    # 'h' rather than the deprecated 'H', which pandas >= 2.2 rejects outright
    grouped = df.groupby(['location', 'parameter', pd.Grouper(key='datetime', freq='h')])

    results = []
    for (location, parameter, hour), group in grouped:
        # Calculate average concentration for the hour
        avg_concentration = group['value'].mean()

        # Calculate AQI
        aqi_data = calculate_aqi(parameter, avg_concentration)

        results.append({
            'location': location,
            'parameter': parameter,
            'datetime': hour,
            'avg_concentration': avg_concentration,
            'unit': group['unit'].iloc[0] if len(group) > 0 else 'unknown',
            'aqi': aqi_data['aqi'],
            'category': aqi_data['category'],
            'num_measurements': len(group)
        })

    results_df = pd.DataFrame(results)

    # Save to JSON for each location
    for location in results_df['location'].unique():
        location_data = results_df[results_df['location'] == location]

        # Sanitize location name for filesystem (match workflow_generator.py logic)
        safe_location = location.replace(' ', '_').replace('-', '_').replace('/', '_')
        output_file = Path(output_dir) / f"{safe_location}_timeseries.json"

        # Convert to dict format
        output_data = {
            'location': location,
            'measurements': location_data.to_dict(orient='records')
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logging.info(f"Saved time series for {location} to {output_file}")

    logging.info(f"Processed {len(results_df)} hourly measurements")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s", stream=sys.stdout, level=logging.INFO
    )

    parser = argparse.ArgumentParser(
        description="Extract AQI time series from air quality measurements"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to input CSV file with measurements"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="timeseries",
        help="Path to output directory for time series data. Default is `timeseries`."
    )

    args = parser.parse_args()
    extract_aqi_timeseries(args.input_file, args.output_dir)
