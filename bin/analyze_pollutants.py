#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def create_visualization(timeseries_file: str, output_dir: str):
    """
    Create visualizations for air quality time series data.

    Args:
        timeseries_file: Path to time series JSON file
        output_dir: Directory to save visualization images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading time series from {timeseries_file}")

    with open(timeseries_file, 'r') as f:
        data = json.load(f)

    location = data['location']
    measurements = pd.DataFrame(data['measurements'])

    if measurements.empty:
        logging.warning(f"No measurements found in {timeseries_file}")
        return

    # Convert datetime
    measurements['datetime'] = pd.to_datetime(measurements['datetime'])

    # Create figure with subplots for each parameter
    parameters = measurements['parameter'].unique()
    n_params = len(parameters)

    fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
    if n_params == 1:
        axes = [axes]

    fig.suptitle(f'Air Quality Analysis - {location}', fontsize=16, y=0.995)

    # Color mapping for AQI categories
    category_colors = {
        'Good': '#00E400',
        'Moderate': '#FFFF00',
        'Unhealthy for Sensitive Groups': '#FF7E00',
        'Unhealthy': '#FF0000',
        'Very Unhealthy': '#8F3F97',
        'Hazardous': '#7E0023'
    }

    for idx, parameter in enumerate(parameters):
        ax = axes[idx]
        param_data = measurements[measurements['parameter'] == parameter].copy()
        param_data = param_data.sort_values('datetime')

        # Plot concentration values
        ax.plot(param_data['datetime'], param_data['avg_concentration'],
                marker='o', linewidth=2, markersize=4, label=f'{parameter.upper()} concentration')

        # Add AQI-based background colors
        for i in range(len(param_data) - 1):
            category = param_data.iloc[i]['category']
            color = category_colors.get(category, '#CCCCCC')
            ax.axvspan(param_data.iloc[i]['datetime'],
                      param_data.iloc[i + 1]['datetime'],
                      alpha=0.2, color=color)

        # Formatting
        ax.set_xlabel('Date/Time', fontsize=10)
        unit = param_data['unit'].iloc[0] if len(param_data) > 0 else ''
        ax.set_ylabel(f'Concentration ({unit})', fontsize=10)
        ax.set_title(f'{parameter.upper()} Levels Over Time', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(param_data) // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    safe_location = location.replace(' ', '_').replace('-', '_').replace('/', '_')

    # Save figure
    output_file = Path(output_dir) / f"{safe_location}_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved visualization to {output_file}")

    # Generate statistics summary
    stats = generate_statistics(measurements, location)
    stats_file = Path(output_dir) / f"{safe_location}_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    logging.info(f"Saved statistics to {stats_file}")


def generate_statistics(measurements: pd.DataFrame, location: str) -> dict:
    """Generate statistical summary of measurements."""
    stats = {
        'location': location,
        'time_range': {
            'start': measurements['datetime'].min(),
            'end': measurements['datetime'].max()
        },
        'parameters': {}
    }

    for parameter in measurements['parameter'].unique():
        param_data = measurements[measurements['parameter'] == parameter]

        stats['parameters'][parameter] = {
            'count': len(param_data),
            'mean_concentration': float(param_data['avg_concentration'].mean()),
            'min_concentration': float(param_data['avg_concentration'].min()),
            'max_concentration': float(param_data['avg_concentration'].max()),
            'std_concentration': float(param_data['avg_concentration'].std()),
            'mean_aqi': float(param_data['aqi'].mean()) if param_data['aqi'].notna().any() else None,
            'max_aqi': float(param_data['aqi'].max()) if param_data['aqi'].notna().any() else None,
            'category_distribution': param_data['category'].value_counts().to_dict()
        }

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s", stream=sys.stdout, level=logging.INFO
    )

    parser = argparse.ArgumentParser(
        description="Analyze air quality data and create visualizations"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to input time series JSON file"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis",
        help="Path to output directory for visualizations. Default is `analysis`."
    )

    args = parser.parse_args()
    create_visualization(args.input_file, args.output_dir)
