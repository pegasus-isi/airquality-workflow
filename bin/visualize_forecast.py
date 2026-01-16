#!/usr/bin/env python3

"""
Visualize AQI forecasts with historical context.

This script creates visualizations showing historical AQI data and
forecasted values with confidence intervals.

Usage:
    python visualize_forecast.py --timeseries timeseries/location_timeseries.json \
                                 --forecast forecasts/location/location_forecast.json \
                                 --output forecasts/location/location_forecast.png
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AQI category colors (EPA standard)
AQI_COLORS = {
    'Good': '#00E400',
    'Moderate': '#FFFF00',
    'Unhealthy for Sensitive Groups': '#FF7E00',
    'Unhealthy': '#FF0000',
    'Very Unhealthy': '#8F3F97',
    'Hazardous': '#7E0023'
}

# AQI breakpoints
AQI_BREAKPOINTS = [
    (0, 50, 'Good'),
    (51, 100, 'Moderate'),
    (101, 150, 'Unhealthy for Sensitive Groups'),
    (151, 200, 'Unhealthy'),
    (201, 300, 'Very Unhealthy'),
    (301, 500, 'Hazardous')
]


def load_timeseries(timeseries_file: str, lookback_days: int = 7) -> pd.DataFrame:
    """
    Load historical timeseries data.

    Args:
        timeseries_file: Path to timeseries JSON file
        lookback_days: Number of days of history to show

    Returns:
        DataFrame with historical AQI data
    """
    logger.info(f"Loading timeseries from {timeseries_file}")

    with open(timeseries_file, 'r') as f:
        timeseries_data = json.load(f)

    measurements = timeseries_data.get('measurements', [])

    if not measurements:
        logger.warning("No measurements found in timeseries")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(measurements)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Get last N days
    cutoff_time = df['datetime'].max() - timedelta(days=lookback_days)
    df = df[df['datetime'] >= cutoff_time]

    # Use AQI if available, otherwise use concentration as proxy
    if 'aqi' not in df.columns:
        logger.warning("AQI not in timeseries, using avg_concentration as proxy")
        df['aqi'] = df['avg_concentration']

    # Aggregate by hour (average across parameters)
    df_hourly = df.groupby(pd.Grouper(key='datetime', freq='h')).agg({
        'aqi': 'mean'
    }).reset_index()

    logger.info(f"Loaded {len(df_hourly)} hours of historical data")

    return df_hourly


def load_forecast(forecast_file: str) -> pd.DataFrame:
    """
    Load forecast data.

    Args:
        forecast_file: Path to forecast JSON file

    Returns:
        DataFrame with forecast data
    """
    logger.info(f"Loading forecast from {forecast_file}")

    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)

    predictions = forecast_data.get('predictions', [])

    if not predictions:
        raise ValueError("No predictions found in forecast file")

    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    logger.info(f"Loaded {len(df)} hours of forecast data")

    return df, forecast_data


def create_forecast_plot(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    forecast_metadata: dict,
    output_path: str
):
    """
    Create forecast visualization.

    Args:
        historical_df: Historical AQI data
        forecast_df: Forecast data with confidence intervals
        forecast_metadata: Forecast metadata
        output_path: Output file path
    """
    logger.info("Creating forecast visualization...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Add AQI category background colors
    for lower, upper, category in AQI_BREAKPOINTS:
        ax.axhspan(
            lower, upper,
            alpha=0.1,
            color=AQI_COLORS[category],
            zorder=0
        )

    # Plot historical data
    if not historical_df.empty:
        ax.plot(
            historical_df['datetime'],
            historical_df['aqi'],
            color='#1f77b4',
            linewidth=2,
            marker='o',
            markersize=3,
            label='Historical AQI',
            zorder=3
        )

        # Vertical line separating historical and forecast
        last_historical_time = historical_df['datetime'].max()
        ax.axvline(
            last_historical_time,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label='Forecast Start',
            zorder=2
        )

    # Plot forecast
    ax.plot(
        forecast_df['datetime'],
        forecast_df['predicted_aqi'],
        color='#ff7f0e',
        linewidth=2,
        linestyle='--',
        marker='s',
        markersize=4,
        label='Forecast AQI',
        zorder=3
    )

    # Plot confidence intervals
    ax.fill_between(
        forecast_df['datetime'],
        forecast_df['confidence_interval_lower'],
        forecast_df['confidence_interval_upper'],
        color='#ff7f0e',
        alpha=0.2,
        label='95% Confidence Interval',
        zorder=1
    )

    # Formatting
    ax.set_xlabel('Date/Time (UTC)', fontsize=12)
    ax.set_ylabel('Air Quality Index (AQI)', fontsize=12)
    ax.set_title(
        f"AQI Forecast - {forecast_metadata.get('location', 'Unknown Location')}\n"
        f"Generated: {pd.to_datetime(forecast_metadata.get('forecast_generated')).strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=14,
        fontweight='bold'
    )

    # X-axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Y-axis limits
    if not historical_df.empty:
        y_min = min(
            historical_df['aqi'].min(),
            forecast_df['confidence_interval_lower'].min()
        )
        y_max = max(
            historical_df['aqi'].max(),
            forecast_df['confidence_interval_upper'].max()
        )
    else:
        y_min = forecast_df['confidence_interval_lower'].min()
        y_max = forecast_df['confidence_interval_upper'].max()

    # Add some padding
    y_range = y_max - y_min
    ax.set_ylim(
        max(0, y_min - 0.1 * y_range),
        min(500, y_max + 0.1 * y_range)
    )

    # Add category labels on right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([25, 75, 125, 175, 250, 400])
    ax2.set_yticklabels([
        'Good', 'Moderate', 'Unhealthy\nfor Sensitive',
        'Unhealthy', 'Very\nUnhealthy', 'Hazardous'
    ], fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)

    # Legend
    model_info = forecast_metadata.get('model_info', {})
    legend_text = [
        f"Model: {model_info.get('architecture', 'LSTM')}",
        f"Horizon: {forecast_metadata.get('forecast_horizon_hours', 24)}h"
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper left',
        fontsize=10,
        title='\n'.join(legend_text)
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {output_path}")


def create_summary_json(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    forecast_metadata: dict,
    output_path: str
):
    """
    Create summary statistics JSON.

    Args:
        historical_df: Historical AQI data
        forecast_df: Forecast data
        forecast_metadata: Forecast metadata
        output_path: Output file path
    """
    summary = {
        'location': forecast_metadata.get('location'),
        'forecast_generated': forecast_metadata.get('forecast_generated'),
        'historical_period': {
            'start': historical_df['datetime'].min().isoformat() if not historical_df.empty else None,
            'end': historical_df['datetime'].max().isoformat() if not historical_df.empty else None,
            'hours': len(historical_df)
        },
        'forecast_period': {
            'start': forecast_df['datetime'].min().isoformat(),
            'end': forecast_df['datetime'].max().isoformat(),
            'hours': len(forecast_df)
        },
        'forecast_statistics': {
            'mean_aqi': float(forecast_df['predicted_aqi'].mean()),
            'min_aqi': float(forecast_df['predicted_aqi'].min()),
            'max_aqi': float(forecast_df['predicted_aqi'].max()),
            'std_aqi': float(forecast_df['predicted_aqi'].std())
        },
        'category_distribution': forecast_df['predicted_category'].value_counts().to_dict()
    }

    # Save
    output_file = Path(output_path).parent / (Path(output_path).stem + '_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AQI forecasts with historical context",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--timeseries",
        type=str,
        required=True,
        help="Path to historical timeseries JSON file"
    )
    parser.add_argument(
        "--forecast",
        type=str,
        required=True,
        help="Path to forecast JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG file path"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Number of days of historical data to show (default: 7)"
    )

    args = parser.parse_args()

    try:
        # Load data
        historical_df = load_timeseries(args.timeseries, args.lookback_days)
        forecast_df, forecast_metadata = load_forecast(args.forecast)

        # Create visualization
        create_forecast_plot(
            historical_df,
            forecast_df,
            forecast_metadata,
            args.output
        )

        # Create summary
        create_summary_json(
            historical_df,
            forecast_df,
            forecast_metadata,
            args.output
        )

        logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
