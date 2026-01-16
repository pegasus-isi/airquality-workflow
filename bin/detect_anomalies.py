#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict


def detect_anomalies_zscore(
    timeseries_file: str,
    threshold: float = 3.0
) -> Dict:
    """
    Detect anomalies in air quality data using Z-score method.

    Args:
        timeseries_file: Path to time series JSON file
        threshold: Z-score threshold for anomaly detection (default: 3.0)

    Returns:
        Dictionary with anomaly detection results
    """
    logging.info(f"Reading time series from {timeseries_file}")

    with open(timeseries_file, 'r') as f:
        data = json.load(f)

    location = data['location']
    measurements = pd.DataFrame(data['measurements'])

    if measurements.empty:
        logging.warning(f"No measurements found in {timeseries_file}")
        return {'location': location, 'anomalies': []}

    # Convert datetime
    measurements['datetime'] = pd.to_datetime(measurements['datetime'])

    anomalies = []

    for parameter in measurements['parameter'].unique():
        param_data = measurements[measurements['parameter'] == parameter].copy()
        param_data = param_data.sort_values('datetime')

        # Calculate Z-scores
        concentrations = param_data['avg_concentration'].values
        mean = np.mean(concentrations)
        std = np.std(concentrations)

        if std == 0:
            logging.warning(f"Standard deviation is 0 for {parameter}, skipping anomaly detection")
            continue

        z_scores = np.abs((concentrations - mean) / std)

        # Detect anomalies
        anomaly_indices = np.where(z_scores > threshold)[0]

        for idx in anomaly_indices:
            row = param_data.iloc[idx]
            anomalies.append({
                'datetime': str(row['datetime']),
                'parameter': parameter,
                'concentration': float(row['avg_concentration']),
                'aqi': float(row['aqi']) if pd.notna(row['aqi']) else None,
                'category': row['category'],
                'z_score': float(z_scores[idx]),
                'severity': 'high' if z_scores[idx] > threshold * 1.5 else 'moderate'
            })

    # Detect sustained high AQI periods
    sustained_alerts = detect_sustained_high_aqi(measurements)

    # Detect sudden spikes
    spike_alerts = detect_spikes(measurements)

    results = {
        'location': location,
        'analysis_period': {
            'start': str(measurements['datetime'].min()),
            'end': str(measurements['datetime'].max())
        },
        'total_measurements': len(measurements),
        'anomalies': {
            'statistical_outliers': anomalies,
            'sustained_high_aqi': sustained_alerts,
            'sudden_spikes': spike_alerts
        },
        'summary': {
            'total_outliers': len(anomalies),
            'total_sustained_alerts': len(sustained_alerts),
            'total_spikes': len(spike_alerts),
            'parameters_affected': list(set([a['parameter'] for a in anomalies]))
        }
    }

    return results


def detect_sustained_high_aqi(measurements: pd.DataFrame, threshold: float = 150, min_duration_hours: int = 3) -> List[Dict]:
    """Detect periods of sustained high AQI levels."""
    alerts = []

    for parameter in measurements['parameter'].unique():
        param_data = measurements[measurements['parameter'] == parameter].copy()
        param_data = param_data.sort_values('datetime')

        # Find consecutive high AQI periods
        high_aqi = param_data['aqi'] >= threshold
        consecutive_start = None
        consecutive_count = 0

        for idx, is_high in enumerate(high_aqi):
            if is_high:
                if consecutive_start is None:
                    consecutive_start = idx
                consecutive_count += 1
            else:
                if consecutive_count >= min_duration_hours:
                    start_row = param_data.iloc[consecutive_start]
                    end_row = param_data.iloc[idx - 1]

                    alerts.append({
                        'parameter': parameter,
                        'start_time': str(start_row['datetime']),
                        'end_time': str(end_row['datetime']),
                        'duration_hours': consecutive_count,
                        'avg_aqi': float(param_data.iloc[consecutive_start:idx]['aqi'].mean()),
                        'max_aqi': float(param_data.iloc[consecutive_start:idx]['aqi'].max())
                    })

                consecutive_start = None
                consecutive_count = 0

        # Check if high period extends to end
        if consecutive_count >= min_duration_hours:
            start_row = param_data.iloc[consecutive_start]
            end_row = param_data.iloc[-1]

            alerts.append({
                'parameter': parameter,
                'start_time': str(start_row['datetime']),
                'end_time': str(end_row['datetime']),
                'duration_hours': consecutive_count,
                'avg_aqi': float(param_data.iloc[consecutive_start:]['aqi'].mean()),
                'max_aqi': float(param_data.iloc[consecutive_start:]['aqi'].max())
            })

    return alerts


def detect_spikes(measurements: pd.DataFrame, spike_factor: float = 2.0) -> List[Dict]:
    """Detect sudden spikes in concentration levels."""
    spikes = []

    for parameter in measurements['parameter'].unique():
        param_data = measurements[measurements['parameter'] == parameter].copy()
        param_data = param_data.sort_values('datetime')

        concentrations = param_data['avg_concentration'].values

        # Calculate hour-over-hour changes
        for i in range(1, len(concentrations)):
            if concentrations[i - 1] > 0:
                change_ratio = concentrations[i] / concentrations[i - 1]

                if change_ratio >= spike_factor:
                    row = param_data.iloc[i]
                    prev_row = param_data.iloc[i - 1]

                    spikes.append({
                        'datetime': str(row['datetime']),
                        'parameter': parameter,
                        'previous_concentration': float(concentrations[i - 1]),
                        'current_concentration': float(concentrations[i]),
                        'change_ratio': float(change_ratio),
                        'absolute_change': float(concentrations[i] - concentrations[i - 1])
                    })

    return spikes


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s", stream=sys.stdout, level=logging.INFO
    )

    parser = argparse.ArgumentParser(
        description="Detect anomalies in air quality time series data"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to input time series JSON file"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path to output JSON file for anomaly results"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for anomaly detection (default: 3.0)"
    )

    args = parser.parse_args()

    results = detect_anomalies_zscore(args.input_file, args.threshold)

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Anomaly detection complete. Results saved to {args.output_file}")
    logging.info(f"Found {results['summary']['total_outliers']} statistical outliers")
    logging.info(f"Found {results['summary']['total_sustained_alerts']} sustained high AQI periods")
    logging.info(f"Found {results['summary']['total_spikes']} sudden spikes")
