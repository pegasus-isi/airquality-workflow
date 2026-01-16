#!/usr/bin/env python3

import json
from argparse import ArgumentParser
from typing import List


def merge_anomaly_results(input_files: List[str]) -> dict:
    """
    Merge multiple anomaly detection result files into a single output.

    Args:
        input_files: List of paths to JSON files containing anomaly results

    Returns:
        Merged dictionary with all anomaly results
    """
    merged_results = {
        'locations': {},
        'summary': {
            'total_locations': 0,
            'total_outliers': 0,
            'total_sustained_alerts': 0,
            'total_spikes': 0,
            'all_parameters': set()
        }
    }

    for input_file in input_files:
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)

            location = data.get('location', 'unknown')

            # Store location-specific results
            merged_results['locations'][location] = data

            # Update summary statistics
            if 'summary' in data:
                merged_results['summary']['total_outliers'] += data['summary'].get('total_outliers', 0)
                merged_results['summary']['total_sustained_alerts'] += data['summary'].get('total_sustained_alerts', 0)
                merged_results['summary']['total_spikes'] += data['summary'].get('total_spikes', 0)

                # Track unique parameters
                for param in data['summary'].get('parameters_affected', []):
                    merged_results['summary']['all_parameters'].add(param)

        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue

    # Convert set to list for JSON serialization
    merged_results['summary']['all_parameters'] = list(merged_results['summary']['all_parameters'])
    merged_results['summary']['total_locations'] = len(merged_results['locations'])

    return merged_results


def main():
    parser = ArgumentParser(description="Merge air quality anomaly detection results")
    parser.add_argument(
        "-i",
        "--input",
        metavar="INPUT_FILE",
        nargs='+',
        help="List of JSON files to be merged.",
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT_FILE",
        type=str,
        default="merged_anomalies.json",
        help="Output file name. Default is `merged_anomalies.json`."
    )

    args = parser.parse_args()

    print(f"Merging {len(args.input)} result files...")
    merged_results = merge_anomaly_results(args.input)

    with open(args.output, 'w') as f:
        json.dump(merged_results, f, indent=2)

    print(f"Merge complete. Results saved to {args.output}")
    print(f"Total locations: {merged_results['summary']['total_locations']}")
    print(f"Total outliers: {merged_results['summary']['total_outliers']}")
    print(f"Total sustained alerts: {merged_results['summary']['total_sustained_alerts']}")
    print(f"Total spikes: {merged_results['summary']['total_spikes']}")


if __name__ == "__main__":
    main()
