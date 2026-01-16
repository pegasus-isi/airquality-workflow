#!/usr/bin/env python3

"""
Prepare features for LSTM-based AQI forecasting.

This script transforms raw time series data into feature matrices suitable
for LSTM training, including temporal features, statistical features, and
sequence generation.

Usage:
    python prepare_features.py --timeseries timeseries/location_timeseries.json \
                               --historical historical/location_historical.csv \
                               --output features/location_train.npz
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_timeseries_data(timeseries_file: str, historical_file: str) -> pd.DataFrame:
    """
    Load and combine timeseries and historical data.

    Args:
        timeseries_file: Path to timeseries JSON file
        historical_file: Path to historical CSV file

    Returns:
        Combined DataFrame with all measurements
    """
    logger.info(f"Loading timeseries from {timeseries_file}")
    logger.info(f"Loading historical from {historical_file}")

    # Load historical CSV
    hist_df = pd.read_csv(historical_file)
    hist_df['datetime'] = pd.to_datetime(hist_df['datetime'], utc=True)

    # Ensure value column is numeric
    hist_df['value'] = pd.to_numeric(hist_df['value'], errors='coerce')

    # Load timeseries JSON (recent data)
    with open(timeseries_file, 'r') as f:
        timeseries_data = json.load(f)

    # Convert timeseries measurements to DataFrame
    ts_records = []
    for measurement in timeseries_data.get('measurements', []):
        ts_records.append({
            'datetime': measurement['datetime'],
            'parameter': measurement['parameter'],
            'value': measurement['avg_concentration'],
            'aqi': measurement['aqi'],
            'unit': measurement['unit']
        })

    if ts_records:
        ts_df = pd.DataFrame(ts_records)
        ts_df['datetime'] = pd.to_datetime(ts_df['datetime'], utc=True)

        # Ensure value column is numeric
        ts_df['value'] = pd.to_numeric(ts_df['value'], errors='coerce')
        ts_df['aqi'] = pd.to_numeric(ts_df['aqi'], errors='coerce')

        # Combine historical and recent data
        # Historical data has 'value' column for concentration
        combined_df = pd.concat([hist_df, ts_df], ignore_index=True)
    else:
        combined_df = hist_df

    # Sort by datetime and remove rows with NaN values
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    combined_df = combined_df.dropna(subset=['value'])

    logger.info(f"Loaded {len(combined_df)} measurements")
    logger.info(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    logger.info(f"Parameters: {combined_df['parameter'].unique().tolist()}")

    return combined_df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from datetime.

    Uses cyclic encoding for hour and month to preserve cyclical nature.

    Args:
        df: DataFrame with datetime column

    Returns:
        DataFrame with additional temporal features
    """
    logger.info("Creating temporal features...")

    # Extract basic temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclic encoding for hour (0-23) and month (1-12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Day of week cyclic encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def create_statistical_features(
    df: pd.DataFrame,
    windows: list = [6, 12, 24, 48]
) -> pd.DataFrame:
    """
    Create statistical features using rolling windows.

    Args:
        df: DataFrame with concentration and AQI values
        windows: List of window sizes (in hours) for rolling statistics

    Returns:
        DataFrame with additional statistical features
    """
    logger.info(f"Creating statistical features with windows: {windows}")

    # Ensure data is sorted by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    # Group by parameter and create rolling features
    stat_dfs = []

    for param in df['parameter'].unique():
        param_df = df[df['parameter'] == param].copy()

        # Ensure value column is numeric
        if 'value' not in param_df.columns or param_df['value'].isna().all():
            logger.warning(f"Skipping parameter {param}: no valid numeric values")
            continue

        # Create rolling features for each window
        for window in windows:
            # Rolling mean
            param_df[f'rolling_mean_{window}h'] = param_df['value'].rolling(
                window=window, min_periods=1
            ).mean()

            # Rolling std
            param_df[f'rolling_std_{window}h'] = param_df['value'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)

            # Rolling min/max
            param_df[f'rolling_min_{window}h'] = param_df['value'].rolling(
                window=window, min_periods=1
            ).min()

            param_df[f'rolling_max_{window}h'] = param_df['value'].rolling(
                window=window, min_periods=1
            ).max()

        stat_dfs.append(param_df)

    if not stat_dfs:
        raise ValueError("No valid parameters found for statistical feature creation")

    # Combine all parameters
    result_df = pd.concat(stat_dfs, ignore_index=True)
    result_df = result_df.sort_values('datetime').reset_index(drop=True)

    return result_df


def pivot_to_multivariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot parameter-wise data to create multivariate time series.

    Each row represents one hour with features from all parameters.

    Args:
        df: DataFrame with parameter-wise data

    Returns:
        Pivoted DataFrame with multivariate features
    """
    logger.info("Pivoting to multivariate format...")

    # Get numeric feature columns (exclude metadata and parameter)
    metadata_cols = ['datetime', 'location', 'location_id', 'sensor_id',
                     'city', 'country', 'unit', 'timestamp', 'hour_bucket', 'parameter']

    # Select only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]

    # If no numeric features, at least use 'value' and 'aqi' if they exist
    if not feature_cols:
        feature_cols = [col for col in ['value', 'aqi'] if col in df.columns]

    logger.info(f"Numeric feature columns: {feature_cols}")

    # Pivot each feature column
    pivoted_dfs = []

    # Group by datetime and parameter, then aggregate numeric columns
    agg_dict = {col: 'mean' for col in feature_cols if col in df.columns}

    hourly_df = df.groupby(['datetime', 'parameter']).agg(agg_dict).reset_index()

    # Pivot to wide format
    for col in feature_cols:
        try:
            pivot = hourly_df.pivot_table(
                index='datetime',
                columns='parameter',
                values=col,
                aggfunc='mean'
            )
            # Flatten column names
            pivot.columns = [f'{col}_{param}' for param in pivot.columns]
            pivoted_dfs.append(pivot)
        except Exception as e:
            logger.warning(f"Failed to pivot column {col}: {e}")
            continue

    if not pivoted_dfs:
        raise ValueError("No numeric features could be pivoted")

    # Combine all pivoted features
    result_df = pd.concat(pivoted_dfs, axis=1)
    result_df = result_df.reset_index()

    # Fill missing values (forward fill, then backward fill)
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')

    logger.info(f"Multivariate shape: {result_df.shape}")
    logger.info(f"Features: {len(result_df.columns) - 1}")  # -1 for datetime

    return result_df


def create_sequences(
    data: np.ndarray,
    lookback: int = 168,
    horizon: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for supervised learning.

    Args:
        data: Array of shape (timesteps, features)
        lookback: Number of past timesteps to use as input (default: 168 = 7 days)
        horizon: Number of future timesteps to predict (default: 24 = 1 day)

    Returns:
        Tuple of (X, y) where:
            X: shape (n_samples, lookback, features)
            y: shape (n_samples, horizon)
    """
    logger.info(f"Creating sequences with lookback={lookback}, horizon={horizon}")

    X, y = [], []

    for i in range(len(data) - lookback - horizon + 1):
        # Input: lookback timesteps with all features
        X.append(data[i:i + lookback, :])

        # Output: horizon timesteps of AQI (first feature, assuming AQI is first)
        # We'll predict average AQI across all parameters
        y.append(data[i + lookback:i + lookback + horizon, 0])  # AQI column

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Created {len(X)} sequences")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def normalize_features(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Normalize features using z-score normalization.

    Args:
        X: Input features (n_samples, lookback, n_features)
        y: Target values (n_samples, horizon)

    Returns:
        Tuple of (X_normalized, y_normalized, scaler_params)
    """
    logger.info("Normalizing features...")

    # Flatten X for computing statistics
    n_samples, lookback, n_features = X.shape
    X_flat = X.reshape(-1, n_features)

    # Compute mean and std for each feature
    X_mean = X_flat.mean(axis=0)
    X_std = X_flat.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero

    # Normalize X
    X_normalized = (X - X_mean) / X_std

    # Normalize y
    y_mean = y.mean()
    y_std = y.std()
    y_std = y_std if y_std > 0 else 1
    y_normalized = (y - y_mean) / y_std

    scaler_params = {
        'X_mean': X_mean.tolist(),
        'X_std': X_std.tolist(),
        'y_mean': float(y_mean),
        'y_std': float(y_std)
    }

    logger.info(f"Normalization complete. y_mean={y_mean:.2f}, y_std={y_std:.2f}")

    return X_normalized, y_normalized, scaler_params


def main():
    parser = argparse.ArgumentParser(
        description="Prepare features for LSTM training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--timeseries",
        type=str,
        required=True,
        help="Path to timeseries JSON file"
    )
    parser.add_argument(
        "--historical",
        type=str,
        required=True,
        help="Path to historical CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz file path for training data"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=168,
        help="Number of past hours to use as input (default: 168 = 7 days)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of future hours to predict (default: 24)"
    )

    args = parser.parse_args()

    try:
        # Load data
        df = load_timeseries_data(args.timeseries, args.historical)

        if df.empty:
            logger.error("No data loaded. Exiting.")
            sys.exit(1)

        # Create temporal features
        df = create_temporal_features(df)

        # Create statistical features
        df = create_statistical_features(df, windows=[6, 12, 24, 48])

        # Pivot to multivariate format
        mvdf = pivot_to_multivariate(df)

        # Extract feature matrix (exclude datetime)
        feature_cols = [col for col in mvdf.columns if col != 'datetime']
        data = mvdf[feature_cols].values

        logger.info(f"Final feature matrix shape: {data.shape}")

        # Check for sufficient data
        min_required = args.lookback + args.horizon
        if len(data) < min_required:
            logger.error(
                f"Insufficient data for sequence creation. "
                f"Need at least {min_required} hours, but only have {len(data)}"
            )
            sys.exit(1)

        # Create sequences
        X, y = create_sequences(data, lookback=args.lookback, horizon=args.horizon)

        if len(X) == 0:
            logger.error("No sequences created. Insufficient data.")
            sys.exit(1)

        # Normalize
        X_norm, y_norm, scaler_params = normalize_features(X, y)

        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save training data
        np.savez_compressed(
            args.output,
            X=X_norm,
            y=y_norm,
            feature_names=feature_cols,
            lookback=args.lookback,
            horizon=args.horizon
        )
        logger.info(f"Training data saved to {args.output}")

        # Save scaler parameters
        scaler_file = output_path.parent / f"{output_path.stem}_scaler.json"
        with open(scaler_file, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        logger.info(f"Scaler parameters saved to {scaler_file}")

        # Log statistics
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Features: {len(feature_cols)}")

        logger.info("Feature preparation completed successfully")

    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
