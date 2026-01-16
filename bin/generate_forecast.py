#!/usr/bin/env python3

"""
Generate AQI forecasts using trained LSTM model.

This script loads a trained LSTM model and generates 24-hour AQI forecasts
with confidence intervals.

Usage:
    python generate_forecast.py --model models/location/location_lstm_checkpoint.pt \
                                --timeseries timeseries/location_timeseries.json \
                                --scaler features/location_train_scaler.json \
                                --output forecasts/location/location_forecast.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AQILSTMModel(nn.Module):
    """LSTM model for AQI forecasting (same as training script)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24
    ):
        super(AQILSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        predictions = self.fc(last_hidden)
        return predictions


def load_model_checkpoint(checkpoint_path: str) -> Tuple[nn.Module, Dict]:
    """
    Load trained LSTM model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file

    Returns:
        Tuple of (model, model_config)
    """
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model_config = checkpoint['model_config']

    model = AQILSTMModel(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        forecast_horizon=model_config['forecast_horizon']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"Input size: {model_config['input_size']}, Forecast horizon: {model_config['forecast_horizon']}")

    return model, model_config


def load_scaler_params(scaler_file: str) -> Dict:
    """
    Load normalization parameters.

    Args:
        scaler_file: Path to scaler JSON file

    Returns:
        Dictionary with scaler parameters
    """
    logger.info(f"Loading scaler from {scaler_file}")

    with open(scaler_file, 'r') as f:
        scaler_params = json.load(f)

    return scaler_params


def load_recent_timeseries(timeseries_file: str, lookback_hours: int = 168) -> pd.DataFrame:
    """
    Load recent timeseries data for inference.

    Args:
        timeseries_file: Path to timeseries JSON file
        lookback_hours: Number of recent hours to load

    Returns:
        DataFrame with recent measurements
    """
    logger.info(f"Loading recent data from {timeseries_file}")

    with open(timeseries_file, 'r') as f:
        timeseries_data = json.load(f)

    measurements = timeseries_data.get('measurements', [])

    if not measurements:
        raise ValueError("No measurements found in timeseries file")

    # Convert to DataFrame
    df = pd.DataFrame(measurements)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Sort by datetime and get most recent lookback_hours
    df = df.sort_values('datetime')
    cutoff_time = df['datetime'].max() - timedelta(hours=lookback_hours)
    df = df[df['datetime'] >= cutoff_time]

    logger.info(f"Loaded {len(df)} measurements")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def prepare_inference_input(
    df: pd.DataFrame,
    scaler_params: Dict,
    lookback: int = 168
) -> Tuple[torch.Tensor, datetime]:
    """
    Prepare input tensor for inference.

    Note: This is a simplified version. In production, you would replicate
    the exact feature engineering from prepare_features.py.

    Args:
        df: DataFrame with recent measurements
        scaler_params: Normalization parameters
        lookback: Number of timesteps to use

    Returns:
        Tuple of (input_tensor, forecast_start_time)
    """
    logger.info("Preparing inference input...")

    # For simplicity, use AQI values as primary feature
    # In production, replicate full feature engineering
    df = df.sort_values('datetime')

    # Get the most recent timestamp
    forecast_start_time = df['datetime'].max() + timedelta(hours=1)

    # Extract AQI values (assuming it's in the dataframe)
    if 'aqi' not in df.columns:
        logger.warning("AQI not in dataframe, using avg_concentration as proxy")
        aqi_col = 'avg_concentration'
    else:
        aqi_col = 'aqi'

    # Create hourly aggregates if needed
    df_hourly = df.groupby([pd.Grouper(key='datetime', freq='h'), 'parameter']).agg({
        aqi_col: 'mean'
    }).reset_index()

    # Pivot to wide format (one row per hour, one column per parameter)
    df_pivot = df_hourly.pivot_table(
        index='datetime',
        columns='parameter',
        values=aqi_col,
        aggfunc='mean'
    )

    # Fill missing values
    df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Get last lookback hours
    if len(df_pivot) < lookback:
        logger.warning(f"Only {len(df_pivot)} hours available, need {lookback}. Padding with zeros.")
        # Pad with zeros at the beginning
        padding = np.zeros((lookback - len(df_pivot), df_pivot.shape[1]))
        data = np.vstack([padding, df_pivot.values])
    else:
        data = df_pivot.values[-lookback:]

    # Normalize using scaler params
    X_mean = np.array(scaler_params['X_mean'])
    X_std = np.array(scaler_params['X_std'])

    # Handle dimension mismatch (scaler may have more features)
    if len(X_mean) > data.shape[1]:
        # Pad data with zeros
        padding = np.zeros((data.shape[0], len(X_mean) - data.shape[1]))
        data = np.hstack([data, padding])
    elif len(X_mean) < data.shape[1]:
        # Truncate data
        data = data[:, :len(X_mean)]

    data_normalized = (data - X_mean) / X_std

    # Convert to tensor (batch_size=1, seq_len, features)
    input_tensor = torch.FloatTensor(data_normalized).unsqueeze(0)

    logger.info(f"Input tensor shape: {input_tensor.shape}")
    logger.info(f"Forecast start time: {forecast_start_time}")

    return input_tensor, forecast_start_time


def generate_predictions(
    model: nn.Module,
    input_tensor: torch.Tensor,
    scaler_params: Dict
) -> np.ndarray:
    """
    Generate denormalized predictions.

    Args:
        model: Trained LSTM model
        input_tensor: Input features
        scaler_params: Normalization parameters

    Returns:
        Denormalized predictions
    """
    logger.info("Generating predictions...")

    with torch.no_grad():
        predictions_normalized = model(input_tensor)

    predictions_normalized = predictions_normalized.numpy()[0]

    # Denormalize
    y_mean = scaler_params['y_mean']
    y_std = scaler_params['y_std']

    predictions = predictions_normalized * y_std + y_mean

    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Predicted AQI range: {predictions.min():.1f} - {predictions.max():.1f}")

    return predictions


def calculate_confidence_intervals(
    predictions: np.ndarray,
    validation_mae: float = 10.0,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction confidence intervals.

    Uses validation MAE as estimate of prediction uncertainty.

    Args:
        predictions: Point predictions
        validation_mae: Mean absolute error from validation
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Simple confidence interval based on MAE
    # In production, use quantile regression or prediction intervals
    z_score = 1.96  # 95% confidence

    interval_width = z_score * validation_mae

    lower_bound = predictions - interval_width
    upper_bound = predictions + interval_width

    # Ensure non-negative AQI
    lower_bound = np.maximum(lower_bound, 0)

    return lower_bound, upper_bound


def classify_aqi_category(aqi: float) -> str:
    """
    Classify AQI value into EPA category.

    Args:
        aqi: AQI value

    Returns:
        Category string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def create_forecast_json(
    location_name: str,
    forecast_start_time: datetime,
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    model_config: Dict
) -> Dict:
    """
    Create forecast JSON output.

    Args:
        location_name: Location name
        forecast_start_time: Start time for forecast
        predictions: Predicted AQI values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        model_config: Model configuration

    Returns:
        Forecast dictionary
    """
    forecast_data = {
        "location": location_name,
        "forecast_generated": datetime.now(datetime.UTC if hasattr(datetime, 'UTC') else None).isoformat() if hasattr(datetime, 'UTC') else datetime.utcnow().isoformat(),
        "forecast_start": forecast_start_time.isoformat(),
        "forecast_horizon_hours": len(predictions),
        "model_info": {
            "architecture": "LSTM",
            "input_size": model_config.get('input_size', 0),
            "hidden_size": model_config.get('hidden_size', 0),
            "num_layers": model_config.get('num_layers', 0)
        },
        "predictions": []
    }

    for i, (pred, lower, upper) in enumerate(zip(predictions, lower_bound, upper_bound)):
        forecast_time = forecast_start_time + timedelta(hours=i)

        forecast_data["predictions"].append({
            "datetime": forecast_time.isoformat(),
            "predicted_aqi": float(pred),
            "confidence_interval_lower": float(lower),
            "confidence_interval_upper": float(upper),
            "predicted_category": classify_aqi_category(pred)
        })

    return forecast_data


def save_forecast(forecast_data: Dict, output_file: str):
    """
    Save forecast to JSON file.

    Args:
        forecast_data: Forecast dictionary
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(forecast_data, f, indent=2)

    logger.info(f"Forecast saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AQI forecasts using trained LSTM model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--timeseries",
        type=str,
        required=True,
        help="Path to recent timeseries JSON file"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        required=True,
        help="Path to scaler JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output forecast JSON file"
    )
    parser.add_argument(
        "--location-name",
        type=str,
        default="Unknown Location",
        help="Location name"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=168,
        help="Lookback hours (default: 168 = 7 days)"
    )
    parser.add_argument(
        "--validation-mae",
        type=float,
        default=10.0,
        help="Validation MAE for confidence intervals (default: 10.0)"
    )

    args = parser.parse_args()

    try:
        # Load model
        model, model_config = load_model_checkpoint(args.model)

        # Load scaler
        scaler_params = load_scaler_params(args.scaler)

        # Load recent data
        df = load_recent_timeseries(args.timeseries, args.lookback)

        # Prepare input
        input_tensor, forecast_start_time = prepare_inference_input(
            df, scaler_params, args.lookback
        )

        # Generate predictions
        predictions = generate_predictions(model, input_tensor, scaler_params)

        # Calculate confidence intervals
        lower_bound, upper_bound = calculate_confidence_intervals(
            predictions, args.validation_mae
        )

        # Create forecast JSON
        forecast_data = create_forecast_json(
            location_name=args.location_name,
            forecast_start_time=forecast_start_time,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_config=model_config
        )

        # Save forecast
        save_forecast(forecast_data, args.output)

        logger.info("Forecast generation completed successfully")
        logger.info(f"Predicted AQI range: {predictions.min():.1f} - {predictions.max():.1f}")

        # Log category distribution
        categories = [classify_aqi_category(p) for p in predictions]
        from collections import Counter
        category_counts = Counter(categories)
        logger.info("Predicted categories:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} hours")

    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
