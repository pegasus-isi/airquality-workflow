#!/usr/bin/env python3

"""
Train LSTM model for AQI forecasting.

This script trains a PyTorch-based LSTM model to predict future AQI values
based on historical air quality patterns.

Usage:
    python train_forecast_model.py --features features/location_train.npz \
                                   --output models/location/ \
                                   --epochs 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AQILSTMModel(nn.Module):
    """
    LSTM model for AQI forecasting.

    Architecture:
        - Multi-layer LSTM with dropout
        - Fully connected output layer
        - Predicts horizon timesteps of AQI values
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            forecast_horizon: Number of timesteps to predict
        """
        super(AQILSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Predictions of shape (batch, forecast_horizon)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]

        # Generate predictions
        predictions = self.fc(last_hidden)

        return predictions


def load_training_data(features_file: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load prepared training data.

    Args:
        features_file: Path to .npz file with training data

    Returns:
        Tuple of (X, y, metadata)
    """
    logger.info(f"Loading training data from {features_file}")

    data = np.load(features_file)

    X = data['X']  # (n_samples, lookback, n_features)
    y = data['y']  # (n_samples, horizon)

    metadata = {
        'feature_names': data['feature_names'].tolist() if 'feature_names' in data else [],
        'lookback': int(data['lookback']) if 'lookback' in data else X.shape[1],
        'horizon': int(data['horizon']) if 'horizon' in data else y.shape[1]
    }

    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Lookback: {metadata['lookback']}, Horizon: {metadata['horizon']}")

    return X, y, metadata


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.

    Args:
        X: Input features
        y: Target values
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    n_samples = len(X)
    split_idx = int(n_samples * (1 - val_split))

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    return X_train, X_val, y_train, y_val


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train model for one epoch.

    Args:
        model: LSTM model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model.

    Args:
        model: LSTM model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device (CPU/GPU)

    Returns:
        Tuple of (MSE loss, MAE loss)
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    mae_criterion = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            mse_loss = criterion(predictions, y_batch)
            mae_loss = mae_criterion(predictions, y_batch)

            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
            n_batches += 1

    avg_mse = total_mse / n_batches
    avg_mae = total_mae / n_batches

    return avg_mse, avg_mae


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Train LSTM model with early stopping.

    Args:
        model: LSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Device (CPU/GPU)

    Returns:
        Dictionary with training history
    """
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    history = {
        'train_loss': [],
        'val_mse': [],
        'val_mae': [],
        'learning_rate': []
    }

    logger.info("Starting training...")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, Learning rate: {learning_rate}, Patience: {patience}")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_mse, val_mae = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_mse)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val MSE: {val_mse:.4f}, "
            f"Val MAE: {val_mae:.4f}"
        )

        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"  New best model (Val MSE: {val_mse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model with Val MSE: {best_val_loss:.4f}")

    history['best_val_loss'] = best_val_loss
    history['epochs_trained'] = epoch + 1

    return history


def save_model(
    model: nn.Module,
    output_dir: str,
    location_name: str,
    history: Dict,
    metadata: Dict
):
    """
    Save trained model and training information.

    Args:
        model: Trained LSTM model
        output_dir: Output directory
        location_name: Location identifier
        history: Training history
        metadata: Model metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    checkpoint_file = output_path / f"{location_name}_lstm_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'forecast_horizon': model.forecast_horizon
        }
    }, checkpoint_file)
    logger.info(f"Model checkpoint saved to {checkpoint_file}")

    # Save training information
    training_info = {
        'model_architecture': 'LSTM',
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'forecast_horizon': model.forecast_horizon,
        'lookback': metadata.get('lookback', 168),
        'training_history': history,
        'best_val_loss': history['best_val_loss'],
        'epochs_trained': history['epochs_trained'],
        'feature_names': metadata.get('feature_names', [])
    }

    info_file = output_path / f"{location_name}_training_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    logger.info(f"Training info saved to {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for AQI forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to .npz file with prepared features"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for model and training info"
    )
    parser.add_argument(
        "--location-name",
        type=str,
        default="location",
        help="Location name (used for output filenames)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="LSTM hidden layer size (default: 128)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs (default: 100)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )

    args = parser.parse_args()

    try:
        # Load data
        X, y, metadata = load_training_data(args.features)

        if len(X) == 0:
            logger.error("No training data available. Exiting.")
            sys.exit(1)

        # Split train/validation
        X_train, X_val, y_train, y_val = split_train_val(X, y, args.val_split)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, args.batch_size
        )

        # Create model
        input_size = X.shape[2]  # Number of features
        forecast_horizon = y.shape[1]  # Number of timesteps to predict

        model = AQILSTMModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            forecast_horizon=forecast_horizon
        )

        logger.info(f"Model created: {input_size} features -> {forecast_horizon} predictions")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            device=device
        )

        # Save model
        save_model(model, args.output, args.location_name, history, metadata)

        logger.info("Training completed successfully")
        logger.info(f"Best validation MSE: {history['best_val_loss']:.4f}")
        logger.info(f"Best validation MAE: {min(history['val_mae']):.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
