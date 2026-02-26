"""
Training script for LSTM Autoencoder

Usage:
    python src/train.py --data data/raw/sensor_data.csv --config configs/config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
from pathlib import Path

from models.lstm_autoencoder import LSTMAutoencoder
from preprocessing.data_processor import SensorDataProcessor
from utils.anomaly_detector import AnomalyDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int = 32
) -> tuple:
    """Create PyTorch data loaders."""
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader


def train(args):
    """Main training function."""
    logger.info("Starting training pipeline...")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        model_config = config.get('model', {})
        training_config = config.get('training', {})
    else:
        model_config = {}
        training_config = {}
    
    # Override config with command line args
    input_dim = args.input_dim or model_config.get('input_dim', 10)
    hidden_dim = args.hidden_dim or model_config.get('hidden_dim', 64)
    latent_dim = args.latent_dim or model_config.get('latent_dim', 32)
    num_layers = args.num_layers or model_config.get('num_layers', 2)
    dropout = args.dropout or model_config.get('dropout', 0.2)
    
    epochs = args.epochs or training_config.get('epochs', 50)
    batch_size = args.batch_size or training_config.get('batch_size', 32)
    learning_rate = args.learning_rate or training_config.get('learning_rate', 0.001)
    
    # Data preprocessing
    logger.info("Loading and preprocessing data...")
    processor = SensorDataProcessor(
        window_size=args.window_size,
        stride=args.stride,
        scaler_type=args.scaler_type
    )
    
    # Load and preprocess data
    normalized_data = processor.preprocess(
        filepath=args.data,
        remove_outliers=args.remove_outliers
    )
    
    # Create windows
    windows = processor.create_windows(normalized_data)
    
    # Split data
    train_windows, val_windows, test_windows = processor.split_data(
        windows,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Update input_dim based on actual data
    actual_input_dim = windows.shape[2]
    if input_dim != actual_input_dim:
        logger.warning(f"Updating input_dim from {input_dim} to {actual_input_dim}")
        input_dim = actual_input_dim
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_windows,
        val_windows,
        batch_size=batch_size
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=args.bidirectional
    )
    
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Train model
    logger.info("Training model...")
    history = model.fit(
        train_loader=train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        early_stopping=args.early_stopping,
        patience=args.patience,
        validation_loader=val_loader
    )
    
    # Calculate anomaly threshold
    logger.info("Calculating anomaly detection threshold...")
    detector = AnomalyDetector(model, threshold_percentile=args.threshold_percentile)
    
    train_tensor = torch.FloatTensor(train_windows)
    threshold = detector.calculate_threshold(train_tensor, device=device)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_tensor = torch.FloatTensor(test_windows)
    test_results = detector.detect(test_tensor, device=device)
    
    logger.info(f"Test set anomaly rate: {test_results['anomaly_rate']:.2%}")
    logger.info(f"Detected {test_results['n_anomalies']} anomalies in {len(test_windows)} windows")
    
    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    logger.info(f"Model saved to {args.output}")
    
    # Save preprocessor
    processor_path = args.output.replace('.pth', '_processor.pkl')
    import pickle
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    logger.info(f"Preprocessor saved to {processor_path}")
    
    # Save training history and threshold
    metadata = {
        'history': history,
        'threshold': float(threshold),
        'config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'window_size': args.window_size
        },
        'test_results': {
            'anomaly_rate': float(test_results['anomaly_rate']),
            'n_anomalies': int(test_results['n_anomalies'])
        }
    }
    
    metadata_path = args.output.replace('.pth', '_metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    logger.info(f"Metadata saved to {metadata_path}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder for Anomaly Detection')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default='models/lstm_autoencoder.pth',
                       help='Output path for trained model')
    
    # Model arguments
    parser.add_argument('--input-dim', type=int, default=None,
                       help='Input dimension (number of sensors)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden layer dimension')
    parser.add_argument('--latent-dim', type=int, default=None,
                       help='Latent space dimension')
    parser.add_argument('--num-layers', type=int, default=None,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Use bidirectional LSTM')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    # Preprocessing arguments
    parser.add_argument('--window-size', type=int, default=100,
                       help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1,
                       help='Sliding window stride')
    parser.add_argument('--scaler-type', type=str, default='standard',
                       choices=['standard', 'minmax'],
                       help='Type of scaler to use')
    parser.add_argument('--remove-outliers', action='store_true',
                       help='Remove outliers during preprocessing')
    
    # Detection arguments
    parser.add_argument('--threshold-percentile', type=float, default=95,
                       help='Percentile for anomaly threshold')
    
    args = parser.parse_args()
    train(args)
