"""
LSTM Autoencoder for Time Series Anomaly Detection

This module implements a stacked LSTM autoencoder for unsupervised
anomaly detection in multivariate time series sensor data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time series anomaly detection.
    
    Architecture:
        Encoder: Stacked LSTM layers that compress input sequences
        Decoder: Stacked LSTM layers that reconstruct input sequences
    
    Args:
        input_dim (int): Number of input features (sensors)
        hidden_dim (int): Hidden dimension size
        latent_dim (int): Latent space dimension
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability
        bidirectional (bool): Use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Calculate multiplier for bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Latent space projection
        self.encoder_to_latent = nn.Linear(
            hidden_dim * self.num_directions,
            latent_dim
        )
        
        # Decoder
        self.latent_to_decoder = nn.Linear(
            latent_dim,
            hidden_dim
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        logger.info(f"Initialized LSTM Autoencoder: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, latent_dim={latent_dim}, "
                   f"num_layers={num_layers}")
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode input sequence to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            latent: Latent representation (batch_size, latent_dim)
            hidden: LSTM hidden states
        """
        # Pass through encoder LSTM
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Project to latent space
        latent = self.encoder_to_latent(hidden)
        latent = self.tanh(latent)
        
        return latent, (hidden, cell)
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation back to sequence.
        
        Args:
            latent: Latent representation (batch_size, latent_dim)
            seq_len: Length of sequence to reconstruct
            
        Returns:
            reconstruction: Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        batch_size = latent.size(0)
        
        # Repeat latent vector for each time step
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Pass through decoder LSTM
        lstm_out, _ = self.decoder_lstm(decoder_input)
        
        # Project to output dimension
        reconstruction = self.output_layer(lstm_out)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            reconstruction: Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        seq_len = x.size(1)
        
        # Encode
        latent, _ = self.encode(x)
        
        # Decode
        reconstruction = self.decode(latent, seq_len)
        
        return reconstruction
    
    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Calculate reconstruction error (anomaly score).
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            reduction: 'none', 'mean', or 'sum'
            
        Returns:
            error: Reconstruction error
        """
        reconstruction = self.forward(x)
        
        # Mean squared error
        error = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
        
        if reduction == 'mean':
            error = error.mean()
        elif reduction == 'sum':
            error = error.sum()
        
        return error
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        early_stopping: bool = True,
        patience: int = 10,
        validation_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> dict:
        """
        Train the autoencoder.
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to train on
            early_stopping: Enable early stopping
            patience: Early stopping patience
            validation_loader: Validation data loader
            
        Returns:
            history: Training history
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []
            
            for batch in train_loader:
                if isinstance(batch, list):
                    batch = batch[0]
                
                batch = batch.to(device)
                
                # Forward pass
                reconstruction = self.forward(batch)
                loss = criterion(reconstruction, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if validation_loader is not None:
                self.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch in validation_loader:
                        if isinstance(batch, list):
                            batch = batch[0]
                        
                        batch = batch.to(device)
                        reconstruction = self.forward(batch)
                        loss = criterion(reconstruction, batch)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # Early stopping
                if early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}")
        
        return history
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            latent_dim=checkpoint['latent_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            bidirectional=checkpoint['bidirectional']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model
