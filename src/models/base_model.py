"""
Base model class for all anomaly detection models.

Provides common interface and functionality.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAnomalyModel(ABC, nn.Module):
    """
    Abstract base class for anomaly detection models.
    
    All anomaly detection models should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        super(BaseAnomalyModel, self).__init__()
        self.model_name = model_name
        self._is_trained = False
        self.metadata = {}
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Model output
        """
        pass
    
    @abstractmethod
    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Calculate reconstruction error (anomaly score).
        
        Args:
            x: Input tensor
            reduction: How to reduce errors ('none', 'mean', 'sum')
            
        Returns:
            error: Reconstruction error
        """
        pass
    
    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            **kwargs: Additional training arguments
            
        Returns:
            history: Training history
        """
        logger.info(f"Training {self.model_name}...")
        # Implementation in subclass
        pass
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_loader: Test data loader
            device: Device to evaluate on
            
        Returns:
            metrics: Evaluation metrics
        """
        self.eval()
        self.to(device)
        
        total_error = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, list):
                    batch = batch[0]
                
                batch = batch.to(device)
                error = self.get_reconstruction_error(batch, reduction='sum')
                
                total_error += error.item()
                n_samples += len(batch)
        
        avg_error = total_error / n_samples
        
        metrics = {
            'avg_reconstruction_error': avg_error,
            'total_samples': n_samples
        }
        
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
    
    def save_model(self, path: str, **kwargs):
        """
        Save model to file.
        
        Args:
            path: Path to save model
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.state_dict(),
            'metadata': {**self.metadata, **kwargs},
            'is_trained': self._is_trained
        }
        
        torch.save(checkpoint, path)
        logger.info(f"{self.model_name} saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """
        Load model from file.
        
        Args:
            path: Path to model file
            device: Device to load model on
            
        Returns:
            model: Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # This should be implemented by subclass
        logger.info(f"Loading model from {path}")
        return checkpoint
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            n_params: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_layers(self, layer_names: list):
        """
        Freeze specific layers.
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen")
    
    def get_device(self) -> torch.device:
        """
        Get the device the model is on.
        
        Returns:
            device: Current device
        """
        return next(self.parameters()).device
    
    def set_metadata(self, **kwargs):
        """
        Set model metadata.
        
        Args:
            **kwargs: Metadata key-value pairs
        """
        self.metadata.update(kwargs)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            metadata: Model metadata
        """
        return self.metadata.copy()
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
    
    def __repr__(self) -> str:
        """String representation of the model."""
        n_params = self.get_num_parameters()
        return (f"{self.model_name}("
                f"parameters={n_params:,}, "
                f"trained={self._is_trained})")
