"""
Tests for LSTM Autoencoder model.
"""

import pytest
import torch
import numpy as np
from src.models.lstm_autoencoder import LSTMAutoencoder


class TestLSTMAutoencoder:
    """Test suite for LSTM Autoencoder."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return LSTMAutoencoder(
            input_dim=10,
            hidden_dim=32,
            latent_dim=16,
            num_layers=2,
            dropout=0.2
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        batch_size = 4
        seq_len = 50
        input_dim = 10
        return torch.randn(batch_size, seq_len, input_dim)
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.input_dim == 10
        assert model.hidden_dim == 32
        assert model.latent_dim == 16
        assert model.num_layers == 2
        assert model.dropout == 0.2
    
    def test_forward_pass(self, model, sample_data):
        """Test forward pass through the model."""
        output = model(sample_data)
        
        # Check output shape matches input shape
        assert output.shape == sample_data.shape
    
    def test_encode(self, model, sample_data):
        """Test encoding functionality."""
        latent, (hidden, cell) = model.encode(sample_data)
        
        # Check latent dimension
        assert latent.shape == (sample_data.size(0), model.latent_dim)
    
    def test_decode(self, model, sample_data):
        """Test decoding functionality."""
        batch_size = sample_data.size(0)
        seq_len = sample_data.size(1)
        
        latent, _ = model.encode(sample_data)
        reconstruction = model.decode(latent, seq_len)
        
        # Check reconstruction shape
        assert reconstruction.shape == sample_data.shape
    
    def test_reconstruction_error(self, model, sample_data):
        """Test reconstruction error calculation."""
        error = model.get_reconstruction_error(sample_data, reduction='none')
        
        # Check error shape (one value per sample)
        assert error.shape == (sample_data.size(0),)
        
        # Check all errors are non-negative
        assert (error >= 0).all()
    
    def test_bidirectional_model(self, sample_data):
        """Test bidirectional LSTM variant."""
        model = LSTMAutoencoder(
            input_dim=10,
            hidden_dim=32,
            latent_dim=16,
            num_layers=2,
            bidirectional=True
        )
        
        output = model(sample_data)
        assert output.shape == sample_data.shape
    
    def test_save_load(self, model, tmp_path):
        """Test model saving and loading."""
        # Save model
        model_path = tmp_path / "test_model.pth"
        model.save(str(model_path))
        
        # Load model
        loaded_model = LSTMAutoencoder.load(str(model_path))
        
        # Check parameters match
        assert loaded_model.input_dim == model.input_dim
        assert loaded_model.hidden_dim == model.hidden_dim
        assert loaded_model.latent_dim == model.latent_dim
    
    def test_different_sequence_lengths(self, model):
        """Test model with different sequence lengths."""
        batch_size = 4
        input_dim = 10
        
        for seq_len in [10, 50, 100, 200]:
            data = torch.randn(batch_size, seq_len, input_dim)
            output = model(data)
            assert output.shape == data.shape
    
    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow properly through the model."""
        # Forward pass
        output = model(sample_data)
        
        # Compute loss
        loss = torch.mean((output - sample_data) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
    
    def test_training_mode(self, model, sample_data):
        """Test model in training and evaluation modes."""
        # Training mode
        model.train()
        output_train = model(sample_data)
        
        # Evaluation mode
        model.eval()
        with torch.no_grad():
            output_eval = model(sample_data)
        
        # Outputs should be different due to dropout
        # (with high probability)
        assert not torch.allclose(output_train, output_eval)


class TestModelEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test with single sample batch."""
        model = LSTMAutoencoder(input_dim=5, hidden_dim=16)
        data = torch.randn(1, 20, 5)
        output = model(data)
        assert output.shape == data.shape
    
    def test_large_batch(self):
        """Test with large batch size."""
        model = LSTMAutoencoder(input_dim=5, hidden_dim=16)
        data = torch.randn(128, 20, 5)
        output = model(data)
        assert output.shape == data.shape
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises((ValueError, RuntimeError)):
            model = LSTMAutoencoder(input_dim=0, hidden_dim=16)
    
    def test_device_compatibility(self):
        """Test CPU/CUDA device compatibility."""
        model = LSTMAutoencoder(input_dim=5, hidden_dim=16)
        data = torch.randn(4, 20, 5)
        
        # CPU test
        model_cpu = model.to('cpu')
        data_cpu = data.to('cpu')
        output = model_cpu(data_cpu)
        assert output.device.type == 'cpu'
        
        # CUDA test (if available)
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            data_cuda = data.to('cuda')
            output = model_cuda(data_cuda)
            assert output.device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
