"""
Tests for SensorDataProcessor.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.data_processor import SensorDataProcessor


class TestSensorDataProcessor:
    """Test suite for SensorDataProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return SensorDataProcessor(
            window_size=10,
            stride=1,
            scaler_type='standard'
        )
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        n_samples = 100
        n_sensors = 5
        
        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        }
        
        for i in range(n_sensors):
            data[f'sensor_{i}'] = np.random.randn(n_samples) * 10 + 50
        
        return pd.DataFrame(data)
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.window_size == 10
        assert processor.stride == 1
        assert not processor.is_fitted
    
    def test_handle_missing_values_interpolate(self, processor):
        """Test missing value interpolation."""
        df = pd.DataFrame({
            'sensor_1': [1.0, np.nan, 3.0, 4.0],
            'sensor_2': [2.0, 3.0, np.nan, 5.0]
        })
        
        result = processor.handle_missing_values(df)
        assert result.isnull().sum().sum() == 0
    
    def test_normalize(self, processor):
        """Test data normalization."""
        data = pd.DataFrame({
            'sensor_1': [1, 2, 3, 4, 5],
            'sensor_2': [10, 20, 30, 40, 50]
        })
        
        normalized = processor.normalize(data, fit=True)
        
        # Check mean ~0 and std ~1
        assert np.abs(normalized.mean()) < 0.1
        assert np.abs(normalized.std() - 1.0) < 0.1
    
    def test_create_windows(self, processor):
        """Test window creation."""
        data = np.random.randn(100, 5)
        windows = processor.create_windows(data)
        
        # Check shape
        expected_n_windows = (100 - 10) // 1 + 1
        assert windows.shape == (expected_n_windows, 10, 5)
    
    def test_create_windows_custom_params(self, processor):
        """Test window creation with custom parameters."""
        data = np.random.randn(50, 3)
        windows = processor.create_windows(data, window_size=5, stride=2)
        
        expected_n_windows = (50 - 5) // 2 + 1
        assert windows.shape[0] == expected_n_windows
        assert windows.shape[1] == 5
        assert windows.shape[2] == 3
    
    def test_split_data(self, processor):
        """Test data splitting."""
        windows = np.random.randn(100, 10, 5)
        train, val, test = processor.split_data(
            windows,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
    
    def test_inverse_transform(self, processor):
        """Test inverse transformation."""
        original_data = pd.DataFrame({
            'sensor_1': [1, 2, 3, 4, 5],
            'sensor_2': [10, 20, 30, 40, 50]
        })
        
        # Normalize
        normalized = processor.normalize(original_data, fit=True)
        
        # Inverse transform
        reconstructed = processor.inverse_transform(normalized)
        
        # Check reconstruction
        np.testing.assert_array_almost_equal(
            reconstructed,
            original_data.values,
            decimal=5
        )
    
    def test_detect_outliers_iqr(self, processor):
        """Test IQR outlier detection."""
        df = pd.DataFrame({
            'sensor_1': [1, 2, 3, 4, 100],  # 100 is outlier
            'sensor_2': [10, 20, 30, 40, 50]
        })
        
        outlier_mask = processor.detect_outliers(df, method='iqr')
        
        # Check that outlier is detected
        assert outlier_mask['sensor_1'].iloc[-1]
    
    def test_streaming_data(self, processor):
        """Test streaming data processing."""
        # Fit processor first
        initial_data = pd.DataFrame({
            'sensor_1': np.random.randn(50)
        })
        processor.normalize(initial_data, fit=True)
        
        # Process streaming data
        new_data = np.random.randn(5, 1)
        windows, buffer = processor.process_streaming_data(new_data, buffer=None)
        
        # Check buffer management
        assert buffer is not None
        assert len(buffer) <= processor.window_size
