"""
Data preprocessing module for sensor time series data.

Handles data loading, cleaning, normalization, and windowing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class SensorDataProcessor:
    """
    Preprocess sensor data for LSTM autoencoder training and inference.
    
    Features:
    - Load data from various sources
    - Handle missing values
    - Normalize/standardize features
    - Create sliding windows
    - Split train/validation/test sets
    """
    
    def __init__(
        self,
        window_size: int = 100,
        stride: int = 1,
        scaler_type: str = 'standard',
        handle_missing: str = 'interpolate'
    ):
        """
        Initialize data processor.
        
        Args:
            window_size: Size of sliding window
            stride: Step size for sliding window
            scaler_type: 'standard' or 'minmax'
            handle_missing: 'drop', 'interpolate', or 'forward_fill'
        """
        self.window_size = window_size
        self.stride = stride
        self.handle_missing = handle_missing
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
        logger.info(f"Initialized SensorDataProcessor with window_size={window_size}, "
                   f"stride={stride}, scaler={scaler_type}")
    
    def load_data(
        self,
        filepath: str,
        timestamp_col: str = 'timestamp',
        sensor_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load sensor data from CSV file.
        
        Args:
            filepath: Path to CSV file
            timestamp_col: Name of timestamp column
            sensor_cols: List of sensor column names (None = all except timestamp)
            
        Returns:
            df: Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Parse timestamp
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)
            df = df.set_index(timestamp_col)
        
        # Select sensor columns
        if sensor_cols is not None:
            df = df[sensor_cols]
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} sensors")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            df: DataFrame with missing values handled
        """
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            return df
        
        logger.warning(f"Found {missing_count} missing values")
        
        if self.handle_missing == 'drop':
            df = df.dropna()
            logger.info("Dropped rows with missing values")
        
        elif self.handle_missing == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
            logger.info("Interpolated missing values")
        
        elif self.handle_missing == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
            logger.info("Forward-filled missing values")
        
        return df
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect and optionally remove outliers.
        
        Args:
            df: Input DataFrame
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            outlier_mask: Boolean DataFrame indicating outliers
        """
        if method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
        
        elif method == 'zscore':
            z_scores = np.abs((df - df.mean()) / df.std())
            outlier_mask = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outlier_count = outlier_mask.sum().sum()
        logger.info(f"Detected {outlier_count} outliers using {method} method")
        
        return outlier_mask
    
    def normalize(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Normalize/standardize the data.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            normalized_data: Normalized numpy array
        """
        if fit:
            normalized_data = self.scaler.fit_transform(df.values)
            self.is_fitted = True
            logger.info("Fitted and transformed data")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized_data = self.scaler.transform(df.values)
            logger.info("Transformed data using existing scaler")
        
        return normalized_data
    
    def create_windows(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
        stride: Optional[int] = None
    ) -> np.ndarray:
        """
        Create sliding windows from time series data.
        
        Args:
            data: Input array of shape (n_samples, n_features)
            window_size: Size of each window (default: self.window_size)
            stride: Stride between windows (default: self.stride)
            
        Returns:
            windows: Array of shape (n_windows, window_size, n_features)
        """
        window_size = window_size or self.window_size
        stride = stride or self.stride
        
        n_samples, n_features = data.shape
        n_windows = (n_samples - window_size) // stride + 1
        
        windows = np.zeros((n_windows, window_size, n_features))
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windows[i] = data[start_idx:end_idx]
        
        logger.info(f"Created {n_windows} windows of size {window_size}")
        return windows
    
    def split_data(
        self,
        windows: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split windowed data into train/val/test sets.
        
        Args:
            windows: Input windows
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            shuffle: Whether to shuffle before splitting
            
        Returns:
            train, val, test: Split datasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        n_windows = len(windows)
        
        if shuffle:
            indices = np.random.permutation(n_windows)
            windows = windows[indices]
        
        train_end = int(n_windows * train_ratio)
        val_end = int(n_windows * (train_ratio + val_ratio))
        
        train = windows[:train_end]
        val = windows[train_end:val_end]
        test = windows[val_end:]
        
        logger.info(f"Split data: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
    
    def preprocess(
        self,
        filepath: str,
        timestamp_col: str = 'timestamp',
        sensor_cols: Optional[List[str]] = None,
        remove_outliers: bool = False,
        outlier_method: str = 'iqr',
        return_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Full preprocessing pipeline.
        
        Args:
            filepath: Path to data file
            timestamp_col: Timestamp column name
            sensor_cols: Sensor column names
            remove_outliers: Whether to remove outliers
            outlier_method: Outlier detection method
            return_dataframe: Return DataFrame instead of array
            
        Returns:
            processed_data: Preprocessed data
        """
        # Load data
        df = self.load_data(filepath, timestamp_col, sensor_cols)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers if requested
        if remove_outliers:
            outlier_mask = self.detect_outliers(df, method=outlier_method)
            # Replace outliers with interpolated values
            df[outlier_mask] = np.nan
            df = self.handle_missing_values(df)
        
        if return_dataframe:
            return df
        
        # Normalize
        normalized_data = self.normalize(df, fit=True)
        
        return normalized_data
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            normalized_data: Normalized data
            
        Returns:
            original_scale_data: Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")
        
        return self.scaler.inverse_transform(normalized_data)
    
    def process_streaming_data(
        self,
        new_data: Union[np.ndarray, pd.DataFrame],
        buffer: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process streaming data for real-time inference.
        
        Args:
            new_data: New sensor readings
            buffer: Existing data buffer
            
        Returns:
            windows: New windows if enough data
            updated_buffer: Updated buffer
        """
        # Convert to array if DataFrame
        if isinstance(new_data, pd.DataFrame):
            new_data = new_data.values
        
        # Normalize
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Run preprocess() first.")
        
        normalized_new = self.scaler.transform(new_data)
        
        # Update buffer
        if buffer is None:
            buffer = normalized_new
        else:
            buffer = np.vstack([buffer, normalized_new])
        
        # Create windows if enough data
        windows = None
        if len(buffer) >= self.window_size:
            windows = self.create_windows(buffer)
            # Keep only the last window_size-1 samples in buffer
            buffer = buffer[-(self.window_size - 1):]
        
        return windows, buffer
