"""
Feature engineering for sensor time series data.

Provides advanced feature extraction and transformation methods.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Extract and engineer features from time series sensor data.
    
    Features include:
    - Statistical features (mean, std, min, max, etc.)
    - Frequency domain features (FFT)
    - Temporal features (rolling statistics)
    - Domain-specific features
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize feature engineer.
        
        Args:
            window_size: Size of rolling window for features
        """
        self.window_size = window_size
        logger.info(f"Initialized FeatureEngineer with window_size={window_size}")
    
    def extract_statistical_features(
        self,
        data: np.ndarray,
        axis: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from data.
        
        Args:
            data: Input data (samples, features) or (samples, window, features)
            axis: Axis along which to compute statistics
            
        Returns:
            features: Dictionary of statistical features
        """
        features = {
            'mean': np.mean(data, axis=axis),
            'std': np.std(data, axis=axis),
            'min': np.min(data, axis=axis),
            'max': np.max(data, axis=axis),
            'median': np.median(data, axis=axis),
            'range': np.ptp(data, axis=axis),
            'iqr': stats.iqr(data, axis=axis),
            'skew': stats.skew(data, axis=axis),
            'kurtosis': stats.kurtosis(data, axis=axis)
        }
        
        # Percentiles
        for p in [25, 75, 90]:
            features[f'p{p}'] = np.percentile(data, p, axis=axis)
        
        return features
    
    def extract_frequency_features(
        self,
        data: np.ndarray,
        sampling_rate: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Extract frequency domain features using FFT.
        
        Args:
            data: Input data (samples, window_size)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            features: Dictionary of frequency features
        """
        # Compute FFT
        fft_vals = np.fft.fft(data, axis=1)
        fft_freq = np.fft.fftfreq(data.shape[1], 1/sampling_rate)
        
        # Get magnitude
        fft_mag = np.abs(fft_vals)
        
        # Only positive frequencies
        pos_freq_idx = fft_freq > 0
        fft_mag_pos = fft_mag[:, pos_freq_idx]
        fft_freq_pos = fft_freq[pos_freq_idx]
        
        features = {
            'dominant_freq': fft_freq_pos[np.argmax(fft_mag_pos, axis=1)],
            'spectral_centroid': np.sum(
                fft_mag_pos * fft_freq_pos, axis=1
            ) / np.sum(fft_mag_pos, axis=1),
            'spectral_energy': np.sum(fft_mag_pos ** 2, axis=1),
            'spectral_entropy': stats.entropy(fft_mag_pos, axis=1)
        }
        
        return features
    
    def extract_temporal_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Extract rolling window temporal features.
        
        Args:
            df: Input DataFrame
            windows: List of window sizes
            
        Returns:
            features_df: DataFrame with temporal features
        """
        features_list = []
        
        for col in df.columns:
            for window in windows:
                # Rolling statistics
                features_list.append(
                    df[col].rolling(window).mean().rename(f'{col}_rolling_mean_{window}')
                )
                features_list.append(
                    df[col].rolling(window).std().rename(f'{col}_rolling_std_{window}')
                )
                
                # Rate of change
                features_list.append(
                    df[col].diff(window).rename(f'{col}_diff_{window}')
                )
        
        features_df = pd.concat(features_list, axis=1)
        return features_df
    
    def extract_change_point_features(
        self,
        data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Detect change points and extract related features.
        
        Args:
            data: Input data (samples, window_size)
            
        Returns:
            features: Change point features
        """
        # First derivative (rate of change)
        first_diff = np.diff(data, axis=1)
        
        # Second derivative (acceleration)
        second_diff = np.diff(first_diff, axis=1)
        
        features = {
            'mean_rate_of_change': np.mean(np.abs(first_diff), axis=1),
            'max_rate_of_change': np.max(np.abs(first_diff), axis=1),
            'mean_acceleration': np.mean(np.abs(second_diff), axis=1),
            'max_acceleration': np.max(np.abs(second_diff), axis=1),
            'num_sign_changes': np.sum(np.diff(np.sign(first_diff), axis=1) != 0, axis=1)
        }
        
        return features
    
    def extract_autocorrelation_features(
        self,
        data: np.ndarray,
        max_lag: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Extract autocorrelation features.
        
        Args:
            data: Input data (samples, window_size)
            max_lag: Maximum lag to compute
            
        Returns:
            features: Autocorrelation features
        """
        n_samples, window_size = data.shape
        
        autocorr_values = np.zeros((n_samples, max_lag))
        
        for i in range(n_samples):
            for lag in range(max_lag):
                if lag < window_size - 1:
                    autocorr_values[i, lag] = np.corrcoef(
                        data[i, :-lag-1] if lag > 0 else data[i],
                        data[i, lag+1:] if lag > 0 else data[i]
                    )[0, 1] if lag > 0 else 1.0
        
        features = {
            'autocorr_mean': np.mean(autocorr_values, axis=1),
            'autocorr_max': np.max(autocorr_values, axis=1),
            'autocorr_lag1': autocorr_values[:, 1] if max_lag > 1 else np.zeros(n_samples)
        }
        
        return features
    
    def extract_domain_specific_features(
        self,
        data: np.ndarray,
        feature_type: str = 'vibration'
    ) -> Dict[str, np.ndarray]:
        """
        Extract domain-specific features.
        
        Args:
            data: Input data
            feature_type: Type of sensor ('vibration', 'temperature', 'pressure')
            
        Returns:
            features: Domain-specific features
        """
        features = {}
        
        if feature_type == 'vibration':
            # RMS (Root Mean Square)
            features['rms'] = np.sqrt(np.mean(data ** 2, axis=1))
            
            # Peak-to-peak
            features['peak_to_peak'] = np.ptp(data, axis=1)
            
            # Crest factor
            features['crest_factor'] = np.max(np.abs(data), axis=1) / features['rms']
            
            # Shape factor
            features['shape_factor'] = features['rms'] / np.mean(np.abs(data), axis=1)
        
        elif feature_type == 'temperature':
            # Temperature-specific features
            features['temp_variance'] = np.var(data, axis=1)
            features['temp_trend'] = np.polyfit(
                np.arange(data.shape[1]), data.T, 1
            )[0]
        
        elif feature_type == 'pressure':
            # Pressure-specific features
            features['pressure_variance'] = np.var(data, axis=1)
            features['pressure_cycles'] = self._count_cycles(data)
        
        return features
    
    def _count_cycles(self, data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Count number of cycles in signal.
        
        Args:
            data: Input data
            threshold: Threshold for peak detection
            
        Returns:
            cycle_counts: Number of cycles per sample
        """
        cycle_counts = np.zeros(len(data))
        
        for i, signal_data in enumerate(data):
            # Normalize
            normalized = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
            
            # Find peaks
            peaks, _ = signal.find_peaks(normalized, height=threshold)
            cycle_counts[i] = len(peaks)
        
        return cycle_counts
    
    def extract_all_features(
        self,
        data: np.ndarray,
        feature_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract all available features.
        
        Args:
            data: Input data (samples, window_size, features)
            feature_types: List of feature types to extract
            
        Returns:
            all_features: Combined feature array
        """
        if feature_types is None:
            feature_types = ['statistical', 'frequency', 'change_point']
        
        all_features = []
        
        # Process each sensor
        for sensor_idx in range(data.shape[2]):
            sensor_data = data[:, :, sensor_idx]
            
            if 'statistical' in feature_types:
                stat_features = self.extract_statistical_features(sensor_data)
                all_features.extend(stat_features.values())
            
            if 'frequency' in feature_types:
                freq_features = self.extract_frequency_features(sensor_data)
                all_features.extend(freq_features.values())
            
            if 'change_point' in feature_types:
                change_features = self.extract_change_point_features(sensor_data)
                all_features.extend(change_features.values())
            
            if 'autocorrelation' in feature_types:
                autocorr_features = self.extract_autocorrelation_features(sensor_data)
                all_features.extend(autocorr_features.values())
        
        # Stack all features
        feature_array = np.column_stack(all_features)
        
        logger.info(f"Extracted {feature_array.shape[1]} features from {data.shape[2]} sensors")
        return feature_array
