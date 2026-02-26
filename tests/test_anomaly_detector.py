"""
Tests for AnomalyDetector.
"""

import pytest
import torch
import numpy as np
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.utils.anomaly_detector import AnomalyDetector


class TestAnomalyDetector:
    """Test suite for AnomalyDetector."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model."""
        return LSTMAutoencoder(
            input_dim=5,
            hidden_dim=16,
            latent_dim=8,
            num_layers=1
        )
    
    @pytest.fixture
    def detector(self, model):
        """Create detector instance."""
        return AnomalyDetector(
            model=model,
            threshold_percentile=95
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return torch.randn(100, 20, 5)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.threshold is None
        assert detector.threshold_percentile == 95
    
    def test_calculate_threshold_percentile(self, detector, sample_data):
        """Test threshold calculation using percentile method."""
        threshold = detector.calculate_threshold(
            sample_data,
            method='percentile'
        )
        
        assert threshold is not None
        assert threshold > 0
        assert detector.threshold == threshold
    
    def test_calculate_threshold_std(self, detector, sample_data):
        """Test threshold calculation using std method."""
        threshold = detector.calculate_threshold(
            sample_data,
            method='std'
        )
        
        assert threshold is not None
        assert threshold > 0
    
    def test_detect_anomalies(self, detector, sample_data):
        """Test anomaly detection."""
        # Calculate threshold first
        detector.calculate_threshold(sample_data[:80])
        
        # Detect on test data
        test_data = sample_data[80:]
        results = detector.detect(test_data)
        
        # Check results structure
        assert 'is_anomaly' in results
        assert 'reconstruction_errors' in results
        assert 'anomaly_scores' in results
        assert 'threshold' in results
        
        # Check shapes
        assert len(results['is_anomaly']) == len(test_data)
        assert len(results['reconstruction_errors']) == len(test_data)
    
    def test_anomaly_score_range(self, detector, sample_data):
        """Test that anomaly scores are in valid range."""
        detector.calculate_threshold(sample_data[:80])
        results = detector.detect(sample_data[80:])
        
        scores = results['anomaly_scores']
        
        # Scores should be between 0 and 1
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
    
    def test_filter_by_duration(self, detector):
        """Test anomaly filtering by duration."""
        # Create anomaly pattern: single, 2 consecutive, 5 consecutive
        is_anomaly = np.array([
            True,  # Single
            False, False,
            True, True,  # 2 consecutive
            False,
            True, True, True, True, True  # 5 consecutive
        ])
        
        # Filter with min_duration=3
        filtered = detector.filter_by_duration(is_anomaly, min_duration=3)
        
        # Only the 5-consecutive block should remain
        assert filtered.sum() == 5
        assert np.all(filtered[-5:])
    
    def test_feature_contributions(self, detector, sample_data):
        """Test feature contribution calculation."""
        detector.calculate_threshold(sample_data[:80])
        results = detector.detect(sample_data[80:], return_scores=True)
        
        contributions = results['feature_contributions']
        
        # Check shape
        assert contributions.shape[0] == len(sample_data[80:])
        assert contributions.shape[1] == 5  # Number of features
        
        # Contributions should sum to ~1 for each sample
        np.testing.assert_array_almost_equal(
            contributions.sum(axis=1),
            np.ones(len(contributions)),
            decimal=5
        )
    
    def test_identify_anomalous_features(self, detector):
        """Test identification of top contributing features."""
        contributions = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.05],
            [0.1, 0.1, 0.7, 0.05, 0.05]
        ])
        
        feature_names = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
        
        top_features = detector.identify_anomalous_features(
            contributions,
            top_k=2,
            feature_names=feature_names
        )
        
        # Check first sample top features
        assert top_features[0]['features'][0] == 'sensor_1'
        assert top_features[0]['features'][1] == 'sensor_2'
        
        # Check second sample top features
        assert top_features[1]['features'][0] == 'sensor_3'
    
    def test_get_anomaly_summary(self, detector, sample_data):
        """Test anomaly summary generation."""
        detector.calculate_threshold(sample_data[:80])
        results = detector.detect(sample_data[80:])
        
        summary = detector.get_anomaly_summary(results)
        
        # Check summary keys
        assert 'total_windows' in summary
        assert 'anomaly_count' in summary
        assert 'anomaly_rate' in summary
        assert 'threshold' in summary
    
    def test_no_anomalies_detected(self, detector, sample_data):
        """Test case with no anomalies."""
        # Set very high threshold
        detector.calculate_threshold(sample_data[:80])
        detector.threshold = 999999  # Very high
        
        results = detector.detect(sample_data[80:])
        
        assert results['n_anomalies'] == 0
        assert results['anomaly_rate'] == 0.0
