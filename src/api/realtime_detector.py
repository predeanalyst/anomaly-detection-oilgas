"""
Real-time anomaly detector API module.

Provides interface for real-time anomaly detection.
"""

import torch
import pickle
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path

from ..models.lstm_autoencoder import LSTMAutoencoder
from ..preprocessing.data_processor import SensorDataProcessor
from ..utils.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """
    Real-time anomaly detector for streaming sensor data.
    
    Handles:
    - Model loading
    - Data buffering
    - Real-time inference
    - Result formatting
    """
    
    def __init__(
        self,
        model_path: str,
        processor_path: Optional[str] = None,
        threshold: Optional[float] = None,
        device: str = 'cpu'
    ):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model
            processor_path: Path to data processor
            threshold: Anomaly detection threshold
            device: Device to run inference on
        """
        self.device = device
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = LSTMAutoencoder.load(model_path, device=device)
        self.model.eval()
        
        # Load processor
        if processor_path is None:
            processor_path = model_path.replace('.pth', '_processor.pkl')
        
        logger.info(f"Loading processor from {processor_path}")
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        # Initialize detector
        self.detector = AnomalyDetector(self.model)
        
        if threshold is not None:
            self.detector.threshold = threshold
        
        # Initialize buffer
        self.buffer = None
        
        logger.info("RealtimeDetector initialized successfully")
    
    def predict(
        self,
        sensor_data: np.ndarray,
        sensor_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Predict anomalies in new sensor data.
        
        Args:
            sensor_data: New sensor readings (n_samples, n_features)
            sensor_names: Names of sensors
            
        Returns:
            result: Detection results
        """
        # Process streaming data
        windows, self.buffer = self.processor.process_streaming_data(
            sensor_data,
            buffer=self.buffer
        )
        
        if windows is None:
            return {
                'status': 'buffering',
                'message': 'Collecting more data for full window',
                'buffer_size': len(self.buffer)
            }
        
        # Convert to tensor
        windows_tensor = torch.FloatTensor(windows).to(self.device)
        
        # Detect anomalies
        results = self.detector.detect(windows_tensor, device=self.device)
        
        # Format results
        formatted_results = self._format_results(
            results,
            sensor_names=sensor_names
        )
        
        return formatted_results
    
    def predict_batch(
        self,
        batch_data: List[np.ndarray],
        sensor_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Predict anomalies for multiple batches.
        
        Args:
            batch_data: List of sensor data arrays
            sensor_names: Names of sensors
            
        Returns:
            results: List of detection results
        """
        results = []
        
        for data in batch_data:
            result = self.predict(data, sensor_names=sensor_names)
            results.append(result)
        
        return results
    
    def _format_results(
        self,
        results: Dict,
        sensor_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Format detection results for API response.
        
        Args:
            results: Raw detection results
            sensor_names: Names of sensors
            
        Returns:
            formatted: Formatted results
        """
        is_anomaly = results['is_anomaly']
        scores = results['anomaly_scores']
        errors = results['reconstruction_errors']
        contributions = results.get('feature_contributions')
        
        predictions = []
        
        for i in range(len(is_anomaly)):
            pred = {
                'index': int(i),
                'is_anomaly': bool(is_anomaly[i]),
                'anomaly_score': float(scores[i]),
                'reconstruction_error': float(errors[i])
            }
            
            # Add feature contributions if anomaly
            if is_anomaly[i] and contributions is not None:
                top_features = self.detector.identify_anomalous_features(
                    contributions[i:i+1],
                    top_k=3,
                    feature_names=sensor_names
                )
                pred['anomalous_features'] = top_features[0]
            
            predictions.append(pred)
        
        return {
            'status': 'success',
            'predictions': predictions,
            'threshold': float(self.detector.threshold),
            'summary': {
                'total_windows': len(is_anomaly),
                'anomaly_count': int(is_anomaly.sum()),
                'anomaly_rate': float(is_anomaly.mean())
            }
        }
    
    def reset_buffer(self):
        """Reset data buffer."""
        self.buffer = None
        logger.info("Buffer reset")
    
    def update_threshold(self, threshold: float):
        """
        Update detection threshold.
        
        Args:
            threshold: New threshold value
        """
        self.detector.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            info: Model information
        """
        return {
            'model_name': 'LSTM Autoencoder',
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'latent_dim': self.model.latent_dim,
            'num_layers': self.model.num_layers,
            'threshold': float(self.detector.threshold) if self.detector.threshold else None,
            'device': str(self.device)
        }
