"""
Anomaly detection and scoring module.

Determines anomalies based on reconstruction errors and provides
detailed anomaly analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detect anomalies using reconstruction error from LSTM autoencoder.
    
    Features:
    - Dynamic threshold calculation
    - Feature-level anomaly contribution
    - Anomaly severity scoring
    - Temporal context analysis
    """
    
    def __init__(
        self,
        model,
        threshold_percentile: float = 95,
        contamination: float = 0.05,
        min_anomaly_duration: int = 1
    ):
        """
        Initialize anomaly detector.
        
        Args:
            model: Trained LSTM autoencoder model
            threshold_percentile: Percentile for threshold calculation
            contamination: Expected proportion of anomalies
            min_anomaly_duration: Minimum consecutive anomalous windows
        """
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.contamination = contamination
        self.min_anomaly_duration = min_anomaly_duration
        
        self.threshold = None
        self.baseline_errors = None
        
        logger.info(f"Initialized AnomalyDetector with threshold_percentile={threshold_percentile}")
    
    def calculate_threshold(
        self,
        train_data: torch.Tensor,
        method: str = 'percentile',
        device: str = 'cpu'
    ) -> float:
        """
        Calculate anomaly threshold from training data.
        
        Args:
            train_data: Training dataset
            method: 'percentile', 'std', or 'mad'
            device: Device to run on
            
        Returns:
            threshold: Calculated threshold value
        """
        self.model.eval()
        self.model.to(device)
        
        errors = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = 32
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size].to(device)
                error = self.model.get_reconstruction_error(batch, reduction='none')
                errors.extend(error.cpu().numpy())
        
        errors = np.array(errors)
        self.baseline_errors = errors
        
        if method == 'percentile':
            threshold = np.percentile(errors, self.threshold_percentile)
        
        elif method == 'std':
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            n_std = stats.norm.ppf(1 - self.contamination)
            threshold = mean_error + n_std * std_error
        
        elif method == 'mad':
            # Median Absolute Deviation
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))
            threshold = median_error + 3 * mad
        
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        self.threshold = threshold
        logger.info(f"Calculated threshold: {threshold:.6f} using {method} method")
        
        return threshold
    
    def detect(
        self,
        data: torch.Tensor,
        device: str = 'cpu',
        return_scores: bool = True
    ) -> Dict:
        """
        Detect anomalies in data.
        
        Args:
            data: Input data tensor
            device: Device to run on
            return_scores: Return detailed scores
            
        Returns:
            results: Dictionary with anomaly detection results
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call calculate_threshold() first.")
        
        self.model.eval()
        self.model.to(device)
        data = data.to(device)
        
        with torch.no_grad():
            # Get reconstruction errors
            reconstruction_errors = self.model.get_reconstruction_error(
                data, reduction='none'
            ).cpu().numpy()
            
            # Get reconstructions for feature-level analysis
            reconstructions = self.model.forward(data).cpu().numpy()
            original_data = data.cpu().numpy()
        
        # Detect anomalies
        is_anomaly = reconstruction_errors > self.threshold
        
        # Calculate anomaly scores (normalized)
        anomaly_scores = self.calculate_anomaly_score(reconstruction_errors)
        
        # Feature contributions
        feature_contributions = self.calculate_feature_contributions(
            original_data, reconstructions
        )
        
        results = {
            'is_anomaly': is_anomaly,
            'reconstruction_errors': reconstruction_errors,
            'anomaly_scores': anomaly_scores,
            'threshold': self.threshold,
            'n_anomalies': int(is_anomaly.sum()),
            'anomaly_rate': float(is_anomaly.mean())
        }
        
        if return_scores:
            results['feature_contributions'] = feature_contributions
        
        return results
    
    def calculate_anomaly_score(
        self,
        reconstruction_errors: np.ndarray
    ) -> np.ndarray:
        """
        Calculate normalized anomaly scores (0-1 scale).
        
        Args:
            reconstruction_errors: Raw reconstruction errors
            
        Returns:
            scores: Normalized anomaly scores
        """
        # Use baseline statistics for normalization
        if self.baseline_errors is not None:
            mean = self.baseline_errors.mean()
            std = self.baseline_errors.std()
            
            # Z-score normalization
            z_scores = (reconstruction_errors - mean) / (std + 1e-8)
            
            # Convert to 0-1 scale using sigmoid
            scores = 1 / (1 + np.exp(-z_scores))
        else:
            # Simple min-max normalization
            min_error = reconstruction_errors.min()
            max_error = reconstruction_errors.max()
            scores = (reconstruction_errors - min_error) / (max_error - min_error + 1e-8)
        
        return scores
    
    def calculate_feature_contributions(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray
    ) -> np.ndarray:
        """
        Calculate contribution of each feature to anomaly score.
        
        Args:
            original: Original data (batch, seq_len, features)
            reconstruction: Reconstructed data (batch, seq_len, features)
            
        Returns:
            contributions: Feature contributions (batch, features)
        """
        # Mean squared error per feature
        feature_errors = np.mean((original - reconstruction) ** 2, axis=1)
        
        # Normalize to get contributions
        total_error = feature_errors.sum(axis=1, keepdims=True)
        contributions = feature_errors / (total_error + 1e-8)
        
        return contributions
    
    def identify_anomalous_features(
        self,
        feature_contributions: np.ndarray,
        top_k: int = 3,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Identify top contributing features for each anomaly.
        
        Args:
            feature_contributions: Feature contribution matrix
            top_k: Number of top features to return
            feature_names: Names of features
            
        Returns:
            top_features: List of dicts with top features per sample
        """
        n_features = feature_contributions.shape[1]
        
        if feature_names is None:
            feature_names = [f"sensor_{i}" for i in range(n_features)]
        
        top_features_list = []
        
        for contrib in feature_contributions:
            # Get top k indices
            top_indices = np.argsort(contrib)[-top_k:][::-1]
            
            top_features = {
                'features': [feature_names[i] for i in top_indices],
                'contributions': [float(contrib[i]) for i in top_indices]
            }
            top_features_list.append(top_features)
        
        return top_features_list
    
    def filter_by_duration(
        self,
        is_anomaly: np.ndarray,
        min_duration: Optional[int] = None
    ) -> np.ndarray:
        """
        Filter anomalies by minimum duration.
        
        Args:
            is_anomaly: Boolean array of anomalies
            min_duration: Minimum consecutive anomalous windows
            
        Returns:
            filtered_anomalies: Filtered boolean array
        """
        min_duration = min_duration or self.min_anomaly_duration
        
        if min_duration <= 1:
            return is_anomaly
        
        filtered = np.zeros_like(is_anomaly, dtype=bool)
        
        # Find consecutive anomaly sequences
        anomaly_changes = np.diff(np.concatenate([[False], is_anomaly, [False]]).astype(int))
        starts = np.where(anomaly_changes == 1)[0]
        ends = np.where(anomaly_changes == -1)[0]
        
        # Keep only sequences >= min_duration
        for start, end in zip(starts, ends):
            if end - start >= min_duration:
                filtered[start:end] = True
        
        logger.info(f"Filtered {is_anomaly.sum()} -> {filtered.sum()} anomalies "
                   f"(min_duration={min_duration})")
        
        return filtered
    
    def analyze_anomaly_context(
        self,
        anomaly_indices: np.ndarray,
        data: np.ndarray,
        context_window: int = 10
    ) -> List[Dict]:
        """
        Analyze temporal context around anomalies.
        
        Args:
            anomaly_indices: Indices of detected anomalies
            data: Original time series data
            context_window: Size of context window
            
        Returns:
            contexts: List of context dictionaries
        """
        contexts = []
        
        for idx in anomaly_indices:
            start = max(0, idx - context_window)
            end = min(len(data), idx + context_window + 1)
            
            context_data = data[start:end]
            
            context = {
                'index': int(idx),
                'before_mean': float(context_data[:context_window].mean()),
                'after_mean': float(context_data[-context_window:].mean()),
                'before_std': float(context_data[:context_window].std()),
                'after_std': float(context_data[-context_window:].std())
            }
            contexts.append(context)
        
        return contexts
    
    def get_anomaly_summary(
        self,
        results: Dict,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate summary statistics for detected anomalies.
        
        Args:
            results: Results from detect()
            feature_names: Names of features
            
        Returns:
            summary: Summary statistics
        """
        is_anomaly = results['is_anomaly']
        anomaly_scores = results['anomaly_scores']
        feature_contributions = results.get('feature_contributions')
        
        anomaly_indices = np.where(is_anomaly)[0]
        
        summary = {
            'total_windows': len(is_anomaly),
            'anomaly_count': len(anomaly_indices),
            'anomaly_rate': float(is_anomaly.mean()),
            'mean_anomaly_score': float(anomaly_scores[is_anomaly].mean()) if is_anomaly.any() else 0.0,
            'max_anomaly_score': float(anomaly_scores.max()),
            'threshold': float(self.threshold)
        }
        
        # Most anomalous features
        if feature_contributions is not None and is_anomaly.any():
            anomalous_contribs = feature_contributions[is_anomaly]
            mean_contrib = anomalous_contribs.mean(axis=0)
            
            n_features = len(mean_contrib)
            if feature_names is None:
                feature_names = [f"sensor_{i}" for i in range(n_features)]
            
            top_features_idx = np.argsort(mean_contrib)[-5:][::-1]
            
            summary['top_anomalous_features'] = [
                {
                    'feature': feature_names[i],
                    'contribution': float(mean_contrib[i])
                }
                for i in top_features_idx
            ]
        
        return summary
