"""
Metrics calculation for anomaly detection evaluation.

Provides various metrics for model performance assessment.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AnomalyMetrics:
    """
    Calculate and track anomaly detection metrics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.predictions_history = []
        self.labels_history = []
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels (0 = normal, 1 = anomaly)
            y_pred: Predicted labels
            y_scores: Anomaly scores (for ROC-AUC)
            
        Returns:
            metrics: Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Rates
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Accuracy
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        
        # ROC-AUC if scores provided
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['roc_auc'] = None
        
        # Additional metrics
        metrics['anomaly_rate_true'] = y_true.mean()
        metrics['anomaly_rate_pred'] = y_pred.mean()
        
        logger.info(f"Calculated metrics: Precision={metrics['precision']:.3f}, "
                   f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        return metrics
    
    def calculate_detection_delay(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate detection delay for anomalies.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamps: Timestamps for each sample
            
        Returns:
            delay_metrics: Detection delay metrics
        """
        delays = []
        
        # Find true anomaly periods
        true_changes = np.diff(np.concatenate([[0], y_true, [0]]))
        anomaly_starts = np.where(true_changes == 1)[0]
        anomaly_ends = np.where(true_changes == -1)[0]
        
        for start, end in zip(anomaly_starts, anomaly_ends):
            # Find first detection in this period
            detections_in_period = np.where(y_pred[start:end] == 1)[0]
            
            if len(detections_in_period) > 0:
                delay = detections_in_period[0]
                delays.append(delay)
        
        if delays:
            delay_metrics = {
                'mean_detection_delay': np.mean(delays),
                'median_detection_delay': np.median(delays),
                'max_detection_delay': np.max(delays),
                'detection_rate': len(delays) / len(anomaly_starts)
            }
        else:
            delay_metrics = {
                'mean_detection_delay': None,
                'median_detection_delay': None,
                'max_detection_delay': None,
                'detection_rate': 0.0
            }
        
        return delay_metrics
    
    def calculate_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        thresholds: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate metrics for different thresholds.
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            thresholds: Array of thresholds to evaluate
            
        Returns:
            threshold_metrics: Metrics for each threshold
        """
        precisions = []
        recalls = []
        f1_scores = []
        fprs = []
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            fprs.append(fpr)
        
        return {
            'thresholds': thresholds,
            'precisions': np.array(precisions),
            'recalls': np.array(recalls),
            'f1_scores': np.array(f1_scores),
            'fprs': np.array(fprs)
        }
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = 'f1',
        n_thresholds: int = 100
    ) -> Tuple[float, float]:
        """
        Find optimal threshold based on a metric.
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            metric: Metric to optimize ('f1', 'precision', 'recall')
            n_thresholds: Number of thresholds to test
            
        Returns:
            optimal_threshold: Best threshold
            best_score: Best metric score
        """
        thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)
        
        threshold_metrics = self.calculate_threshold_metrics(
            y_true, y_scores, thresholds
        )
        
        if metric == 'f1':
            scores = threshold_metrics['f1_scores']
        elif metric == 'precision':
            scores = threshold_metrics['precisions']
        elif metric == 'recall':
            scores = threshold_metrics['recalls']
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.6f} "
                   f"(score: {best_score:.3f})")
        
        return optimal_threshold, best_score
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for classes
        """
        if target_names is None:
            target_names = ['Normal', 'Anomaly']
        
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        )
        
        print("\n" + "="*50)
        print("Classification Report")
        print("="*50)
        print(report)
        print("="*50 + "\n")
    
    def update_history(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update prediction history.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        self.predictions_history.extend(y_pred.tolist())
        self.labels_history.extend(y_true.tolist())
    
    def get_cumulative_metrics(self) -> Dict[str, float]:
        """
        Get metrics over entire history.
        
        Returns:
            metrics: Cumulative metrics
        """
        if not self.predictions_history:
            return {}
        
        y_true = np.array(self.labels_history)
        y_pred = np.array(self.predictions_history)
        
        return self.calculate_metrics(y_true, y_pred)
