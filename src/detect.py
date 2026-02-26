"""
Detection script for running inference on new data.

Usage:
    python src/detect.py --model models/model.pth --data data/raw/new_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

from api.realtime_detector import RealtimeDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Detect anomalies in sensor data')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data CSV')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Detection threshold')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Initialize detector
    logger.info("Initializing detector...")
    detector = RealtimeDetector(
        model_path=args.model,
        threshold=args.threshold,
        device=args.device
    )
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Extract sensor columns (excluding timestamp)
    sensor_cols = [col for col in df.columns if col != 'timestamp']
    sensor_names = sensor_cols
    sensor_data = df[sensor_cols].values
    
    logger.info(f"Processing {len(sensor_data)} samples with {len(sensor_cols)} sensors")
    
    # Run detection
    logger.info("Running anomaly detection...")
    result = detector.predict(sensor_data, sensor_names=sensor_names)
    
    if result['status'] == 'success':
        # Print summary
        summary = result['summary']
        logger.info(f"\nDetection Summary:")
        logger.info(f"Total windows: {summary['total_windows']}")
        logger.info(f"Anomalies detected: {summary['anomaly_count']}")
        logger.info(f"Anomaly rate: {summary['anomaly_rate']:.2%}")
        logger.info(f"Threshold: {result['threshold']:.6f}")
        
        # Save results
        if args.output:
            logger.info(f"\nSaving results to {args.output}")
            
            # Create results DataFrame
            predictions = result['predictions']
            results_df = pd.DataFrame(predictions)
            
            # Add timestamp if available
            if 'timestamp' in df.columns:
                # Map predictions to timestamps
                timestamps = df['timestamp'].iloc[:len(predictions)]
                results_df.insert(0, 'timestamp', timestamps.values)
            
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            logger.info("Results saved successfully")
            
            # Print sample of anomalies
            anomalies = results_df[results_df['is_anomaly'] == True]
            if len(anomalies) > 0:
                logger.info(f"\nSample anomalies:")
                print(anomalies.head(10))
    else:
        logger.warning(f"Detection status: {result['status']}")
        logger.warning(f"Message: {result.get('message', 'Unknown')}")


if __name__ == '__main__':
    main()
