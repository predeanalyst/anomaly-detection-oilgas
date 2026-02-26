"""
Generate synthetic sensor data for testing and demonstration.

This script creates realistic time-series sensor data with injected anomalies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os


def generate_normal_data(
    n_samples: int,
    n_sensors: int,
    frequency: str = '1min',
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate normal sensor data with realistic patterns.
    
    Args:
        n_samples: Number of time points
        n_sensors: Number of sensors
        frequency: Sampling frequency
        seed: Random seed
        
    Returns:
        df: DataFrame with sensor data
    """
    np.random.seed(seed)
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    data = {'timestamp': timestamps}
    
    # Generate sensor data with different characteristics
    for i in range(n_sensors):
        if i % 3 == 0:
            # Temperature-like: slow variation with daily cycle
            base = 25 + 5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))
            noise = np.random.normal(0, 0.5, n_samples)
            sensor_data = base + noise
            
        elif i % 3 == 1:
            # Pressure-like: moderate variation with trend
            base = 100 + 0.001 * np.arange(n_samples)
            noise = np.random.normal(0, 2, n_samples)
            sensor_data = base + noise
            
        else:
            # Vibration-like: high frequency with occasional spikes
            base = 10
            noise = np.random.normal(0, 1, n_samples)
            sensor_data = base + noise
        
        data[f'sensor_{i+1}'] = sensor_data
    
    df = pd.DataFrame(data)
    return df


def inject_sensor_drift(
    df: pd.DataFrame,
    sensor_col: str,
    start_idx: int,
    drift_rate: float = 0.01
) -> pd.DataFrame:
    """
    Inject gradual sensor drift anomaly.
    
    Args:
        df: Input DataFrame
        sensor_col: Column to inject drift into
        start_idx: Index to start drift
        drift_rate: Rate of drift per sample
        
    Returns:
        df: Modified DataFrame
    """
    n_samples = len(df)
    drift_length = n_samples - start_idx
    
    # Create drift pattern
    drift = np.zeros(n_samples)
    drift[start_idx:] = drift_rate * np.arange(drift_length)
    
    df[sensor_col] = df[sensor_col] + drift
    return df


def inject_equipment_failure(
    df: pd.DataFrame,
    sensor_cols: list,
    failure_idx: int,
    failure_duration: int = 100
) -> pd.DataFrame:
    """
    Inject equipment failure anomaly affecting multiple sensors.
    
    Args:
        df: Input DataFrame
        sensor_cols: List of columns to affect
        failure_idx: Index where failure starts
        failure_duration: Duration of failure
        
    Returns:
        df: Modified DataFrame
    """
    end_idx = min(failure_idx + failure_duration, len(df))
    
    for col in sensor_cols:
        # Sudden drop followed by irregular behavior
        df.loc[failure_idx:end_idx, col] *= 0.5
        noise = np.random.normal(0, 5, end_idx - failure_idx + 1)
        df.loc[failure_idx:end_idx, col] += noise
    
    return df


def inject_spike_anomaly(
    df: pd.DataFrame,
    sensor_col: str,
    spike_indices: list,
    spike_magnitude: float = 50
) -> pd.DataFrame:
    """
    Inject spike anomalies.
    
    Args:
        df: Input DataFrame
        sensor_col: Column to inject spikes into
        spike_indices: List of indices for spikes
        spike_magnitude: Magnitude of spikes
        
    Returns:
        df: Modified DataFrame
    """
    for idx in spike_indices:
        if idx < len(df):
            df.loc[idx, sensor_col] += spike_magnitude
    
    return df


def generate_dataset_with_anomalies(
    n_samples: int = 10000,
    n_sensors: int = 10,
    anomaly_rate: float = 0.05,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate complete dataset with various anomalies.
    
    Args:
        n_samples: Number of samples
        n_sensors: Number of sensors
        anomaly_rate: Approximate proportion of anomalous data
        seed: Random seed
        
    Returns:
        df: Complete dataset with anomalies
    """
    np.random.seed(seed)
    
    # Generate normal data
    print(f"Generating {n_samples} samples with {n_sensors} sensors...")
    df = generate_normal_data(n_samples, n_sensors, seed=seed)
    
    # Inject sensor drift (gradual degradation)
    drift_sensor = f'sensor_{np.random.randint(1, n_sensors+1)}'
    drift_start = int(n_samples * 0.6)
    print(f"Injecting sensor drift in {drift_sensor} starting at index {drift_start}")
    df = inject_sensor_drift(df, drift_sensor, drift_start, drift_rate=0.005)
    
    # Inject equipment failure
    failure_sensors = [f'sensor_{i}' for i in np.random.choice(range(1, n_sensors+1), 3, replace=False)]
    failure_idx = int(n_samples * 0.75)
    print(f"Injecting equipment failure at index {failure_idx} affecting {failure_sensors}")
    df = inject_equipment_failure(df, failure_sensors, failure_idx, failure_duration=200)
    
    # Inject random spikes
    n_spikes = int(n_samples * anomaly_rate * 0.1)
    spike_sensor = f'sensor_{np.random.randint(1, n_sensors+1)}'
    spike_indices = np.random.choice(range(n_samples), n_spikes, replace=False)
    print(f"Injecting {n_spikes} spikes in {spike_sensor}")
    df = inject_spike_anomaly(df, spike_sensor, spike_indices.tolist())
    
    # Add some missing values
    n_missing = int(n_samples * 0.01)
    missing_indices = np.random.choice(range(n_samples), n_missing, replace=False)
    missing_sensor = f'sensor_{np.random.randint(1, n_sensors+1)}'
    print(f"Adding {n_missing} missing values in {missing_sensor}")
    df.loc[missing_indices, missing_sensor] = np.nan
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic sensor data')
    parser.add_argument('--output', type=str, default='data/raw/sensor_data.csv',
                       help='Output file path')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples')
    parser.add_argument('--sensors', type=int, default=10,
                       help='Number of sensors')
    parser.add_argument('--anomaly-rate', type=float, default=0.05,
                       help='Anomaly rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data
    df = generate_dataset_with_anomalies(
        n_samples=args.samples,
        n_sensors=args.sensors,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed
    )
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nDataset saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())


if __name__ == '__main__':
    main()
