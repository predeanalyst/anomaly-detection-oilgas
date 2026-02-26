"""
Logging utility for the anomaly detection system.

Provides structured logging with different handlers and formatters.
"""

import logging
import logging.config
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class StructuredLogger:
    """
    Structured logger with JSON output support.
    """
    
    def __init__(self, name: str, log_level: str = 'INFO'):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
    
    def log(self, level: str, message: str, **kwargs):
        """
        Log message with additional context.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        getattr(self.logger, level.lower())(json.dumps(log_data))


def setup_logging(
    config_path: Optional[str] = None,
    default_level: str = 'INFO',
    log_dir: str = 'logs'
):
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to logging configuration file
        default_level: Default logging level
        log_dir: Directory for log files
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    if config_path and os.path.exists(config_path):
        # Load configuration from file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logging.config.dictConfig(config)
    else:
        # Default configuration
        logging.basicConfig(
            level=getattr(logging, default_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{log_dir}/anomaly_detection.log')
            ]
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(name)


class AnomalyLogger:
    """
    Specialized logger for anomaly detection events.
    """
    
    def __init__(self, log_file: str = 'logs/anomalies.log'):
        """
        Initialize anomaly logger.
        
        Args:
            log_file: Path to anomaly log file
        """
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('anomaly_logger')
        self.logger.setLevel(logging.INFO)
        
        # File handler for anomaly logs
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_anomaly(
        self,
        timestamp: str,
        anomaly_score: float,
        threshold: float,
        affected_sensors: list,
        equipment_id: Optional[str] = None,
        work_order: Optional[str] = None
    ):
        """
        Log an anomaly detection event.
        
        Args:
            timestamp: Timestamp of anomaly
            anomaly_score: Anomaly score
            threshold: Detection threshold
            affected_sensors: List of affected sensors
            equipment_id: Equipment ID
            work_order: Created work order number
        """
        log_entry = {
            'timestamp': timestamp,
            'anomaly_score': anomaly_score,
            'threshold': threshold,
            'affected_sensors': affected_sensors,
            'equipment_id': equipment_id,
            'work_order': work_order
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def get_anomaly_history(self, n: int = 100) -> list:
        """
        Get recent anomaly history.
        
        Args:
            n: Number of recent anomalies
            
        Returns:
            history: List of anomaly records
        """
        history = []
        
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines[-n:]:
                try:
                    # Parse log line
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        timestamp, data = parts
                        history.append(json.loads(data.strip()))
                except:
                    continue
        
        return history
