# Quick Start Guide

This guide will help you get up and running with the Anomaly Detection System in under 10 minutes.

## Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM
- (Optional) CUDA-capable GPU for faster training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/anomaly-detection-system.git
cd anomaly-detection-system
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Generate Sample Data

We'll generate synthetic sensor data for testing:

```bash
python scripts/generate_sample_data.py \
  --output data/raw/sensor_data.csv \
  --samples 10000 \
  --sensors 10
```

This creates a CSV file with:
- 10,000 time points
- 10 sensors
- Injected anomalies (sensor drift, equipment failure, spikes)

## Train the Model

Train the LSTM autoencoder on the sample data:

```bash
python src/train.py \
  --data data/raw/sensor_data.csv \
  --config configs/config.yaml \
  --output models/lstm_autoencoder.pth \
  --epochs 20 \
  --early-stopping
```

Expected output:
```
Loading and preprocessing data...
Loaded 10000 rows with 10 sensors
Created 9901 windows of size 100
Split data: train=6931, val=1485, test=1485
Training model...
Epoch 1/20 - Train Loss: 0.452103, Val Loss: 0.398234
Epoch 2/20 - Train Loss: 0.312456, Val Loss: 0.287654
...
Model saved to models/lstm_autoencoder.pth
```

Training takes 5-10 minutes on CPU, 1-2 minutes on GPU.

## Test Detection

Create a simple test script to detect anomalies:

```python
# test_detection.py
import torch
import pickle
import pandas as pd
import numpy as np

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.utils.anomaly_detector import AnomalyDetector

# Load model
model = LSTMAutoencoder.load('models/lstm_autoencoder.pth')

# Load preprocessor
with open('models/lstm_autoencoder_processor.pkl', 'rb') as f:
    processor = pickle.load(f)

# Initialize detector
detector = AnomalyDetector(model, threshold_percentile=95)

# Load some test data
df = pd.read_csv('data/raw/sensor_data.csv')
test_data = df.iloc[-1000:].drop('timestamp', axis=1).values

# Preprocess
normalized_data = processor.scaler.transform(test_data)
windows = processor.create_windows(normalized_data)

# Calculate threshold
train_windows = torch.FloatTensor(windows[:700])
detector.calculate_threshold(train_windows)

# Detect anomalies
test_windows = torch.FloatTensor(windows[700:])
results = detector.detect(test_windows)

print(f"Analyzed {len(test_windows)} windows")
print(f"Detected {results['n_anomalies']} anomalies")
print(f"Anomaly rate: {results['anomaly_rate']:.2%}")
```

Run it:
```bash
python test_detection.py
```

## Using Docker

### Quick Start with Docker

```bash
# Build the image
docker build -t anomaly-detector .

# Generate sample data
docker run -v $(pwd)/data:/app/data anomaly-detector \
  python scripts/generate_sample_data.py \
  --output /app/data/raw/sensor_data.csv

# Train the model
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           anomaly-detector \
  python src/train.py \
  --data /app/data/raw/sensor_data.csv \
  --output /app/models/lstm_autoencoder.pth
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trainer

# Stop services
docker-compose down
```

## Next Steps

### 1. Explore the Codebase

Key files to understand:
- `src/models/lstm_autoencoder.py` - Core ML model
- `src/preprocessing/data_processor.py` - Data preprocessing
- `src/utils/anomaly_detector.py` - Anomaly detection logic
- `src/utils/sap_connector.py` - SAP integration

### 2. Customize Configuration

Edit `configs/config.yaml` to adjust:
- Model architecture (hidden dimensions, layers)
- Training parameters (epochs, batch size)
- Detection thresholds
- SAP integration settings

### 3. Use Your Own Data

Replace the sample data with your actual sensor data:

```python
# Your data should be in CSV format:
# timestamp, sensor_1, sensor_2, ..., sensor_n

python src/train.py \
  --data path/to/your/data.csv \
  --output models/your_model.pth
```

### 4. Set Up SAP Integration

1. Copy `.env.example` to `.env`
2. Fill in your SAP credentials:
   ```
   SAP_SERVER=your-server
   SAP_CLIENT=100
   SAP_USER=your-username
   SAP_PASSWORD=your-password
   ```

3. Update equipment mapping in `configs/config.yaml`

### 5. Deploy to Production

See `docs/deployment.md` for production deployment guides:
- AWS deployment
- Kubernetes setup
- Monitoring configuration

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or window size in config:
```yaml
training:
  batch_size: 16  # Reduce from 32

preprocessing:
  window_size: 50  # Reduce from 100
```

### Issue: PyRFC Installation Fails

**Solution**: SAP RFC library is optional. The system will use REST API fallback:
```bash
# Skip PyRFC
pip install -r requirements.txt | grep -v pyrfc
```

### Issue: CUDA Not Available

**Solution**: Train on CPU (just slower):
```bash
python src/train.py --no-cuda ...
```

### Issue: Import Errors

**Solution**: Make sure you're in the virtual environment:
```bash
# Activate venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install in editable mode
pip install -e .
```

## Getting Help

- **Documentation**: Check `docs/` folder
- **Issues**: Open an issue on GitHub
- **Examples**: See `notebooks/` for Jupyter examples

## Example End-to-End Workflow

```bash
# 1. Setup
git clone <repo>
cd anomaly-detection-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python scripts/generate_sample_data.py

# 3. Train model
python src/train.py \
  --data data/raw/sensor_data.csv \
  --output models/model.pth \
  --epochs 20

# 4. Test detection
python test_detection.py

# 5. (Optional) Deploy
docker-compose up -d
```

That's it! You now have a working anomaly detection system.

## What's Next?

- Read the [Architecture Documentation](docs/architecture.md)
- Explore [API Documentation](docs/api.md)
- Check out [Jupyter Notebooks](notebooks/)
- Review [Contributing Guidelines](CONTRIBUTING.md)

Happy detecting! 🚀
