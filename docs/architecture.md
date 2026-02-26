# System Architecture

## Overview

The Anomaly Detection System is designed as a modular, scalable solution for real-time monitoring of offshore production equipment. The architecture follows modern software engineering practices with clear separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Sensors  │  │  SCADA   │  │   IoT    │  │  Manual  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Data Ingestion Layer    │
        │  - Streaming pipelines    │
        │  - Batch processing       │
        │  - Data validation        │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │  Preprocessing Pipeline   │
        │  - Normalization          │
        │  - Missing value handling │
        │  - Outlier detection      │
        │  - Window creation        │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │    ML Processing Layer    │
        │  ┌─────────────────────┐  │
        │  │  LSTM Autoencoder   │  │
        │  │  - Encoding         │  │
        │  │  - Latent space     │  │
        │  │  - Decoding         │  │
        │  │  - Error calculation│  │
        │  └─────────────────────┘  │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Anomaly Detection       │
        │  - Threshold comparison   │
        │  - Feature contribution   │
        │  - Severity scoring       │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Business Logic Layer    │
        │  - Alert generation       │
        │  - Work order creation    │
        │  - SAP integration        │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │    Presentation Layer     │
        │  - Dashboard              │
        │  - API endpoints          │
        │  - Notifications          │
        └───────────────────────────┘
```

## Component Details

### 1. Data Ingestion Layer

**Purpose**: Collect and validate sensor data from multiple sources

**Components**:
- Stream processors for real-time data
- Batch loaders for historical data
- Data validators and cleaners
- Buffer management

**Technologies**:
- Apache Kafka (optional, for high-volume streaming)
- Custom Python connectors
- CSV/Excel file readers

### 2. Preprocessing Pipeline

**Purpose**: Transform raw sensor data into ML-ready format

**Components**:
- `SensorDataProcessor` class
  - Handles missing values
  - Removes outliers
  - Normalizes features
  - Creates sliding windows

**Key Features**:
- Configurable window sizes
- Multiple scaling strategies
- Outlier detection methods (IQR, Z-score)
- Temporal alignment

### 3. ML Processing Layer

**Purpose**: Learn normal patterns and detect deviations

**Components**:
- `LSTMAutoencoder` model
  - Stacked LSTM encoder
  - Latent space representation
  - Stacked LSTM decoder
  - Reconstruction error calculation

**Model Architecture**:
```
Input: (batch, sequence_length, n_sensors)
  ↓
Encoder LSTM Layers
  ↓
Latent Representation: (batch, latent_dim)
  ↓
Decoder LSTM Layers
  ↓
Output: (batch, sequence_length, n_sensors)
```

**Training Strategy**:
- Unsupervised learning on normal operations
- Mean Squared Error (MSE) loss
- Adam optimizer
- Early stopping with validation set

### 4. Anomaly Detection

**Purpose**: Identify and score anomalies

**Components**:
- `AnomalyDetector` class
  - Dynamic threshold calculation
  - Feature-level contribution analysis
  - Severity scoring
  - Temporal filtering

**Detection Methods**:
1. **Percentile-based**: Threshold at 95th percentile of training errors
2. **Statistical**: Mean + k*std from training distribution
3. **MAD**: Median Absolute Deviation based

### 5. Integration Layer

**Purpose**: Connect to enterprise systems

**Components**:
- `SAPConnector` class
  - RFC-based connection (PyRFC)
  - REST API fallback
  - Work order creation
  - Equipment metadata retrieval

**Workflow**:
1. Anomaly detected
2. Map sensor to equipment
3. Determine priority based on severity
4. Create maintenance work order
5. Log to SAP ECC

### 6. Presentation Layer

**Purpose**: Visualize and interact with the system

**Components**:
- **Dashboard** (Streamlit)
  - Real-time monitoring
  - Historical analysis
  - Alert management
  
- **REST API** (FastAPI)
  - Model inference endpoints
  - Configuration management
  - Health checks

## Data Flow

### Training Flow

```
Raw Data → Load → Clean → Normalize → Window → Split → Train → Validate → Save Model
```

### Inference Flow

```
New Data → Preprocess → Model → Reconstruction Error → Threshold → Alert → SAP
```

## Scalability Considerations

### Horizontal Scaling
- Multiple inference workers
- Load balancing with NGINX/HAProxy
- Distributed training (PyTorch DDP)

### Vertical Scaling
- GPU acceleration for model inference
- Batch processing for efficiency
- In-memory caching

### Storage
- Model versioning in cloud storage (S3, GCS, Azure)
- Time-series database for sensor data (InfluxDB, TimescaleDB)
- Document store for alerts (MongoDB)

## Security Architecture

### Data Security
- Encryption at rest and in transit
- Secure credential management (environment variables, AWS Secrets Manager)
- Data anonymization for testing

### Access Control
- API authentication (JWT tokens)
- Role-based access control (RBAC)
- SAP authorization delegation

### Audit Trail
- All predictions logged
- Work order creation tracked
- User actions recorded

## Deployment Architecture

### Development
```
Local Machine → Docker Compose → Multiple Services
```

### Production
```
Cloud Infrastructure → Kubernetes → Auto-scaling Pods
```

**Components**:
- Training cluster (GPU instances)
- Inference cluster (CPU instances, auto-scaling)
- Database cluster (managed services)
- Message queue (Kafka/RabbitMQ)

## Monitoring & Observability

### Metrics
- Model performance (precision, recall, F1)
- Inference latency
- System resource usage
- Alert frequency

### Logging
- Structured logging (JSON format)
- Centralized log aggregation (ELK stack)
- Error tracking (Sentry)

### Alerting
- Critical system failures
- Model drift detection
- Excessive false positives
- Infrastructure issues

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| ML Framework | PyTorch, scikit-learn |
| Data Processing | Pandas, NumPy, SciPy |
| Web Framework | FastAPI, Streamlit |
| Database | MongoDB, PostgreSQL, SQLite |
| SAP Integration | PyRFC, REST APIs |
| Containerization | Docker, Docker Compose |
| Orchestration | Kubernetes |
| Cloud | AWS, GCP, Azure |
| Monitoring | Prometheus, Grafana, TensorBoard |
| CI/CD | GitHub Actions |

## Future Enhancements

1. **Multi-model Ensemble**: Combine LSTM with other architectures (CNN, Transformer)
2. **Explainable AI**: SHAP values, attention visualization
3. **Adaptive Thresholds**: Dynamic threshold adjustment based on operational context
4. **Federated Learning**: Train across multiple sites without centralizing data
5. **Edge Deployment**: Run inference on edge devices near sensors
6. **Advanced Analytics**: Predictive maintenance, remaining useful life (RUL) estimation
