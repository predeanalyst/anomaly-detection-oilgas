# API Documentation

## Overview

The Anomaly Detection System provides a RESTful API for real-time anomaly detection and system management.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API uses API key authentication. Include your API key in the request headers:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### 1. Health Check

Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-02-26T10:00:00Z"
}
```

### 2. Predict Anomalies

Detect anomalies in sensor data.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "data": [
    [23.5, 45.2, 78.1, 12.3, 56.7, 89.0, 34.2, 67.8, 90.1, 23.4],
    [23.6, 45.3, 78.0, 12.4, 56.8, 89.1, 34.3, 67.9, 90.2, 23.5]
  ],
  "sensor_names": ["temp_1", "pressure_1", "vibration_1", ...],
  "timestamp": "2024-02-26T10:00:00Z"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "index": 0,
      "is_anomaly": false,
      "anomaly_score": 0.234,
      "reconstruction_error": 0.0012
    },
    {
      "index": 1,
      "is_anomaly": true,
      "anomaly_score": 0.876,
      "reconstruction_error": 0.0234,
      "anomalous_features": [
        {
          "feature": "temp_1",
          "contribution": 0.45
        },
        {
          "feature": "pressure_1",
          "contribution": 0.32
        }
      ]
    }
  ],
  "threshold": 0.0156,
  "model_version": "1.0.0"
}
```

### 3. Batch Predict

Process multiple batches of sensor data.

**Endpoint:** `POST /predict/batch`

**Request Body:**
```json
{
  "batches": [
    {
      "batch_id": "batch_001",
      "data": [[...], [...]]
    },
    {
      "batch_id": "batch_002",
      "data": [[...], [...]]
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "batch_id": "batch_001",
      "predictions": [...],
      "processing_time_ms": 45.2
    }
  ],
  "total_batches": 2,
  "total_predictions": 200
}
```

### 4. Get Model Info

Retrieve information about the loaded model.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "model_name": "LSTM Autoencoder",
  "version": "1.0.0",
  "input_dim": 10,
  "hidden_dim": 64,
  "latent_dim": 32,
  "num_layers": 2,
  "trained_on": "2024-02-20T15:30:00Z",
  "threshold": 0.0156,
  "performance": {
    "precision": 0.942,
    "recall": 0.918,
    "f1_score": 0.930
  }
}
```

### 5. Update Threshold

Update the anomaly detection threshold.

**Endpoint:** `POST /model/threshold`

**Request Body:**
```json
{
  "threshold": 0.02,
  "method": "manual"
}
```

**Response:**
```json
{
  "success": true,
  "old_threshold": 0.0156,
  "new_threshold": 0.02
}
```

### 6. Get Statistics

Retrieve system statistics.

**Endpoint:** `GET /stats`

**Query Parameters:**
- `start_date` (optional): Start date (ISO format)
- `end_date` (optional): End date (ISO format)

**Response:**
```json
{
  "total_predictions": 10000,
  "total_anomalies": 450,
  "anomaly_rate": 0.045,
  "avg_inference_time_ms": 42.3,
  "date_range": {
    "start": "2024-02-01T00:00:00Z",
    "end": "2024-02-26T10:00:00Z"
  }
}
```

### 7. Create Work Order

Manually create a SAP work order for an anomaly.

**Endpoint:** `POST /sap/work-order`

**Request Body:**
```json
{
  "equipment_id": "PUMP-001",
  "description": "Anomaly detected in temperature sensor",
  "priority": "High",
  "anomaly_details": {
    "anomaly_score": 0.876,
    "timestamp": "2024-02-26T10:00:00Z",
    "affected_sensors": ["temp_1", "pressure_1"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "work_order_number": "WO-20240226001",
  "created_at": "2024-02-26T10:01:00Z",
  "status": "CREATED"
}
```

### 8. Get Work Orders

Retrieve created work orders.

**Endpoint:** `GET /sap/work-orders`

**Query Parameters:**
- `limit` (optional): Number of results (default: 50)
- `status` (optional): Filter by status

**Response:**
```json
{
  "work_orders": [
    {
      "work_order_number": "WO-20240226001",
      "equipment_id": "PUMP-001",
      "description": "Anomaly detected in temperature sensor",
      "priority": "High",
      "status": "CREATED",
      "created_at": "2024-02-26T10:01:00Z"
    }
  ],
  "total": 1
}
```

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data format",
    "details": {
      "field": "data",
      "issue": "Expected array of arrays"
    }
  },
  "timestamp": "2024-02-26T10:00:00Z"
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid request data |
| `MODEL_ERROR` | Model inference failed |
| `SAP_ERROR` | SAP integration error |
| `NOT_FOUND` | Resource not found |
| `UNAUTHORIZED` | Invalid or missing API key |
| `RATE_LIMIT` | Too many requests |
| `INTERNAL_ERROR` | Server error |

## Rate Limiting

- **Anonymous**: 100 requests/hour
- **Authenticated**: 1000 requests/hour
- **Premium**: 10,000 requests/hour

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1709035200
```

## Example Usage

### Python

```python
import requests

url = "http://localhost:8000/api/v1/predict"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

data = {
    "data": [[23.5, 45.2, 78.1, 12.3, 56.7, 89.0, 34.2, 67.8, 90.1, 23.4]],
    "timestamp": "2024-02-26T10:00:00Z"
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

if result["predictions"][0]["is_anomaly"]:
    print(f"Anomaly detected! Score: {result['predictions'][0]['anomaly_score']}")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[23.5, 45.2, 78.1, 12.3, 56.7, 89.0, 34.2, 67.8, 90.1, 23.4]],
    "timestamp": "2024-02-26T10:00:00Z"
  }'
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    data: [[23.5, 45.2, 78.1, 12.3, 56.7, 89.0, 34.2, 67.8, 90.1, 23.4]],
    timestamp: new Date().toISOString()
  })
});

const result = await response.json();
console.log(result);
```

## Webhooks

Configure webhooks to receive real-time anomaly alerts.

**Endpoint:** `POST /webhooks`

**Request Body:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["anomaly.detected", "work_order.created"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**
```json
{
  "event": "anomaly.detected",
  "timestamp": "2024-02-26T10:00:00Z",
  "data": {
    "anomaly_score": 0.876,
    "affected_sensors": ["temp_1"],
    "equipment_id": "PUMP-001"
  }
}
```

## SDK Support

Official SDKs are available for:
- Python: `pip install anomaly-detection-client`
- JavaScript/Node.js: `npm install @anomaly-detection/client`

## Support

For API support:
- Email: api-support@yourcompany.com
- Documentation: https://docs.yourcompany.com
- Status: https://status.yourcompany.com
