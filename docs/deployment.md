# Deployment Guide

This guide covers deployment options for the Anomaly Detection System in various environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [AWS Deployment](#aws-deployment)
- [Google Cloud Platform](#google-cloud-platform)
- [Azure Deployment](#azure-deployment)
- [Kubernetes](#kubernetes)
- [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development

### Quick Setup

```bash
git clone https://github.com/yourusername/anomaly-detection-system.git
cd anomaly-detection-system

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Running Services

```bash
# Train model
python src/train.py --data data/raw/sensor_data.csv

# Start API
uvicorn src.api.main:app --reload --port 8000

# Start dashboard
streamlit run src/dashboard/app.py --server.port 8501
```

---

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t anomaly-detector:latest .

# Run container
docker run -d \
  --name anomaly-detector \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e SAP_SERVER=your-server \
  -e SAP_USER=your-user \
  anomaly-detector:latest
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

**Services included:**
- API server (port 8000)
- Dashboard (port 8501)
- MongoDB (port 27017)
- Jupyter (port 8888)

---

## AWS Deployment

### Option 1: EC2 Instance

#### 1. Launch EC2 Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxx \
  --subnet-id subnet-xxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=anomaly-detector}]'
```

#### 2. Setup Instance

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Clone and deploy
git clone https://github.com/yourusername/anomaly-detection-system.git
cd anomaly-detection-system
docker-compose up -d
```

#### 3. Configure Security Group

Open ports:
- 22 (SSH)
- 8000 (API)
- 8501 (Dashboard)

### Option 2: ECS (Elastic Container Service)

#### 1. Push to ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag anomaly-detector:latest your-account.dkr.ecr.us-east-1.amazonaws.com/anomaly-detector:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/anomaly-detector:latest
```

#### 2. Create Task Definition

```json
{
  "family": "anomaly-detector",
  "containerDefinitions": [
    {
      "name": "anomaly-detector",
      "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/anomaly-detector:latest",
      "memory": 2048,
      "cpu": 1024,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "SAP_SERVER", "value": "your-server"},
        {"name": "SAP_CLIENT", "value": "100"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/anomaly-detector",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048"
}
```

#### 3. Create Service

```bash
aws ecs create-service \
  --cluster your-cluster \
  --service-name anomaly-detector \
  --task-definition anomaly-detector \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Option 3: SageMaker

```python
# scripts/deploy_sagemaker.py
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

role = 'arn:aws:iam::your-account:role/SageMakerRole'

pytorch_model = PyTorchModel(
    model_data='s3://your-bucket/models/model.tar.gz',
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py'
)

predictor = pytorch_model.deploy(
    instance_type='ml.m5.xlarge',
    initial_instance_count=1
)

print(f"Endpoint: {predictor.endpoint_name}")
```

---

## Google Cloud Platform

### Cloud Run Deployment

```bash
# Build and submit to Cloud Build
gcloud builds submit --tag gcr.io/your-project/anomaly-detector

# Deploy to Cloud Run
gcloud run deploy anomaly-detector \
  --image gcr.io/your-project/anomaly-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars SAP_SERVER=your-server,SAP_CLIENT=100
```

### GKE (Google Kubernetes Engine)

```bash
# Create cluster
gcloud container clusters create anomaly-detector-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --region us-central1

# Get credentials
gcloud container clusters get-credentials anomaly-detector-cluster --region us-central1

# Deploy (see Kubernetes section)
kubectl apply -f k8s/
```

---

## Azure Deployment

### Azure Container Instances

```bash
# Create resource group
az group create --name anomaly-detector-rg --location eastus

# Create container
az container create \
  --resource-group anomaly-detector-rg \
  --name anomaly-detector \
  --image your-registry.azurecr.io/anomaly-detector:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables SAP_SERVER=your-server SAP_CLIENT=100 \
  --secure-environment-variables SAP_PASSWORD=your-password
```

### Azure Kubernetes Service (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group anomaly-detector-rg \
  --name anomaly-detector-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group anomaly-detector-rg --name anomaly-detector-cluster

# Deploy (see Kubernetes section)
kubectl apply -f k8s/
```

---

## Kubernetes

### Deployment Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detector
  labels:
    app: anomaly-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-detector
  template:
    metadata:
      labels:
        app: anomaly-detector
    spec:
      containers:
      - name: anomaly-detector
        image: your-registry/anomaly-detector:latest
        ports:
        - containerPort: 8000
        env:
        - name: SAP_SERVER
          valueFrom:
            secretKeyRef:
              name: sap-credentials
              key: server
        - name: SAP_PASSWORD
          valueFrom:
            secretKeyRef:
              name: sap-credentials
              key: password
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Configuration

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detector-service
spec:
  type: LoadBalancer
  selector:
    app: anomaly-detector
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### Secrets

```bash
# Create secrets
kubectl create secret generic sap-credentials \
  --from-literal=server=your-server \
  --from-literal=client=100 \
  --from-literal=user=your-user \
  --from-literal=password=your-password
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-detector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detector
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods
kubectl get services
kubectl logs -f deployment/anomaly-detector
```

---

## Monitoring & Maintenance

### Health Checks

```bash
# API health
curl http://your-server:8000/health

# Container health
docker ps
docker logs anomaly-detector
```

### Logging

```bash
# Docker logs
docker-compose logs -f api

# Kubernetes logs
kubectl logs -f deployment/anomaly-detector

# AWS CloudWatch
aws logs tail /ecs/anomaly-detector --follow
```

### Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'anomaly-detector'
    static_configs:
      - targets: ['anomaly-detector:8000']
```

### Backup

```bash
# Backup models
aws s3 sync models/ s3://your-bucket/models/backup/

# Backup database
mongodump --uri mongodb://localhost:27017/anomaly_detection
```

### Updates

```bash
# Rolling update with Docker
docker-compose pull
docker-compose up -d

# Kubernetes rolling update
kubectl set image deployment/anomaly-detector \
  anomaly-detector=your-registry/anomaly-detector:v2.0.0

# Check rollout status
kubectl rollout status deployment/anomaly-detector

# Rollback if needed
kubectl rollout undo deployment/anomaly-detector
```

### SSL/TLS

```bash
# Using Let's Encrypt with Nginx
docker run -d \
  --name nginx-proxy \
  -p 80:80 \
  -p 443:443 \
  -v /path/to/certs:/etc/nginx/certs \
  nginx-proxy

# Or use AWS Certificate Manager with ALB
```

## Troubleshooting

### Common Issues

**Issue: Container fails to start**
```bash
# Check logs
docker logs anomaly-detector

# Check resources
docker stats
```

**Issue: Out of memory**
```bash
# Increase memory limit
docker run -m 4g anomaly-detector

# Kubernetes
resources:
  limits:
    memory: "8Gi"
```

**Issue: SAP connection fails**
```bash
# Test connectivity
telnet your-sap-server 3300

# Check credentials
echo $SAP_PASSWORD
```

## Best Practices

1. **Use secrets management** (AWS Secrets Manager, Azure Key Vault)
2. **Enable auto-scaling** for production workloads
3. **Set up monitoring** (Prometheus, Grafana, CloudWatch)
4. **Implement CI/CD** for automated deployments
5. **Regular backups** of models and configurations
6. **Use load balancers** for high availability
7. **Enable logging** at appropriate levels
8. **Monitor costs** on cloud platforms

---

For additional support, consult:
- [Architecture Documentation](architecture.md)
- [API Documentation](api.md)
- Cloud provider documentation
