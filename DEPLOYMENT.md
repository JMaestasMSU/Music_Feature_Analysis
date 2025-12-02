# Deployment Guide - Music Genre Classification System

This guide covers deploying the music genre classification system in various environments.

---

## Local Development Deployment

### 1. Start Backend API

**macOS/Linux:**
```bash
cd backend/
python app.py

# API runs at: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Windows:**
```powershell
cd backend
python app.py
```

### 2. Start Model Inference Server (Optional)

```bash
cd app/
python main.py

# API runs at: http://localhost:8001
```

### 3. Test API

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
python backend/test_api.py
```

---

## Docker Deployment

### 1. Build Containers

```bash
cd docker/
docker-compose build
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Verify Running

```bash
docker-compose ps
docker-compose logs -f backend
```

### 4. Test Endpoints

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### 5. Stop Services

```bash
docker-compose down
```

---

## Production Deployment (Kubernetes)

### 1. Create Deployment Manifests

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: music-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: music-backend
  template:
    metadata:
      labels:
        app: music-backend
    spec:
      containers:
      - name: backend
        image: music-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_DIR
          value: "/app/models/trained_models"
```

### 2. Deploy to Kubernetes

```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml
```

### 3. Check Status

```bash
kubectl get pods
kubectl get services
kubectl logs -f deployment/music-backend
```

---

## Environment Variables

```bash
# Backend configuration
export MODEL_DIR="models/trained_models"
export MODEL_NAME="genre_classifier_production"
export LOG_LEVEL="info"
export MAX_WORKERS=4

# API configuration
export API_HOST="0.0.0.0"
export API_PORT=8000
export API_RELOAD=false

# Model configuration
export DEVICE="cpu"  # or "cuda" for GPU
export BATCH_SIZE=64
```

---

## Monitoring and Logging

### Application Logs

```bash
# Docker
docker-compose logs -f backend

# Kubernetes
kubectl logs -f deployment/music-backend
```

### Health Checks

```bash
# Local
curl http://localhost:8000/health

# Docker
curl http://localhost:8000/health

# Kubernetes
kubectl port-forward svc/music-backend 8000:8000
curl http://localhost:8000/health
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Model Not Loading

```bash
# Check model files exist
ls -lh models/trained_models/

# Check permissions
chmod +r models/trained_models/*
```

### Docker Build Fails

```bash
# Clean docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

---

## Performance Optimization

### 1. Enable GPU (if available)

```bash
# Update docker-compose.yml
services:
  model-server:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Increase Workers

```bash
# Update uvicorn command
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Add Caching

```python
# In backend/app.py
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_prediction(features_hash):
    # Cache frequently requested predictions
    pass
```