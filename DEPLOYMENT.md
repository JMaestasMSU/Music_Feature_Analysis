# Deployment Guide

This document explains how to run and deploy the Music Feature Analysis system.

---

## For Grading (View Deliverables Only)

If you're here to **grade the project**, follow these steps:

### 1. View Deliverable Notebooks

```bash
# Open EDA notebook
jupyter notebook notebooks/01_EDA.ipynb

# Open Modeling notebook
jupyter notebook notebooks/02_Modeling.ipynb
```

**Grading Rubric**: See [README.md](README.md) for point breakdown

### 2. View Presentation

```bash
# Open presentation PDF
open presentation/presentation.pdf

# Read summary documentation
cat presentation/SUMMARY.md
```

### 3. Run Quick Tests (Optional - Proves Components Work)

```bash
cd tests/
bash run_all_tests.sh
```

Takes < 2 minutes. Shows:
- FFT spectral analysis works
- Audio features extract correctly
- CNN architecture trains
- Bayesian optimization searches

**No grading required for tests**, but demonstrates system integrity.

---

## For Development (Run the System)

### Prerequisites

```bash
# Python 3.9+
python --version

# pip
pip --version
```

### Option 1: Backend Only (Lightweight, No GPU)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Access**:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health

**Limitations**: 
- Model predictions use dummy model (no real inference)
- Good for testing API structure

### Option 2: Full Stack (Requires Docker)

```bash
cd docker
docker-compose up -d
```

**Services**:
- Backend: http://localhost:8000
- Model Server: http://localhost:8001
- Combined: Full end-to-end system

**Monitor**:
```bash
docker-compose logs -f
docker ps
```

**Stop**:
```bash
docker-compose down
```

### Option 3: Development with GPU (Docker + NVIDIA)

```bash
# Requires NVIDIA Docker runtime installed
cd docker
docker-compose up -d

# Verify GPU
docker exec music_analysis_model_server nvidia-smi
```

---

## Configuration

### Backend Configuration

Edit `backend/.env`:

```bash
ENVIRONMENT=development
LOG_LEVEL=INFO
SAMPLE_RATE=44100
N_MELS=128
ENABLE_CACHING=true
MAX_FILE_SIZE=52428800
```

### Docker Configuration

Edit `docker/docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8000:8000"  # Change port here
    environment:
      - ENVIRONMENT=production
```

---

## Testing

### Quick Component Tests

```bash
cd tests/
bash run_all_tests.sh
```

**Output**: if all pass, if any fail

### Test API Endpoints

```bash
cd backend
python test_api.py
```

**Tests**:
- Backend health check
- Model server health check
- API endpoints
- Synthetic predictions

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Get features
curl http://localhost:8000/api/v1/analysis/features

# Get FFT info
curl http://localhost:8000/api/v1/analysis/fft-validation
```

---

## Docker Deployment

### Build Containers

```bash
cd docker

# Build backend
docker build -f Dockerfile.backend -t music-backend .

# Build model server
docker build -f Dockerfile.model -t music-model .
```

### Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop services
docker-compose down -v
```

### Push to Registry (Production)

```bash
# Tag images
docker tag music-backend:latest myregistry/music-backend:latest
docker tag music-model:latest myregistry/music-model:latest

# Push
docker push myregistry/music-backend:latest
docker push myregistry/music-model:latest
```

---

## Kubernetes Deployment (Production)

### Prerequisites

```bash
kubectl version
helm version
```

### Deploy

```bash
cd kubernetes

# Create namespace
kubectl create namespace music-analysis

# Create ConfigMap
kubectl apply -f configmap.yaml -n music-analysis

# Deploy backend
kubectl apply -f deployment.yaml -n music-analysis
kubectl apply -f service.yaml -n music-analysis
```

### Monitor

```bash
# Check deployments
kubectl get deployments -n music-analysis

# Check pods
kubectl get pods -n music-analysis

# View logs
kubectl logs -f deployment/backend -n music-analysis

# Port forward
kubectl port-forward svc/backend 8000:8000 -n music-analysis
```

---

## Performance Tuning

### Backend Optimization

```python
# backend/config.py
# Adjust based on available resources
CACHE_TTL = 3600        # Cache results 1 hour
ENABLE_CACHING = True   # Enable caching
```

### Model Optimization

```python
# models/app/server.py
# Batch size affects throughput vs latency
BATCH_SIZE = 32  # Increase for throughput
```

### Docker Resources

```yaml
# docker/docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## Troubleshooting

### Backend Won't Start

```bash
# Check port is available
lsof -i :8000

# Check dependencies
pip install -r requirements.txt

# Run with verbose logging
LOG_LEVEL=DEBUG python app.py
```

### Model Server Not Responding

```bash
# Check Docker logs
docker-compose logs model-server

# Check GPU availability
docker exec music_analysis_model_server nvidia-smi

# Verify model file exists
ls -la models/trained_models/cnn_best_model.pt
```

### Tests Failing

```bash
# Run individual test with output
python tests/quick_cnn_test.py

# Check dependencies
pip install torch numpy scipy librosa

# Clean and retry
rm -rf __pycache__ .pytest_cache
python tests/quick_cnn_test.py
```

### API Not Responding

```bash
# Check service health
curl -v http://localhost:8000/api/v1/health

# Check logs
docker-compose logs backend

# Restart service
docker-compose restart backend
```

---

## Monitoring

### Logs

```bash
# Backend logs
docker-compose logs -f backend

# Model server logs
docker-compose logs -f model-server

# All logs
docker-compose logs -f
```

### Health Checks

```bash
# Automated
watch curl http://localhost:8000/api/v1/health

# Manual
curl http://localhost:8000/api/v1/status
```

---

## Stopping Services

```bash
# Stop containers (keep volumes)
docker-compose stop

# Stop and remove containers (keep volumes)
docker-compose down

# Stop and remove everything
docker-compose down -v

# Remove Docker images
docker-compose down --rmi all
```

---

## See Also

- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [tests/README.md](tests/README.md) - Test documentation
- [API_GUIDE.md](API_GUIDE.md) - API endpoint reference

---

**Last Updated**: 2024
