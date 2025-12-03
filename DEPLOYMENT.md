# Deployment Guide - Music Feature Analysis System

This guide covers deploying the music feature analysis system in various environments.

**Note**: This is the **production system** with advanced models. For grading deliverables, see `notebooks/01_EDA.ipynb` and `notebooks/02_Modeling.ipynb`.

---

## Architecture Overview

- **Notebooks** (`notebooks/`): Academic deliverables with baseline models (8 genres, traditional ML).
- **Production System** (`backend/`, `models/`, `docker/`, `kubernetes/`): Scalable, advanced models with CNNs, multi-label classification, and real-time inference.

---

## Local Development Deployment

### 1. Start Backend API

```bash
cd backend/
python app.py

# API runs at: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 2. Start Model Inference Server

```bash
cd models/app/
python server.py

# Model server runs at: http://localhost:8001
# Supports: CNN, LSTM, ensemble models
```

### 3. Test API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test multi-label prediction
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
curl http://localhost:8000/api/v1/health
curl http://localhost:8001/health
```

### 5. Stop Services

```bash
docker-compose down
```

---

## Production Deployment (Kubernetes)

### 1. Apply Manifests

```bash
cd kubernetes/
kubectl create namespace music-analysis
kubectl apply -f configmap.yaml -n music-analysis
kubectl apply -f deployment.yaml -n music-analysis
kubectl apply -f service.yaml -n music-analysis
```

### 2. Check Status

```bash
kubectl get pods -n music-analysis
kubectl get services -n music-analysis
kubectl logs -f deployment/backend -n music-analysis
```

### 3. Enable Auto-scaling

```bash
kubectl autoscale deployment backend --cpu-percent=70 --min=2 --max=10 -n music-analysis
```

---

## Advanced Features (Production System)

### 1. Multi-Label Genre Classification

- **Model**: CNN with sigmoid output layer.
- **Endpoint**: `POST /api/v1/analysis/predict-multilabel`
- **Input**: Audio file (WAV, MP3, FLAC).
- **Output**: Genre probabilities for 50+ genres/sub-genres.

### 2. Audio Similarity Search

- **Model**: CNN encoder → embedding vector.
- **Endpoint**: `POST /api/v1/analysis/similarity`
- **Input**: Audio file.
- **Output**: Top-K similar songs from database.

### 3. Real-time Streaming Analysis

- **Protocol**: WebSocket.
- **Endpoint**: `ws://localhost:8000/api/v1/stream`
- **Input**: Audio stream chunks.
- **Output**: Real-time genre predictions every 2 seconds.

### 4. Model Explainability

- **Endpoint**: `POST /api/v1/analysis/explain`
- **Input**: Audio file.
- **Output**: Spectrogram with Grad-CAM heatmap showing important regions.

---

## Environment Variables

```bash
# Backend configuration
export MODEL_DIR="models/trained_models"
export MODEL_TYPE="cnn"  # Options: cnn, lstm, ensemble
export LOG_LEVEL="info"
export MAX_WORKERS=4

# API configuration
export API_HOST="0.0.0.0"
export API_PORT=8000
export API_RELOAD=false

# Model configuration
export DEVICE="cuda"  # Use GPU if available
export BATCH_SIZE=64
export NUM_CLASSES=50  # Multi-label support
export ENABLE_MULTILABEL=true
```

---

## Model Training & Versioning

### 1. Train New CNN Model

```bash
cd models/
python train_cnn.py --dataset custom_genres --epochs 100 --model-name cnn_v2
```

### 2. Experiment Tracking (MLflow)

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

### 3. Deploy New Model Version

```bash
# Update model path in backend/.env
MODEL_NAME=cnn_v2

# Restart backend
docker-compose restart backend
```

---

## Monitoring and Logging

### Application Logs

```bash
# Docker
docker-compose logs -f backend

# Kubernetes
kubectl logs -f deployment/music-backend -n music-analysis
```

### Model Performance Metrics

```bash
# View Prometheus metrics
curl http://localhost:8000/metrics

# Grafana dashboard
open http://localhost:3000
```

### Health Checks

```bash
# Local
curl http://localhost:8000/api/v1/health

# Kubernetes
kubectl port-forward svc/backend 8000:8000 -n music-analysis
curl http://localhost:8000/api/v1/health
```

---

## Roadmap (Future Enhancements)

### Phase 1: Expand Genre Coverage
- [ ] Multi-label classification (50+ genres)
- [ ] Sub-genre detection
- [ ] User-defined custom genres

### Phase 2: Advanced Models
- [ ] CNN + LSTM hybrid architecture
- [ ] Transfer learning with VGGish/YAMNet
- [ ] Attention mechanisms for interpretability

### Phase 3: Real-time & Streaming
- [ ] WebSocket streaming API
- [ ] Real-time dashboard
- [ ] Live performance monitoring

### Phase 4: Generative AI
- [ ] Audio style transfer
- [ ] Genre mixing/blending
- [ ] Music generation (VAE/GAN)

### Phase 5: Production Scale
- [ ] Model A/B testing framework
- [ ] Distributed training (multi-GPU)
- [ ] Edge deployment (mobile/embedded)

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

### GPU Not Detected

```bash
# Verify CUDA
nvidia-smi

# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
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

```yaml
# docker/docker-compose.yml
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

### 4. Batch Predictions

```bash
# Process multiple files at once
POST /api/v1/analysis/batch-predict
```

---

## See Also

- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [tests/README.md](tests/README.md) - Test documentation
- [API_GUIDE.md](API_GUIDE.md) - API endpoint reference
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - AI agent guidelines

---

**Last Updated**: 2024