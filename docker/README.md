# Docker Configuration

This directory contains Docker configurations for the Music Feature Analysis system.

## Available Dockerfiles

### 1. Dockerfile.backend
**Purpose:** FastAPI backend service with audio upload and prediction

**Base Image:** continuumio/miniconda3:latest

**Services:**
- Audio file upload and processing
- Genre classification API
- Feature extraction from audio
- Multi-label prediction

**Port:** 8000

**Environment:** Conda environment from `environments/environment.yml`

**Usage:**
```bash
docker-compose up backend
```

### 2. Dockerfile.model
**Purpose:** Alternative lightweight model inference service

**Base Image:** continuumio/miniconda3:latest

**Services:**
- Model inference
- Feature-based prediction
- Lightweight API for predictions

**Port:** 8001

**Environment:** Conda environment from `environments/environment.yml`

**Usage:**
```bash
docker-compose --profile full up model
```

### 3. Dockerfile.training
**Purpose:** Model training service

**Base Image:** continuumio/miniconda3:latest

**Services:**
- Multi-label CNN training
- Model checkpointing
- Training metrics logging

**Environment:** Conda environment from `environments/environment-gpu-cuda11.8.yml` (GPU)

**Usage:**
```bash
docker-compose --profile training up training
```

**Note:** This is a one-time service that trains the model and exits. Requires GPU for reasonable training time.

### 4. Dockerfile.feature-extraction
**Purpose:** Audio feature extraction service

**Base Image:** continuumio/miniconda3:latest

**Services:**
- Extract mel-spectrograms from audio
- Process FMA dataset
- Generate feature vectors

**Environment:** Conda environment from `environments/environment-cpu.yml`

**Usage:**
```bash
docker-compose --profile preprocessing up feature-extraction
```

**Note:** This is a one-time service that processes the dataset and exits. Takes 2-4 hours for 25k tracks.

## Docker Compose Profiles

The `docker-compose.yml` file uses profiles to separate services into logical groups:

### Default (No Profile)
Starts only the essential services:
- **backend**: Main API service

```bash
docker-compose up
```

### `preprocessing` Profile
Runs data preprocessing:
- **feature-extraction**: Extract features from audio

```bash
docker-compose --profile preprocessing up
```

### `training` Profile
Runs model training:
- **training**: Train the multi-label CNN

```bash
docker-compose --profile training up
```

### `full` Profile
Starts all API services:
- **backend**: Main API (port 8000)
- **model**: Lightweight inference API (port 8001)

```bash
docker-compose --profile full up
```

## Volume Mounts

All services use the following volume mounts:

```yaml
volumes:
  - ./data:/app/data              # Dataset and processed features
  - ./models:/app/models          # Model definitions
  - ./models/trained_models:/app/models/trained_models  # Trained model weights
  - ./logs:/app/logs              # Application logs
  - ./outputs:/app/outputs        # Training outputs and plots
```

### Read-Only vs Read-Write

- **Backend & Model services**: Use `:ro` (read-only) for data and models
- **Training & Feature Extraction**: Use read-write access to save outputs

## Network Configuration

All services run on the `music-analysis-net` bridge network, allowing inter-service communication.

```yaml
networks:
  music-analysis-net:
    driver: bridge
```

## Environment Variables

Common environment variables used across services:

```bash
PYTHONUNBUFFERED=1                                          # Python unbuffered output
MODEL_PATH=/app/models/trained_models/multilabel_cnn_best.pt  # Model weights path
LOG_LEVEL=INFO                                             # Logging level
CUDA_VISIBLE_DEVICES=0                                     # GPU device (training only)
```

## Building Images

### Build All Images
```bash
docker-compose build
```

### Build Specific Service
```bash
docker-compose build backend
docker-compose build training
```

### Build Without Cache
```bash
docker-compose build --no-cache
```

## Running Services

### Start Services in Foreground
```bash
docker-compose up backend
```

### Start Services in Background
```bash
docker-compose up -d backend
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f backend
docker-compose logs -f training
```

### Restart Service
```bash
docker-compose restart backend
```

## GPU Support

### Enable GPU for Training

The training service supports NVIDIA GPUs through CUDA. To enable:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml` to add GPU reservation:

```yaml
training:
  # ... other config ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. Verify GPU is available:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Health Checks

### Backend Health Check
```bash
curl http://localhost:8000/health
```

### Model Service Health Check
```bash
curl http://localhost:8001/health
```

### Docker Health Check
```bash
docker-compose ps
# Look for "healthy" status
```

## Troubleshooting

### Container Fails to Start
```bash
# Check logs
docker-compose logs backend

# Check if port is already in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Remove and recreate
docker-compose down
docker-compose up backend
```

### Model Not Found
```bash
# Check if model file exists
ls -lh models/trained_models/

# Run training first
docker-compose --profile training up training
```

### Out of Memory
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or add to docker-compose.yml:
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Conda Environment Errors
```bash
# Rebuild without cache
docker-compose build --no-cache backend

# Check conda environment file exists
ls -lh environments/environment.yml
```

### Permission Errors
```bash
# Fix file permissions
chmod -R 755 data/ models/ logs/ outputs/

# Or run as current user
docker-compose run --user $(id -u):$(id -g) backend
```

## Development Workflow

### 1. Setup Dataset
```bash
python scripts/setup_fma_dataset.py
```

### 2. Extract Features
```bash
docker-compose --profile preprocessing up feature-extraction
```

### 3. Train Model
```bash
docker-compose --profile training up training
```

### 4. Start API
```bash
docker-compose up backend
```

### 5. Test Changes
```bash
# Rebuild after code changes
docker-compose build backend
docker-compose up backend

# Or use bind mount for live reload (not recommended for production)
```

### 6. Deploy
```bash
# Tag and push images
docker tag music-analysis-backend:latest your-registry/music-analysis-backend:v1.0
docker push your-registry/music-analysis-backend:v1.0
```

## Production Considerations

### Security
- Don't run as root (add `USER` directive in Dockerfiles)
- Use secrets for sensitive config
- Scan images for vulnerabilities: `docker scan music-analysis-backend`

### Optimization
- Use multi-stage builds to reduce image size
- Enable Docker BuildKit: `DOCKER_BUILDKIT=1 docker build`
- Use `.dockerignore` to exclude unnecessary files

### Monitoring
- Add logging to external service (ELK, CloudWatch, etc.)
- Use Prometheus metrics endpoint
- Set up Grafana dashboards

## See Also

- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Full deployment guide
- [README.md](../README.md) - Project overview
- [docker-compose.yml](../docker-compose.yml) - Service configuration
