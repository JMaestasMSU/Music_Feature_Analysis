# Quick Start - Local Deployment

This guide will get you up and running with the Music Feature Analysis system on your local machine using Docker.

## Prerequisites

- **Docker** and **Docker Compose** installed ([Get Docker](https://docs.docker.com/get-docker/))
- **50GB+ free disk space** (for FMA Medium dataset)
- **Python 3.9+** (for setup script)
- **(Optional)** NVIDIA GPU with CUDA 11.8+ for training

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Music_Feature_Analysis
```

### 2. Download and Setup Dataset

```bash
# Install required Python packages for setup script
pip install requests tqdm

# Download FMA Medium dataset (~25GB) and setup directories
python scripts/download_fma_medium.py

# If you already have the dataset downloaded:
python scripts/download_fma_medium.py --skip-download
```

**What this does:**
- Creates directory structure: `data/raw/`, `data/processed/`, `data/metadata/`
- Downloads FMA Medium dataset (25,000 tracks, ~25GB)
- Downloads FMA metadata files (~342MB)
- Extracts and organizes files into numbered directories (000/, 001/, etc.)
- Validates dataset structure


**Expected structure:**
```
data/
├── raw/
│   ├── 000/
│   │   ├── 000002.mp3
│   │   ├── 000003.mp3
│   │   └── ...
│   ├── 001/
│   │   └── ...
│   └── ...
├── metadata/
│   ├── tracks.csv
│   ├── genres.csv
│   └── ...
└── processed/  (will be created by feature extraction)
```


### 3. Extract Audio Features

```bash
# Using Docker (recommended - uses conda environment)
docker-compose --profile preprocessing up feature-extraction

# OR locally if you have conda environment setup:
conda activate music-feature-analysis
python scripts/extract_audio_features.py
```

**What this does:**
- Processes all audio files in `data/raw/`
- Extracts mel-spectrograms and audio features
- Saves features to `data/processed/extracted_features.pkl`
- Creates spectrogram images in `data/processed/spectrograms/`

**Time:** ~2-4 hours for 25,000 tracks (depends on CPU)

### 4. Train the Model

```bash
# Using Docker with GPU (recommended)
docker-compose --profile training up training

# OR locally with GPU:
conda activate music-feature-analysis
python scripts/train_multilabel_cnn.py
```

**What this does:**
- Loads extracted features from `data/processed/`
- Trains multi-label CNN model
- Saves best model to `models/trained_models/multilabel_cnn_best.pt`
- Generates training plots in `outputs/`
- Logs metrics to `logs/`

**Time:** ~30-60 minutes with GPU, several hours with CPU

### 5. Start the API

```bash
# Start the backend API service
docker-compose up backend

# API will be available at:
# - http://localhost:8000
# - Swagger docs: http://localhost:8000/docs
```

**What this includes:**
- Backend API with genre classification
- Audio file upload and prediction
- Swagger UI for interactive API testing
- Health monitoring endpoints

### 6. Test the System

Open your browser and go to [http://localhost:8000/docs](http://localhost:8000/docs) to access the Swagger UI.

**Try these endpoints:**

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Get Available Genres**
   ```bash
   curl http://localhost:8000/genres
   ```

3. **Predict from Audio File**
   - In Swagger UI: Navigate to `POST /predict-from-audio`
   - Click "Try it out"
   - Upload an MP3/WAV file
   - Set `top_k` to 5
   - Click "Execute"

4. **Predict from Features**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                       1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
          "top_k": 3}'
   ```

## Docker Compose Profiles

The `docker-compose.yml` uses profiles to separate services:

- **Default** (no profile): Only starts the backend API
- **`--profile preprocessing`**: Runs feature extraction
- **`--profile training`**: Runs model training
- **`--profile full`**: Starts backend + model inference service

Examples:

```bash
# Only backend API
docker-compose up backend

# Feature extraction
docker-compose --profile preprocessing up feature-extraction

# Training
docker-compose --profile training up training

# All services (backend + model server)
docker-compose --profile full up
```

## Useful Commands

### View Logs
```bash
docker-compose logs -f backend
docker-compose logs -f training
```

### Rebuild After Code Changes
```bash
docker-compose build backend
docker-compose up backend
```

### Stop Services
```bash
docker-compose down
```

### Remove Containers and Volumes
```bash
docker-compose down -v
```

### Check Running Containers
```bash
docker-compose ps
```

## Troubleshooting

### Port 8000 Already in Use
```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process or change port in docker-compose.yml
```

### Dataset Download Failed
- Check internet connection
- Try manual download from: https://github.com/mdeff/fma
- Place files in `data/raw/` and run with `--skip-download`

### Docker Build Failed
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### GPU Not Detected (for Training)
```bash
# Check NVIDIA GPU
nvidia-smi

# Install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Verify PyTorch sees GPU
docker run --rm --gpus all pytorch/pytorch:latest python -c "import torch; print(torch.cuda.is_available())"
```

### Model Not Loading
```bash
# Check model file exists
ls -lh models/trained_models/

# Verify training completed successfully
docker-compose logs training | tail -n 50
```

## Next Steps

1. **Try different audio files**: Upload various genres to test accuracy
2. **Retrain with more data**: Modify training parameters in `scripts/train_multilabel_cnn.py`
3. **Explore the notebooks**: Check `notebooks/` for analysis and visualization
4. **Add new genres**: Update genre mappings in metadata
5. **Deploy to production**: See [DEPLOYMENT.md](DEPLOYMENT.md) for Kubernetes setup

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                     Music Feature Analysis                 │
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Frontend   │──▶│  Backend API │───▶│    Model     │  │
│  │ (Swagger UI) │    │   (FastAPI)  │    │  (PyTorch)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                            │                       │       │
│                            ▼                       ▼       │
│                      ┌──────────────┐    ┌──────────────┐  │
│                      │   Feature    │    │   Trained    │  │
│                      │  Extraction  │    │   Models     │  │
│                      │  (Librosa)   │    │   (.pt)      │  │
│                      └──────────────┘    └──────────────┘  │
│                            │                               │
│                            ▼                               │
│                      ┌──────────────┐                      │
│                      │     FMA      │                      │
│                      │   Dataset    │                      │
│                      │  (25k tracks)│                      │
│                      └──────────────┘                      │
└────────────────────────────────────────────────────────────┘
```

## Support

- **Documentation**: See [README.md](README.md) and [DEPLOYMENT.md](DEPLOYMENT.md)
- **API Reference**: [http://localhost:8000/docs](http://localhost:8000/docs) (when running)
- **Issues**: Check existing scripts and test files for examples

---

**Time to Complete Setup:** 3-5 hours (mostly download and feature extraction time)
**Disk Space Required:** ~50GB
**Recommended:** 16GB RAM, NVIDIA GPU for training
