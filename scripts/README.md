# Scripts Directory

This directory contains all setup, data processing, and training scripts for the Music Feature Analysis system.

---

## Table of Contents

1. [Environment Setup Scripts](#environment-setup-scripts)
2. [Dataset Download Scripts](#dataset-download-scripts)
3. [Feature Extraction Scripts](#feature-extraction-scripts)
4. [Training Scripts](#training-scripts)
5. [Utility Scripts](#utility-scripts)
6. [Quick Start Guide](#quick-start-guide)

---

## Environment Setup Scripts

### `install_requirements.sh` (Linux/macOS)
**Purpose:** Bootstrap script to setup Python environment on Unix systems with CPU or GPU support

**Usage:**
```bash
cd scripts/
chmod +x install_requirements.sh

# CPU-only (default) - for inference and feature extraction
./install_requirements.sh

# GPU with CUDA 11.8 - for training
./install_requirements.sh --gpu

# Base environment
./install_requirements.sh --base

# Custom environment name
./install_requirements.sh --gpu --name my-gpu-env

# Show help
./install_requirements.sh --help
```

**Options:**
- `--gpu, --cuda`: Install GPU environment with CUDA 11.8 (requires NVIDIA GPU)
- `--base`: Install base environment (general use)
- `--name NAME`: Custom environment name
- `--help, -h`: Show help message

**What it does:**
- Detects if conda is available
- Creates conda environment based on selected type:
  - **CPU** (default): `environments/environment-cpu.yml` → `mfa-cpu`
  - **GPU**: `environments/environment-gpu-cuda11.8.yml` → `mfa-gpu-cuda11-8`
  - **Base**: `environments/environment.yml` → `music-feature-analysis`
- Falls back to pip if conda not available (CPU-only PyTorch)
- Verifies GPU support for GPU environments

**When to use:**
- **CPU mode**: Feature extraction, API inference, development without GPU
- **GPU mode**: Model training with NVIDIA GPU (much faster)
- **Base mode**: General development

**Requirements for GPU:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- CUDA 11.8 compatible GPU

---

### `install_requirements.ps1` (Windows)
**Purpose:** Bootstrap script to setup Python environment on Windows with CPU or GPU support

**Usage:**
```powershell
cd scripts/

# CPU-only (default) - for inference and feature extraction
.\install_requirements.ps1

# GPU with CUDA 11.8 - for training
.\install_requirements.ps1 -GPU

# GPU with dev dependencies
.\install_requirements.ps1 -GPU -Dev

# Base environment
.\install_requirements.ps1 -Base

# Custom environment name
.\install_requirements.ps1 -GPU -EnvName my-gpu-env

# Show help
.\install_requirements.ps1 -Help
```

**Parameters:**
- `-GPU`: Install GPU environment with CUDA 11.8 (requires NVIDIA GPU)
- `-Base`: Install base environment (general use)
- `-EnvName`: Custom conda environment name
- `-Dev`: Install development dependencies (pytest, black, flake8, etc.)
- `-Help`: Show help message

**What it does:**
- Detects if conda is available
- Creates conda environment based on selected type:
  - **CPU** (default): `environments/environment-cpu.yml` → `mfa-cpu`
  - **GPU**: `environments/environment-gpu-cuda11.8.yml` → `mfa-gpu-cuda11-8`
  - **Base**: `environments/environment.yml` → `music-feature-analysis`
- Falls back to pip if conda not available (CPU-only PyTorch)
- Optionally installs dev dependencies
- Verifies GPU support for GPU environments

**When to use:**
- **CPU mode**: Feature extraction, API inference, development without GPU
- **GPU mode**: Model training with NVIDIA GPU (much faster)
- **Base mode**: General development

**Requirements for GPU:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- CUDA 11.8 compatible GPU

---

## Dataset Download Scripts

### `download_fma_medium.py` (25GB)
**Purpose:** Download and setup FMA Medium dataset (25,000 tracks)

**Usage:**
```bash
# Download and extract dataset
python scripts/download_fma_medium.py

# Skip download if files already exist
python scripts/download_fma_medium.py --skip-download
```

**What it does:**
1. Creates directory structure: `data/raw/`, `data/processed/`, `data/metadata/`
2. Downloads FMA Medium dataset (~25GB) from official source
3. Downloads FMA metadata (~342MB)
4. Extracts files to `data/raw/000/`, `data/raw/001/`, etc.
5. Validates dataset structure

**Output:**
```
data/
├── raw/
│   ├── 000/
│   │   ├── 000002.mp3
│   │   ├── 000003.mp3
│   │   └── ...
│   ├── 001/
│   │   └── ...
│   └── ... (156 directories)
└── metadata/
    ├── tracks.csv
    ├── genres.csv
    ├── features.csv
    └── ...
```

**When to use:**
- Production training with full dataset
- Local deployment setup
- Research with larger dataset

**Time:** 1-2 hours (download) + 30 minutes (extraction)

---

### `download_fma_small.py` (8GB)
**Purpose:** Download and setup FMA Small dataset (8,000 tracks) - faster for testing

**Usage:**
```bash
python scripts/download_fma_small.py
```

**What it does:**
1. Creates directory structure (same as medium)
2. Downloads FMA Small dataset (~8GB)
3. Downloads FMA metadata (~342MB)
4. Extracts files to `data/raw/`
5. Validates dataset structure

**Output:** Same structure as FMA Medium, but with fewer tracks

**When to use:**
- Quick testing and development
- Learning the system
- Limited disk space
- Faster iteration cycles

**Time:** 20-30 minutes (download) + 10 minutes (extraction)

---

## Feature Extraction Scripts

### `extract_audio_features.py`
**Purpose:** Extract mel-spectrograms and audio features from raw audio files

**Usage:**
```bash
# Extract features from all audio in data/raw/
python scripts/extract_audio_features.py

# Or using Docker
docker-compose --profile preprocessing up feature-extraction
```

**What it does:**
1. Scans `data/raw/` for all MP3 files
2. Loads audio and extracts:
   - Mel-spectrograms (128 mel bins)
   - MFCCs (20 coefficients)
   - Spectral features (centroid, rolloff, bandwidth)
   - Zero-crossing rate
   - Chroma features
3. Saves to `data/processed/extracted_features.pkl`
4. Generates spectrogram images in `data/processed/spectrograms/`

**Output:**
```
data/processed/
├── extracted_features.pkl  (features + labels)
└── spectrograms/
    ├── 000002.png
    ├── 000003.png
    └── ...
```

**When to use:**
- After downloading FMA dataset
- Before training models
- When you need fresh feature extraction

**Time:** 2-4 hours for FMA Medium (CPU), 30 minutes for FMA Small

---

### `feature_extraction.py`
**Purpose:** Helper library with feature extraction functions

**Usage:**
```python
from scripts.feature_extraction import extract_features, features_to_array

# Extract features from audio file
features = extract_features('path/to/audio.mp3')

# Convert to array for model
feature_array = features_to_array(features)
```

**What it provides:**
- `extract_features(audio_path)` - Extract audio features
- `extract_mel_spectrogram(audio, sr)` - Extract mel-spectrogram
- `features_to_array(features)` - Convert features dict to numpy array
- Common feature extraction parameters and configurations

**When to use:**
- Used internally by `extract_audio_features.py`
- Used by backend API for real-time prediction
- Import when you need feature extraction in custom scripts

---

### `process_audio_files.py`
**Purpose:** Create features.pkl from pre-computed FMA metadata (quick setup)

**Usage:**
```bash
python scripts/process_audio_files.py
```

**What it does:**
1. Loads `data/metadata/features.csv` (pre-computed by FMA)
2. Loads `data/metadata/tracks.csv` (genre labels)
3. Selects key feature groups (MFCC, spectral, temporal)
4. Merges features with genre labels
5. Cleans data (removes NaN values)
6. Saves to `data/processed/features.pkl`

**Output:** `data/processed/features.pkl` (8,000 tracks × 500 features)

**When to use:**
- Quick setup without extracting features from audio
- Using pre-computed FMA features
- Faster than `extract_audio_features.py`
- Good for initial exploration

**Time:** 1-2 minutes

---

## Training Scripts

### `train_multilabel_cnn.py`
**Purpose:** Train multi-label CNN model for genre classification

**Usage:**
```bash
# Train with default settings
python scripts/train_multilabel_cnn.py

# Or using Docker with GPU
docker-compose --profile training up training
```

**What it does:**
1. Loads features from `data/processed/extracted_features.pkl`
2. Applies data augmentation (SpecAugment, Mixup)
3. Trains multi-label CNN model
4. Saves best model to `models/trained_models/multilabel_cnn_best.pt`
5. Generates training plots in `outputs/`
6. Logs metrics to `logs/`

**Model Architecture:**
- 4 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Fully connected layers with ReLU activation
- Sigmoid output for multi-label classification

**Hyperparameters:**
- Batch size: 64
- Learning rate: 0.001
- Epochs: 50 (early stopping enabled)
- Optimizer: Adam
- Loss: BCEWithLogitsLoss

**Output:**
```
models/trained_models/
├── multilabel_cnn_best.pt         (best model weights)
├── multilabel_cnn_genres.json     (genre names)
├── multilabel_cnn_metadata.json   (training metrics)
└── multilabel_cnn_scaler.pkl      (feature scaler)

outputs/
├── training_loss.png
├── validation_metrics.png
└── confusion_matrix.png

logs/
└── training_YYYYMMDD_HHMMSS.log
```

**When to use:**
- After feature extraction
- When you have new training data
- To retrain with different hyperparameters
- Initial model training for deployment

**Time:** 30-60 minutes with GPU, 4-6 hours with CPU

---

## Utility Scripts

### `analyze_dataset_structure.py`
**Purpose:** Analyze CSV files to understand dataset structure

**Usage:**
```bash
python scripts/analyze_dataset_structure.py data/metadata/tracks.csv
python scripts/analyze_dataset_structure.py data/metadata/features.csv
```

**What it does:**
- Analyzes data types, unique values, statistics
- Detects categorical vs numerical columns
- Identifies missing values and null percentages
- Generates summary for AI coding assistants

**When to use:**
- Understanding new datasets
- Debugging data issues
- Documenting dataset structure
- Initial data exploration

---

### `create_sample_model.py` (Dev/Testing)
**Purpose:** Create tiny sklearn model for testing

**Usage:**
```bash
python scripts/create_sample_model.py --out models/sample_model.joblib
```

**What it does:**
- Generates synthetic classification data
- Trains small RandomForestClassifier
- Saves to specified path

**When to use:**
- Testing model loading code
- CI/CD pipeline testing
- Development without real models

---

## Quick Start Guide

### Option 1: Docker (Recommended)

```bash
# 1. Setup dataset
pip install requests tqdm
python scripts/download_fma_medium.py

# 2. Extract features
docker-compose --profile preprocessing up feature-extraction

# 3. Train model
docker-compose --profile training up training

# 4. Start API
docker-compose up backend
```

### Option 2: Local Development (Conda)

```bash
# 1. Setup environment
# Choose CPU (default) or GPU based on your hardware

# Windows with GPU:
cd scripts
.\install_requirements.ps1 -GPU

# Windows CPU-only:
.\install_requirements.ps1

# Linux/macOS with GPU:
cd scripts
chmod +x install_requirements.sh
./install_requirements.sh --gpu

# Linux/macOS CPU-only:
./install_requirements.sh

# 2. Activate environment
conda activate mfa-gpu-cuda11-8  # If you installed GPU
# OR
conda activate mfa-cpu           # If you installed CPU

# 3. Download dataset (choose one)
python scripts/download_fma_small.py      # Faster (8GB)
# OR
python scripts/download_fma_medium.py     # More data (25GB)

# 4. Extract features (works with CPU or GPU environment)
python scripts/extract_audio_features.py

# 5. Train model (use GPU environment for faster training!)
python scripts/train_multilabel_cnn.py

# 6. Start API
cd backend/
python app.py
```

### Option 3: Quick Setup (Pre-computed Features)

```bash
# 1. Download FMA metadata only
cd data/metadata/
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
cd ../../

# 2. Create features from metadata (no audio processing)
python scripts/process_audio_files.py

# 3. Train on pre-computed features
python scripts/train_multilabel_cnn.py

# Note: This uses FMA's pre-computed features (faster but less control)
```

---

## Script Execution Order

For a complete setup from scratch:

```
1. install_requirements.sh/ps1    (one time - setup environment)
   ↓
2. download_fma_medium.py         (one time - download dataset)
   ↓
3. extract_audio_features.py      (one time - extract features)
   ↓
4. train_multilabel_cnn.py        (train model)
   ↓
5. Start backend API               (deploy)
```

---

## Environment Files Reference

Located in `environments/` directory - **3 different environment files** for different use cases:

### `environment-cpu.yml` → `mfa-cpu` (Default)
**Purpose:** CPU-only environment for inference and feature extraction

**Contains:**
- Python 3.10
- PyTorch 2.2.0 (CPU-only)
- NumPy, Pandas, Scikit-learn
- Librosa, Matplotlib, Seaborn
- UMAP-learn

**Best for:**
- Feature extraction from audio
- Running the backend API
- Development and testing
- Systems without NVIDIA GPU
- Faster environment setup

**Not ideal for:**
- Model training (very slow on CPU)

---

### `environment-gpu-cuda11.8.yml` → `mfa-gpu-cuda11-8`
**Purpose:** GPU-accelerated environment for fast model training

**Contains:**
- Python 3.10
- PyTorch 2.2 with CUDA 11.8 support
- TorchVision 0.15, TorchAudio 2.2
- CUDA Toolkit 11.8
- NumPy, Pandas, Scikit-learn
- Librosa, Matplotlib, Seaborn
- UMAP-learn

**Best for:**
- Training models (30-60min vs 4-6hrs on CPU!)
- GPU-accelerated inference
- Research with large datasets

**Requires:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- CUDA 11.8 compatible GPU (RTX 20/30/40 series, Tesla, A100, etc.)

---

### `environment.yml` → `music-feature-analysis`
**Purpose:** Base environment with minimal PyTorch (general use)

**Contains:**
- Python 3.10
- PyTorch (via pip, CPU-only)
- NumPy, Pandas, Scikit-learn
- Librosa, Matplotlib, Seaborn
- UMAP-learn

**Best for:**
- General development
- When you don't need specific CPU/GPU optimizations
- Quick setup for exploration

---

## How to Choose an Environment

| Use Case | Environment | Command |
|----------|------------|---------|
| **Feature extraction** | CPU | `./install_requirements.sh` |
| **API inference** | CPU | `.\install_requirements.ps1` |
| **Model training** | GPU | `./install_requirements.sh --gpu` |
| **Development** | CPU or Base | `.\install_requirements.ps1` |
| **Research (large data)** | GPU | `.\install_requirements.ps1 -GPU` |

**Quick Decision:**
- Have NVIDIA GPU? → Use GPU environment for training
- No GPU / Mac M1/M2? → Use CPU environment
- Docker? → Automatically handled by Dockerfiles

---

## Troubleshooting

### Script not found
```bash
# Make sure you're in the project root
cd /path/to/Music_Feature_Analysis
python scripts/script_name.py
```

### Permission denied
```bash
chmod +x scripts/*.sh
```

### Module not found
```bash
# Activate conda environment
conda activate music-feature-analysis

# Or verify PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Dataset download fails
- Check internet connection
- Try manual download from https://github.com/mdeff/fma
- Use `--skip-download` flag if files already exist

### Feature extraction takes too long
- Use FMA Small instead of FMA Medium
- Use `process_audio_files.py` with pre-computed features
- Run in Docker with more CPU cores

---

## See Also

- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Full deployment documentation
- [README.md](../README.md) - Project overview
- [docker/README.md](../docker/README.md) - Docker configuration
