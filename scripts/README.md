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

### `download_fma.py` - Unified FMA Dataset Downloader
**Purpose:** Download and setup any FMA dataset size with a single script

**Usage:**
```bash
# Download small dataset (default) - 8GB, 8K tracks
python scripts/download_fma.py

# Download specific size
python scripts/download_fma.py --size small     # 8GB, 8K tracks, 8 genres
python scripts/download_fma.py --size medium    # 25GB, 25K tracks, 16 genres
python scripts/download_fma.py --size large     # 93GB, 106K tracks, 161 genres
python scripts/download_fma.py --size full      # 879GB, full audio

# Skip download if files already exist
python scripts/download_fma.py --skip-download

# See all options
python scripts/download_fma.py --help
```

**Dataset Sizes:**

| Size | Tracks | Genres | Size | Best For |
|------|--------|--------|------|----------|
| **small** | 8,000 | 8 balanced | ~8GB | Testing, learning, quick iteration |
| **medium** | 25,000 | 16 unbalanced | ~25GB | Standard training, experiments |
| **large** | 106,574 | 161 | ~93GB | Production, research, best accuracy |
| **full** | 106,574 | 161 | ~879GB | Full audio quality (30s clips in others) |

**What it does:**
1. Creates directory structure: `data/raw/`, `data/processed/`, `data/metadata/`
2. Downloads selected FMA dataset from official source
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
│   └── ... (numbered directories)
└── metadata/
    ├── tracks.csv
    ├── genres.csv
    ├── features.csv
    └── ...
```

**Download Times (approximate):**
- **small:** 30-60 minutes
- **medium:** 1-2 hours
- **large:** 3-6 hours
- **full:** 12-24+ hours

---

## Feature Extraction Scripts

### `extract_audio_features.py`
**Purpose:** Extract mel-spectrograms and audio features from raw audio files for EDA

**Usage:**
```bash
# Extract features (uses all CPU cores by default, top-level genres)
python scripts/extract_audio_features.py

# Use detailed subgenres instead of top-level genres (50+ genres)
python scripts/extract_audio_features.py --use-subgenres

# Specify number of workers for parallel processing
python scripts/extract_audio_features.py --num-workers 4

# Process more samples per genre
python scripts/extract_audio_features.py --samples-per-genre 200

# Limit total samples for quick testing
python scripts/extract_audio_features.py --max-samples 500

# Filter out rare genres (minimum samples per genre)
python scripts/extract_audio_features.py --use-subgenres --min-samples-per-genre 20

# Generate more spectrogram images (random mix)
python scripts/extract_audio_features.py --num-spectrogram-examples 200

# Generate spectrograms per genre (even distribution)
python scripts/extract_audio_features.py --use-subgenres --spectrograms-per-genre 10

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

**Genre Options:**
- **Default (no flag):** Uses 16 top-level genres (Rock, Electronic, Hip-Hop, etc.)
- **--use-subgenres:** Uses detailed subgenres (70+ genres like Tech-House, Thrash, etc.)
- **--min-samples-per-genre:** Filter out rare genres with too few samples

**Spectrogram Image Options:**
- **--num-spectrogram-examples N:** Generate N total spectrogram images (random mix, default: 50)
- **--spectrograms-per-genre N:** Generate N images per genre (even distribution, overrides num-spectrogram-examples)

**When to use:**
- For exploratory data analysis (EDA) and visualization
- After downloading FMA dataset
- To generate example spectrograms
- Use --use-subgenres to match your training data genres

**Performance:**
- Uses multiprocessing for fast extraction
- Default: 100 samples per genre
- Time: ~5-10 minutes for 1,500 samples with multiprocessing (vs 30+ minutes single-threaded)
- With --use-subgenres and 70 genres: ~15-20 minutes for 7,000 samples

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
python scripts/download_fma.py              # Downloads small (default)
# OR choose a specific size:
python scripts/download_fma.py --size medium
python scripts/download_fma.py --size large

# 2. Prepare spectrograms for CNN
python scripts/prepare_cnn_spectrograms.py

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

# 3. Download dataset
python scripts/download_fma.py                  # Downloads small (default)
python scripts/download_fma.py --size medium    # Or medium
python scripts/download_fma.py --size large     # Or large

# 4. Prepare spectrograms for CNN training
python scripts/prepare_cnn_spectrograms.py

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
1. install_requirements.sh/ps1         (one time - setup environment)
   ↓
2. download_fma.py                     (one time - download dataset)
   [--size small|medium|large]
   ↓
3. prepare_cnn_spectrograms.py         (one time - prepare spectrograms)
   ↓
4. train_multilabel_cnn.py             (train model)
   ↓
5. Start backend API                    (deploy)
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
