# Music Feature Analysis - Advanced Genre Classification

A music genre classification system using multi-label CNNs.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)

---

## Overview

A music genre classification system that scales from 8 to 100+ genres without code changes.

**Key Features:**
- Multi-label CNN (songs can have multiple genres)
- ResNet-style architecture with attention mechanisms
- SpecAugment & Mixup data augmentation
- Config-based training (YAML files)
- REST API with Docker deployment
- Bayesian hyperparameter optimization

---

## Project Structure

```
Music_Feature_Analysis/
├── README.md                          ← You are here
├── ADVANCED_CNN_GUIDE.md              ← Complete technical reference
├── ARCHITECTURE.md                    ← System design
├── DEPLOYMENT.md                      ← Production deployment
├── PROPOSAL.md                        ← Original project proposal
│
├── notebooks/                         JUPYTER NOTEBOOKS
│   ├── 01_EDA.ipynb                   Grading deliverable: EDA
│   ├── 02_Modeling.ipynb              Grading deliverable: Modeling
│   ├── cnn_development.ipynb          Original CNN (8 genres)
│   ├── multilabel_cnn_demo.ipynb      Advanced demo (50+ genres)
│   └── bayesian_optimization.ipynb    Hyperparameter tuning
│
├── models/                            NEURAL NETWORKS
│   ├── cnn_model.py                   AudioCNN + MultiLabelAudioCNN
│   ├── audio_augmentation.py          SpecAugment, Mixup, etc.
│   ├── bayesian_optimizer.py          Hyperparameter search
│   ├── random_forest.py               Baseline comparison
│   └── trained_models/                Saved checkpoints
│
├── scripts/                           TRAINING & UTILITIES
│   ├── train_multilabel_cnn.py        Production training script
│   ├── feature_extraction.py          Audio preprocessing
│   └── ...
│
├── configs/                           EXPERIMENT CONFIGS
│   ├── multilabel_50genres.yaml       50-genre config
│   └── baseline_8genres.yaml          8-genre baseline
│
├── backend/                           REST API
│   ├── app.py                         FastAPI application
│   ├── routes/                        API endpoints
│   └── services/                      Model inference
│
├── data/                              DATASETS (not in repo)
│   ├── raw/                           Original audio files
│   └── processed/                     Spectrograms
│
├── docker/                            CONTAINERIZATION
│   └── docker-compose.yml             Multi-service deployment
│
└── tests/                             TESTING
    └── ...
```

---

## Quick Start

### Option 1: Interactive Demo (5 min)

```bash
# Open the advanced CNN demo
jupyter notebook notebooks/multilabel_cnn_demo.ipynb

# See:
# - Architecture comparison (baseline vs advanced)
# - Data augmentation examples
# - Multi-label training
# - Embedding visualization
```

### Option 2: Train a Model (30 min)

```bash
# Train on 50 genres
python scripts/train_multilabel_cnn.py \
    --config configs/multilabel_50genres.yaml

# Or 100 genres - just change the config!
python scripts/train_multilabel_cnn.py \
    --num-genres 100 \
    --epochs 100
```

### Option 3: Deploy the API

```bash
# Start backend
cd backend
python app.py

# API available at http://localhost:8000
# Upload audio for prediction:
curl -X POST http://localhost:8000/api/v1/analysis/predict \
     -F "file=@song.mp3"
```

---

## What's New vs Baseline

| Feature | Baseline (Notebooks) | Advanced (Production) |
|---------|---------------------|----------------------|
| **Genres** | 8 (hardcoded) | Unlimited (config) |
| **Classification** | Single-label | Multi-label |
| **Architecture** | 4 conv layers | 20+ layers (ResNet) |
| **Residual Blocks** | | |
| **Attention** | | Channel attention |
| **Augmentation** | | SpecAugment + Mixup |
| **Training** | Notebook only | Production pipeline |
| **Configuration** | Hardcoded | YAML configs |
| **Hyperparameter Tuning** | Manual | Bayesian optimization |

---

## Core Components

### 1. Multi-Label CNN ([models/cnn_model.py](models/cnn_model.py))

```python
from models.cnn_model import MultiLabelAudioCNN

# Create model for any number of genres
model = MultiLabelAudioCNN(
    num_genres=50,        # Configurable
    base_channels=64,     # Network width
    use_attention=True    # Channel attention
)

# Scales gracefully:
# 8 genres   → 3.2M params
# 50 genres  → 3.5M params (+9%)
# 100 genres → 3.6M params (+12%)
```

**Features:**
- ResNet-style residual blocks (20+ layers)
- Channel attention (learns important frequencies)
- Multi-label output (sigmoid per genre)
- Embedding extraction (similarity search)

### 2. Data Augmentation ([models/audio_augmentation.py](models/audio_augmentation.py))

```python
from models.audio_augmentation import SpectrogramAugmentation

aug = SpectrogramAugmentation()

# SpecAugment (Google's technique)
augmented = aug.spec_augment(spectrogram)

# Mixup (blend two samples)
mixed, lambda_val = aug.mixup(spec1, spec2)
```

**Impact:** 5-10x effective dataset increase.

### 3. Bayesian Optimization ([models/bayesian_optimizer.py](models/bayesian_optimizer.py))

```python
from models.bayesian_optimizer import BayesianOptimizer

# Automatically tune hyperparameters
optimizer = BayesianOptimizer(
    param_space={
        'learning_rate': (1e-5, 1e-2),
        'batch_size': [16, 32, 64, 128],
        'base_channels': [32, 64, 128]
    }
)

best_params = optimizer.optimize(n_trials=50)
```

**Use Cases:**
- Find optimal learning rate
- Tune architecture (depth, width)
- Balance precision vs recall (threshold tuning)
- Compare against grid search (Bayesian is 10x faster)

**Future Improvements:**
- Add early stopping based on validation loss
- Support for categorical hyperparameters (optimizer type)
- Multi-objective optimization (accuracy + speed)
- Integration with training script

### 4. Production Training ([scripts/train_multilabel_cnn.py](scripts/train_multilabel_cnn.py))

```bash
# Config-based training
python scripts/train_multilabel_cnn.py \
    --config configs/multilabel_50genres.yaml

# Saves:
# - Best model checkpoint
# - Training history (JSON)
# - Training curves (PNG)
# - Test metrics
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | This file - main overview |
| [ADVANCED_CNN_GUIDE.md](ADVANCED_CNN_GUIDE.md) | Complete technical reference |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment guide |

---

## Grading Deliverables

**Required notebooks (unchanged):**
- [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb) - Exploratory Data Analysis
- [notebooks/02_Modeling.ipynb](notebooks/02_Modeling.ipynb) - Baseline modeling (8 genres)

**Additional notebooks (show advanced work):**
- [notebooks/multilabel_cnn_demo.ipynb](notebooks/multilabel_cnn_demo.ipynb) - Advanced CNN demo
- [notebooks/bayesian_optimization.ipynb](notebooks/bayesian_optimization.ipynb) - Hyperparameter tuning
- [notebooks/cnn_development.ipynb](notebooks/cnn_development.ipynb) - Original CNN work

---

## Example Workflows

### Workflow 1: Baseline Comparison

```bash
# 1. Train baseline (8 genres, no attention)
python scripts/train_multilabel_cnn.py \
    --config configs/baseline_8genres.yaml

# 2. Train advanced (50 genres, with attention)
python scripts/train_multilabel_cnn.py \
    --config configs/multilabel_50genres.yaml

# 3. Compare results
ls -ltr models/trained_models/*/test_results.json
```

### Workflow 2: Hyperparameter Tuning

```bash
# Option A: Bayesian optimization (recommended)
jupyter notebook notebooks/bayesian_optimization.ipynb

# Option B: Grid search
for lr in 0.0001 0.001 0.01; do
    python scripts/train_multilabel_cnn.py \
        --lr $lr \
        --experiment-name "lr_${lr}_50genres"
done
```

### Workflow 3: Scale to More Genres

```bash
# Easy - just change the config!
python scripts/train_multilabel_cnn.py \
    --config configs/multilabel_50genres.yaml \
    --num-genres 100 \
    --experiment-name "cnn_100genres"
```

---

## Troubleshooting

**Out of memory:**
```yaml
# Reduce batch size in config
batch_size: 16  # Default: 32
```

**Model not improving:**
```yaml
# Increase epochs or lower learning rate
epochs: 200
lr: 0.0001
```

**Need more data:**
```python
# Enable augmentation (automatic in training script)
# Increases effective dataset size 5-10x
```

---

## Performance Expectations

| Dataset Size | Genres | Training Time (GPU) | Expected F1 |
|-------------|--------|---------------------|-------------|
| 1K samples | 8 | ~5 min | 0.75-0.85 |
| 5K samples | 20 | ~15 min | 0.70-0.80 |
| 10K samples | 50 | ~30 min | 0.65-0.75 |
| 50K+ samples | 100+ | ~2 hrs | 0.70-0.85 |

---

## Getting Real Data

**Free Music Archive (FMA):**
```bash
# Easy setup with our unified script
python scripts/download_fma.py              # Small: 8K tracks, 8GB
python scripts/download_fma.py --size medium    # Medium: 25K tracks, 25GB
python scripts/download_fma.py --size large     # Large: 106K tracks, 93GB
```

**Manual download:**
```bash
# If you prefer manual download
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
```

**Million Song Dataset:**
- 1M songs with genre tags
- Largest public dataset
- Audio features pre-computed

---

## Key Features

- Unlimited genres - Scale from 8 to 200+ by editing config
- Multi-label classification - Songs can have multiple genres
- Docker, API, deployment included
- ResNet, Attention, SpecAugment, Mixup
- Bayesian optimization - Automated hyperparameter tuning
- Backwards compatible - Original notebooks still work

---

## Future Enhancements

### Phase 1: Advanced Models
- [ ] LSTM for temporal modeling
- [ ] Transfer learning (VGGish/YAMNet)
- [ ] Ensemble methods

### Phase 2: Real-time Systems
- [ ] WebSocket streaming API
- [ ] Real-time dashboard
- [ ] Edge deployment (mobile)

### Phase 3: Explainability
- [ ] Grad-CAM visualization
- [ ] Feature importance
- [ ] Genre decision boundaries

### Phase 4: Production Scale
- [ ] A/B testing framework
- [ ] Distributed training
- [ ] Model versioning (MLflow)

---

## License

Academic use only (CS 3120 project)
