# Music Feature Analysis and Genre Classification

**CS 3120 Machine Learning Project**  
**Author:** Jarred Maestas  
**Fall 2025**

## Project Overview

This project applies machine learning techniques to music genre classification using the Free Music Archive (FMA) dataset. The goal is to compare traditional machine learning approaches using hand-crafted audio features against deep learning methods that learn features directly from raw spectrograms.

## Documentation & Maintenance

Foundational project docs live in the `docs/` folder and at the repo root:

- `QUICKSTART.md` - quick start to run locally and create a sample model
- `DEPLOYMENT.md` - docker and deployment guidance
- `LOCKING.md` - notes on creating reproducible conda/pip lockfiles
- `CONTRIBUTING.md` - how to contribute, code style, and PR process
- `MAINTENANCE.md` - release and maintenance checklist

Please open issues with the templates in `.github/ISSUE_TEMPLATE/` and use the PR template at `.github/PULL_REQUEST_TEMPLATE.md` for changes. For urgent fixes, tag maintainers in a PR.

If you'd like to contribute, see `CONTRIBUTING.md` for steps, local checks, and style guidelines.

Repository housekeeping:

- Issue templates: `.github/ISSUE_TEMPLATE/`
- PR template: `.github/PULL_REQUEST_TEMPLATE.md`
- Labels: `.github/ISSUE_LABELS.md`


**Dataset:** 8,000 30-second audio clips across 8 musical genres with pre-computed audio features and metadata.

## Motivation

Music data combines complex temporal and frequency patterns, offering unique challenges beyond standard classification datasets. This project explores how different machine learning approaches handle high-dimensional audio data and compares the effectiveness of hand-crafted features versus learned representations for genre classification.

## Technical Approach

### Feature Extraction
- **MFCCs** (Mel-Frequency Cepstral Coefficients): Capturing timbral characteristics
- **Spectral features**: Centroid, rolloff, bandwidth for frequency content analysis
- **Temporal features**: Zero-crossing rate, energy, rhythm patterns
- **Chroma vectors**: Harmonic and pitch content representation
- **Fourier transforms**: Time-frequency domain analysis via spectrograms

### Modeling Strategy

**Baseline Model**
- Random Forest classifier using extracted audio features
- Establishes performance benchmarks for multi-class genre classification

**Advanced Models**
1. **Convolutional Neural Network (CNN)**: Processes raw spectrograms for end-to-end learning, comparing learned features versus hand-crafted features
2. **Autoencoder + K-Means**: Unsupervised feature learning to discover latent music patterns, with cluster assignments enhancing supervised classification

### Evaluation
- Stratified 5-fold cross-validation
- Genre-weighted F1 scores (accounting for potential class imbalance)
- Comparison across feature representations: raw audio, hand-crafted features, learned embeddings

## Repository Structure

```
.
├── data/                    # Dataset and metadata (not tracked)
├── preprocessing/           # Audio loading and feature extraction
│   ├── audio_loader.py
│   ├── fourier_analysis.m
│   └── feature_extraction.py
├── models/                  # ML model implementations
│   ├── random_forest.py
│   ├── cnn_model.py
│   └── autoencoder.py
├── visualization/           # EDA and result plots
├── notebooks/              # Analysis notebooks
└── outputs/                # Generated features and results
```

## Tools and Technologies

**Languages:** Python 3.x, MATLAB

**Key Libraries:**
- **Audio Processing:** librosa, scipy.signal
- **Machine Learning:** scikit-learn, PyTorch
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn, librosa.display
- **Dimensionality Reduction:** umap-learn

## Current Progress

**Completed:**
- Dataset acquisition and organization (8,000 tracks, 8 genres)
- Development environment setup
- Fourier transform pipeline implementation
- Audio normalization and batch processing utilities
- Initial spectral analysis and spectrogram generation
- Basic temporal feature extraction (zero-crossing rate, energy)

**In Progress:**
- Full MFCC feature extraction (13 coefficients per frame)
- Spectral feature calculations (centroid, rolloff, bandwidth)
- Chroma feature extraction for harmonic analysis
- Feature aggregation into track-level representations

**Next Steps:**
- Complete comprehensive feature extraction for all tracks
- Exploratory data analysis with t-SNE/PCA visualizations
- Baseline Random Forest implementation
- CNN model development for spectrogram classification

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification

# Install Python dependencies
pip install librosa scikit-learn torch pandas numpy matplotlib seaborn umap-learn

# MATLAB dependencies
# Ensure Signal Processing Toolbox is available
```

## Usage

```python
# Feature extraction example
from preprocessing.feature_extraction import extract_features

features = extract_features('path/to/audio.mp3')
# Returns: MFCCs, spectral features, temporal features, chroma vectors
```

```python
# Model training example (coming soon)
from models.random_forest import train_model

model = train_model(features, labels, cv_folds=5)
```

## Expected Outcomes

- Comprehensive understanding of audio feature engineering for ML
- Practical comparison of traditional ML versus deep learning on audio data
- Analysis of which audio characteristics best predict musical genre
- Experience with CNN architectures for spectrogram classification
- Knowledge of unsupervised learning for feature discovery

## Dataset

**Free Music Archive (FMA)**
- 8,000 tracks, 30 seconds each
- 8 genres (rock, electronic, hip-hop, classical, jazz, folk, pop, experimental)
- Pre-computed features and rich metadata available

Dataset must be downloaded separately from [FMA repository](https://github.com/mdeff/fma).

## License

This project is for educational purposes as part of CS 3120 coursework.

## Contact

Jarred Maestas - CS 3120 Fall 2025

---

*Note: This is an active project. Repository structure and code will be updated as development progresses.*
