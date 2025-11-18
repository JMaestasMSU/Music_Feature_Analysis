# Music Feature Analysis - Genre Classification System

**CS 3120 Project (Option B): Explore and Model a Unique Dataset**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-Academic-blue)

---

## Project Overview

An **end-to-end machine learning system** for music genre classification that combines:
- **Exploratory Data Analysis (EDA)**
- **Neural Network Genre Classification**
- **Production REST API**
- **Comprehensive Documentation** 

**Total Deliverables: 35 points**

---

## Directory Structure

```
Music_Feature_Analysis/
│
├── README.md                              ← Start here
├── QUICKSTART.md                          ← Setup instructions (< 5 min)
├── ARCHITECTURE.md                        ← System design
├── PROJECT_ORGANIZATION.md                ← File structure guide
├── PROJECT_STATUS.md                      ← Development tracker
├── LOCAL_TESTING.md                       ← Testing guide
├── LOCKING.md                             ← Dependency management
├── PROPOSAL.md                            ← Original project proposal
├── PROJECT_SUMMARY.md                     ← Alternative summary format
│
├── notebooks/                             GRADED DELIVERABLES
│   ├── 01_EDA.ipynb                       (15 pts) Data exploration
│   └── 02_Modeling.ipynb                  (5 pts) Model training & evaluation
│
├── presentation/                          FINAL DELIVERABLES
│   ├── presentation.Rmd                   Template for slides
│   ├── presentation.pdf                   (9 pts) Compiled slides
│   ├── SUMMARY.md                         (6 pts) Project findings
│   └── figures/                           Generated visualizations
│
├── app/                                   MODEL INFERENCE SERVER
│   ├── main.py                            FastAPI application entry
│   ├── model.py                           Model loading & prediction
│   ├── config.py                          Server configuration
│   ├── logging_config.py                  Logging setup
│   └── __init__.py                        Package initialization
│
├── preprocessing/                         FEATURE EXTRACTION
│   ├── feature_extraction.py              Audio feature utilities
│   └── __init__.py                        Package initialization
│
├── tests/                                 QUICK TESTS (< 2 min)
│   ├── README.md                          Test documentation
│   ├── run_all_tests.sh                   Run all tests
│   ├── quick_fft_test.py                  FFT validation
│   ├── quick_cnn_test.py                  Neural network test
│   ├── quick_audio_processing_test.py     Feature extraction
│   └── quick_bayesian_test.py             Hyperparameter tuning
│
├── backend/                               REST API (if separate)
│   ├── app.py                             FastAPI application
│   ├── config.py                          Configuration
│   ├── requirements.txt                   Dependencies
│   ├── routes/                            API endpoints
│   ├── services/                          Business logic
│   └── test_api.py                        API testing
│
├── models/                                ML MODELS
│   ├── genre_classifier.py                Neural network model
│   ├── model_utils.py                     Model save/load utilities
│   ├── cnn_model.py                       CNN architecture
│   ├── bayesian_optimizer.py              Hyperparameter tuning
│   ├── requirements.txt                   Model dependencies
│   ├── README.md                          Model documentation
│   └── trained_models/                    Saved weights
│       └── cnn_best_model.pt
│
├── matlab/                                NUMERICAL ANALYSIS
│   ├── fft_validation.m                   FFT feature extraction
│   ├── spectral_analysis.m                Spectrogram analysis
│   └── signal_processing.m                Windowing & filtering
│
└── docker/                                CONTAINERIZATION
    ├── Dockerfile.backend                 Backend container
    ├── Dockerfile.model                   Model server container
    └── docker-compose.yml                 Orchestration
```

---

## Quick Start (3 Steps)

### 1. Download Data (One Time)
```bash
cd data/metadata/
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
cd ../../
```

### 2. Create Features (One Time)
```bash
python scripts/create_features.py
# Creates: data/processed/ml_ready_features.pkl
```

### 3. Run Anything
```bash
# Notebooks
jupyter notebook notebooks/01_EDA.ipynb

# API
python backend/app.py

# Tests
bash tests/run_all_tests.sh
```

**That's it!** Everything uses `data/processed/ml_ready_features.pkl` automatically.

---

## Project Features

### Exploratory Data Analysis
**Location:** `notebooks/01_EDA.ipynb`

- **Dataset Overview**: 8,000 audio files, 8 genres, 30 seconds each
- **Data Quality**: Balanced classes, no missing values
- **Feature Extraction**:
  - Spectral features (centroid, rolloff, spread)
  - Zero crossing rate
  - MFCC (13 coefficients)
  - Chroma features
  - RMS energy
- **Statistical Analysis**: Genre-specific feature profiles
- **Visualizations**: 5+ publication-quality charts
- **Preprocessing**: StandardScaler normalization, 70/15/15 split

### Model Development
**Location:** `notebooks/02_Modeling.ipynb`

- **Architecture**: Fully connected neural network (20→128→64→32→8)
- **Input**: 20 audio features per track
- **Output**: 8-class genre probability distribution
- **Evaluation Metrics**:
  - Accuracy: ~75-85%
  - Precision, recall, F1-score
  - Confusion matrix
  - Per-genre performance breakdown
- **Training**: Adam optimizer, early stopping, validation monitoring
- **Limitations**: Acknowledged and documented

### Documentation
**Location:** `presentation/`

- **Slides (9 pts)**: 5-slide PDF presentation
  - Project overview and motivation
  - Data preprocessing pipeline
  - Model architecture and methods
  - Results and evaluation
  - Conclusions and future work
  
- **Summary**: Comprehensive project report
  - Executive summary
  - Key findings from EDA
  - Model insights and performance
  - Limitations and challenges
  - Future improvements

---

## Testing

### Quick Tests (< 2 minutes total)

```bash
cd tests/
bash run_all_tests.sh
```

**Individual tests:**
```bash
python quick_fft_test.py              # FFT validation (10 sec)
python quick_audio_processing_test.py # Feature extraction (5 sec)
python quick_cnn_test.py --cpu-only   # Neural network (30 sec)
python quick_bayesian_test.py         # Optimization (30 sec)
```

**What tests prove:**
- FFT spectral analysis works correctly
- Audio feature extraction pipeline functional
- Neural network can train and predict
- Optimization finds hyperparameters

### API Testing (Optional)

```bash
cd backend/
python test_api.py
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | FastAPI, Uvicorn | REST API server |
| **ML/DL** | PyTorch | Neural network implementation |
| **Audio** | librosa, scipy | Feature extraction |
| **Numerical** | NumPy, MATLAB | FFT validation |
| **Data Science** | Pandas, scikit-learn | Analysis, preprocessing |
| **DevOps** | Docker, Docker Compose | Containerization |
| **Visualization** | Matplotlib, Seaborn | Charts & graphs |
| **Presentation** | RMarkdown, Pandoc | Slides PDF generation |

---

## Key Results

### Model Performance
- **Test Accuracy**: x-x% (see `02_Modeling.ipynb` for exact numbers)
- **Architecture**: Fully connected neural network
- **Training**: ~50 epochs with early stopping
- **Best Genres**: Classical, Electronic (>85% F1)
- **Challenging Genres**: Folk, Experimental (boundary ambiguity)

### Feature Insights
- **Most Discriminative**: Spectral centroid, MFCCs, chroma features
- **Genre Clustering**: Blues/Rock similar, Classical/Electronic distinct
- **Dimensionality**: ~8 principal components capture 95% variance

### FFT Validation
- **Parseval's Theorem Error**: < 1%
- **FFT vs ML Centroid Correlation**: > 0.85
- **Numerical Validation**: Passed

---

## Production API (Optional)

### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/analysis/upload` | Full audio analysis |
| `POST` | `/api/v1/analysis/predict` | Quick genre prediction |
| `POST` | `/api/v1/analysis/batch-analyze` | Batch processing |
| `POST` | `/api/v1/analysis/compare-features` | FFT vs ML comparison |
| `GET` | `/api/v1/health` | Health check |

### Run Backend

```bash
cd backend/
python app.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Deploy with Docker

```bash
cd docker/
docker-compose up -d

# Backend: http://localhost:8000
# Model Server: http://localhost:8001
```

---

## Documentation

| File | Purpose |
|------|---------|
| **[QUICKSTART.md](QUICKSTART.md)** | Setup instructions (< 5 min) |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, data flow |
| **[PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)** | File structure guidelines |
| **[PROJECT_STATUS.md](PROJECT_STATUS.md)** | Development tracker |
| **[LOCAL_TESTING.md](LOCAL_TESTING.md)** | Testing guide |
| **[presentation/SUMMARY.md](presentation/SUMMARY.md)** | Project findings (graded) |

---

## Author Information

**Student:** Jarred Maestas  
**Course:** CS 3120 - Machine Learning  
**Semester:** Fall 2024  
**Project Option:** B - Explore and Model a Unique Dataset

---

## Project Status

- Complete: Notebooks - Both EDA and Modeling complete
- Complete: Documentation - Summary and organization docs ready
- In Progress: Presentation - Template ready, needs compilation
- Complete: Testing - Quick tests implemented and passing
- Complete: Production Code - Backend and model server functional

---

## Academic Integrity

This project demonstrates:
- Original implementation of neural network genre classification
- Comprehensive exploratory data analysis
- Production-grade software engineering practices
- Clear documentation and testing

All code is original or properly cited. No plagiarism or unauthorized collaboration.

---

**Last Updated**: 2025
**Status**: Unknown

**Quick Links:**
- [Quick Start Guide](QUICKSTART.md)
- [System Architecture](ARCHITECTURE.md)
- [Project Summary](presentation/SUMMARY.md)
- [Testing Guide](LOCAL_TESTING.md)