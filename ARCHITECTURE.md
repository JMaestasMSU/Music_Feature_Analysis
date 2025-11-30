# System Architecture - Music Feature Analysis

## Overview

This project demonstrates a complete machine learning pipeline from exploratory data analysis through production deployment, combining academic rigor with enterprise-grade software engineering.

```
┌─────────────────────────────────────────────────────────────┐
│                 GRADED DELIVERABLES (35 pts)                 │
│                                                              │
│  ┌────────────────────┐  ┌───────────────────────────────┐ │
│  │   Notebooks (20)   │  │  Presentation (15)            │ │
│  │                    │  │                               │ │
│  │  01_EDA.ipynb      │  │  presentation.pdf (9)         │ │
│  │  • Data exploration│  │  • 5 professional slides      │ │
│  │  • Feature analysis│  │                               │ │
│  │  • Visualizations  │  │  SUMMARY.md (6)               │ │
│  │  (15 pts)          │  │  • Findings & insights        │ │
│  │                    │  │  • Limitations                │ │
│  │  02_Modeling.ipynb │  │  • Future work                │ │
│  │  • Architecture    │  │                               │ │
│  │  • Training        │  │                               │ │
│  │  • Evaluation      │  │                               │ │
│  │  (5 pts)           │  │                               │ │
│  └────────────────────┘  └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         PRODUCTION COMPONENTS (Not Graded, Optional)         │
│                                                              │
│  Backend API (8000) ────► Model Server (8001)                │
│  • FastAPI              • PyTorch inference                 │
│  • Audio upload         • GPU-accelerated                   │
│  • Feature extraction   • Batch processing                  │
│  • Orchestration        • Model management                  │
│                                                              │
│  MATLAB/NumPy FFT ──────► Validation & Analysis             │
│  • Parseval's theorem   • Feature comparison                │
│  • Spectral analysis    • Numerical validation              │
└─────────────────────────────────────────────────────────────┘
```

---

## Grading Deliverables

### 1. Exploratory Data Analysis Notebook (15 pts)
**File**: `notebooks/01_EDA.ipynb`

**Content Required**:
- Dataset overview (size, samples, duration, genres)
- Data quality assessment (missing values, balance)
- Feature extraction explanation
- Statistical analysis by genre
- Visualizations (5+ charts)
- Preprocessing strategy
- Train/val/test split approach

**Grading Criteria**:
- Thorough exploration of dataset
- Appropriate visualizations
- Well-organized, commented code

### 2. Modeling Notebook (5 pts)
**File**: `notebooks/02_Modeling.ipynb`

**Content Required**:
- Model architecture explanation
- Training procedure
- Evaluation metrics (accuracy, precision, recall, F1)
- Performance results
- Limitations acknowledged

**Grading Criteria**:
- Basic evaluation metrics present
- Model performance clearly shown
- Limitations explained

### 3. Presentation Slides (9 pts)
**File**: `presentation/presentation.pdf`

**Slide Requirements** (5 slides, ~6 minutes):
1. **Project Overview** (3 pts)
   - Problem statement
   - Why interesting/important
   - Project goals

2. **Data Preprocessing** (3 pts)
   - Dataset description
   - Feature engineering steps
   - Data cleaning

3. **Models & Methods** (3 pts)
   - Model choice explanation
   - Why chosen
   - Alternative approaches considered

4. **Results & Evaluation** (3 pts)
   - Performance metrics
   - Key findings
   - Challenges encountered

5. **Conclusion & Future Work** (3 pts)
   - Lessons learned
   - Improvements possible
   - Next steps

### 4. Documentation Summary (6 pts)
**File**: `presentation/SUMMARY.md`

**Content Required**:
- Executive summary
- Key findings from EDA
- Model insights
- Feature importance
- Limitations
- Future improvements

---

## Production Components (Enterprise-Grade, Not Graded)

### Model Inference Server (FastAPI)
**Location**: `app/`

**Components**:
- `main.py` - FastAPI application with endpoints for:
  - Health checks
  - Model inference
  - Batch predictions
  - Model information
- `model.py` - Model loading, caching, and prediction logic
- `config.py` - Server configuration (port, host, model paths)
- `logging_config.py` - Structured logging setup

**Features**:
- RESTful API for genre prediction
- Model loading on startup with caching
- Error handling and validation
- Swagger/OpenAPI documentation at `/docs`

### Feature Extraction Module
**Location**: `preprocessing/`

**Components**:
- `feature_extraction.py` - Audio processing utilities:
  - Load audio files (librosa)
  - Extract spectral features (centroid, rolloff, spread)
  - Compute MFCCs
  - Calculate temporal features (ZCR, RMS)
  - Preprocessing and normalization

**Purpose**:
- Centralized feature extraction logic
- Shared by notebooks and API
- Ensures consistency across pipeline

### Backend API (Optional Separate Service)
**Location**: `backend/`

If separate from `app/`, provides:
- Audio file uploads
- Real-time genre predictions
- Batch processing
- Feature comparison (FFT vs ML)

---

## Data Flow

```
Audio File Upload
       │
       ▼
Feature Extraction (preprocessing/)
 • Load audio (librosa)
 • Normalize
 • Extract features (spectral, MFCC, temporal)
       │
       ├─────────────────────┬──────────────┐
       ▼                     ▼              ▼
  Raw Features        Preprocessed       FFT Analysis
  (20 features)       StandardScaled     (MATLAB/NumPy)
       │                     │              │
       └─────────────────────┼──────────────┘
                             ▼
                    Model Server (app/)
                             │
       ┌─────────────────────┴──────────────┐
       ▼                                    ▼
  Neural Network                   Validation Report
  • Genre: X                       • FFT features
  • Confidence: Y%                 • ML features
  • Top-3: [...]                   • Correlation
```

---

## Key Design Decisions

### 1. Separate Backend & Model Services
- **Backend** (CPU): Handles requests, orchestration
- **Model** (GPU): Heavy computation, inference only
- **Benefit**: Scale independently, optimal resource usage

### 2. FFT Validation
- Validates ML features against numerical analysis
- Demonstrates understanding of signal processing
- Ensures feature extraction correctness

### 3. Containerization
- Docker Compose for local development
- Easy deployment and reproducibility
- Kubernetes-ready for production scale

### 4. Clean Notebook Structure
- EDA separate from modeling
- Code well-commented
- Results clearly documented

---

## Testing Strategy

### Quick Tests (< 2 minutes)
Located: `tests/`

- Prove FFT works
- Prove CNN works
- Prove audio processing works
- Prove optimization works

### Integration Tests
Located: `backend/test_api.py`

- Prove API endpoints work
- Test with synthetic data
- Validate responses

---

## Technology Choices

| Decision | Choice | Reason |
|----------|--------|--------|
| Framework | PyTorch | Industry standard, GPU support |
| API | FastAPI | Modern, async, auto-documentation |
| Audio | librosa | De-facto standard, reliable |
| FFT | NumPy + MATLAB | Validates with gold standard |
| Notebooks | Jupyter | Interactive, reproducible |
| Presentation | RMarkdown | Reproducible, professional |
| Containers | Docker | Industry standard |

---

## Scalability Considerations

### Current (Development)
- Single machine deployment
- CPU/GPU on same host
- Local file storage

### Production (Kubernetes)
- Horizontal scaling: 3+ backend replicas
- GPU nodes for model server
- Database for result persistence
- Load balancer for traffic distribution
- Monitoring (Prometheus) and logging (ELK)

---

## Security Notes

Input validation on file uploads  
File size limits (50 MB max)  
CORS configuration  
Error handling (no sensitive data in responses)  
Container isolation  

---

See also:
- [DEPLOYMENT.md](DEPLOYMENT.md) - How to run
- [README.md](README.md) - Quick start
