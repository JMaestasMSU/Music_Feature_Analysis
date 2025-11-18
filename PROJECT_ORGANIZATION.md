# Project Organization Guidelines

This document explains the project structure, what goes where, and how to prepare deliverables for grading.

---

## Directory Structure

### Graded Deliverables (35 points total)

**For professors**: Grade **only** these files.

```
notebooks/
├── 01_EDA.ipynb              ← GRADE: 15 points
│   ├── Dataset overview
│   ├── Feature extraction
│   ├── Statistical analysis
│   ├── 5+ visualizations
│   └── Preprocessing strategy
│
└── 02_Modeling.ipynb         ← GRADE: 5 points
    ├── Neural network architecture
    ├── Training procedure
    ├── Evaluation metrics
    ├── Results analysis
    └── Limitations

presentation/
├── presentation.pdf          ← GRADE: 9 points
│   ├── Slide 1: Overview
│   ├── Slide 2: Data preprocessing
│   ├── Slide 3: Models & methods
│   ├── Slide 4: Results
│   └── Slide 5: Conclusion
│
└── SUMMARY.md                ← GRADE: 6 points
    ├── Executive summary
    ├── Key findings
    ├── Model insights
    └── Limitations & future work
```

**Submission Requirements**:
- [ ] Both notebooks execute without errors
- [ ] All visualizations have titles and labels
- [ ] Code is well-commented
- [ ] Presentation is PDF format (< 10 MB)
- [ ] Summary is comprehensive and well-written

### Production Code (Not Graded)

Demonstrates enterprise-grade practices:

```
app/                          Model inference server (FastAPI)
├── main.py                   FastAPI application entry point
├── model.py                  Model loading & prediction logic
├── config.py                 Server configuration
├── logging_config.py         Logging configuration
└── __init__.py              Package initialization

preprocessing/                Feature extraction utilities
├── feature_extraction.py     Audio feature extraction functions
└── __init__.py              Package initialization

backend/                      REST API server (if separate from app/)
├── app.py                    Alternative FastAPI application
├── config.py                 Configuration management
├── routes/                   API endpoints
├── services/                 Core business logic
└── test_api.py              API testing

models/                       ML implementation
├── genre_classifier.py       Neural network model
├── model_utils.py           Model save/load utilities
├── cnn_model.py             CNN architecture (legacy)
├── bayesian_optimizer.py    Hyperparameter tuning
├── requirements.txt         Model dependencies
├── README.md                Model documentation
└── trained_models/          Saved weights
    └── genre_classifier_production.pt

matlab/                       Numerical analysis
├── fft_validation.m         FFT feature extraction
├── spectral_analysis.m      Spectrogram analysis
└── signal_processing.m      Windowing, filtering

docker/                       Containerization
├── Dockerfile.backend       FastAPI container
├── Dockerfile.model         Model server container
└── docker-compose.yml       Orchestration
```

### Data Storage (Not in Git)

```
data/                             Local data storage (excluded from git)
├── README.md                     Data documentation
├── .gitkeep                      Ensures directory exists
├── raw/                          Original audio files (~8 GB)
│   └── fma_small/                FMA dataset audio files
│       ├── 000/
│       ├── 001/
│       └── ...
├── processed/                    Extracted features & splits
│   ├── features.csv              Audio features for all tracks
│   └── preprocessed_data.pkl     Train/val/test split
└── metadata/                     Dataset metadata (~5 MB)
    ├── tracks.csv                Track information & genres
    ├── genres.csv                Genre definitions
    └── features.csv              Pre-computed audio features
```

**Why excluded from git:**
- Audio files are large (~8 GB total)
- Features can be regenerated
- Users download their own copy
- Prevents repo bloat

**See [data/README.md](data/README.md) for download instructions.**

### Testing (`tests/`)

Quick integration tests (not graded, but proves system works):

```
tests/
├── README.md                 Test documentation
├── run_all_tests.sh         Run all tests at once
├── quick_fft_test.py        (~10 sec)
├── quick_cnn_test.py        (~30 sec)
├── quick_audio_processing_test.py  (~5 sec)
└── quick_bayesian_test.py   (~30 sec)
```

**Purpose**: Demonstrate components work before grading.

---

## File Organization Rules

### DO Include in Grading Directories

**`notebooks/`**:
- Complete, well-documented Jupyter notebooks
- Clear section headers
- Explanatory text before code
- Output cells showing results
- Visualizations with titles/labels
- Conclusions and insights

**`presentation/`**:
- Final PDF file
- High-quality visualizations
- Markdown summary document
- Professional formatting

### DON'T Include in Grading Directories

- Scratch/experimental notebooks
- Raw data files
- Log files
- Temporary files
- Development-only scripts

### DO Keep in `backend/`, `models/`, `docker/`

- Clean, production-ready code
- Proper error handling
- Configuration files
- Documentation

### DON'T Keep in Production Directories

- Jupyter notebooks (use `.py` files)
- Experimental code
- Debug scripts
- Unfinished features

---

## Workflow

### For Students

1. **Develop in Notebooks**
   - Work in `notebooks/01_EDA.ipynb`
   - Work in `notebooks/02_Modeling.ipynb`
   - Clean up before submission

2. **Create Presentation**
   - Edit `presentation/presentation.Rmd`
   - Compile to PDF
   - Save as `presentation/presentation.pdf`

3. **Document Findings**
   - Write `presentation/SUMMARY.md`
   - Include key findings
   - Document limitations

4. **Test (Optional)**
   - Run tests: `cd tests/ && bash run_all_tests.sh`
   - Validates system works
   - Demonstrates mastery

5. **Submit**
   - Upload PDF to Canvas
   - Include notebooks
   - Include SUMMARY.md

### For Graders

1. **Review Notebooks**
   - Open `notebooks/01_EDA.ipynb` → Grade EDA (15 pts)
   - Open `notebooks/02_Modeling.ipynb` → Grade Modeling (5 pts)
   - Use rubric in [README.md](README.md)

2. **Review Presentation**
   - Open `presentation/presentation.pdf` → Grade slides (9 pts)
   - Check 5 slides present
   - Verify content per rubric

3. **Review Documentation**
   - Read `presentation/SUMMARY.md` → Grade summary (6 pts)
   - Check findings documented
   - Verify limitations addressed

4. **Optional**: Run Tests
   - `cd tests/ && bash run_all_tests.sh`
   - Demonstrates system integrity
   - Shows understanding of components

---

## Detailed Grading Rubric

### EDA Notebook (15 points)

| Component | Points | Requirements |
|-----------|--------|--------------|
| **Dataset Overview** | 3 | Size, genres, duration, structure |
| **Feature Extraction** | 4 | Spectral, MFCC, temporal features explained |
| **Statistical Analysis** | 3 | Genre comparison, feature distributions |
| **Visualizations** | 3 | 5+ charts, properly labeled |
| **Preprocessing** | 2 | Train/val/test split, normalization |

### Modeling Notebook (5 points)

| Component | Points | Requirements |
|-----------|--------|--------------|
| **Architecture** | 2 | Clear explanation of model design |
| **Evaluation** | 2 | Accuracy, precision, recall, F1-score |
| **Results & Limitations** | 1 | Performance + acknowledged constraints |

### Presentation Slides (9 points)

| Component | Points | Requirements |
|-----------|--------|--------------|
| **Slide 1: Overview** | 1.8 | Problem statement, motivation, goals |
| **Slide 2: Data** | 1.8 | Dataset description, preprocessing |
| **Slide 3: Models** | 1.8 | Architecture, methods, rationale |
| **Slide 4: Results** | 1.8 | Performance metrics, key findings |
| **Slide 5: Conclusion** | 1.8 | Lessons learned, future work |
| **Overall Quality** | 1.8 | Professional appearance, clarity |

### Documentation Summary (6 points)

| Component | Points | Requirements |
|-----------|--------|--------------|
| **Executive Summary** | 1 | Concise project overview |
| **EDA Findings** | 1.5 | Key insights from data exploration |
| **Model Insights** | 1.5 | Performance analysis, patterns |
| **Limitations & Future** | 2 | Honest assessment, improvements |

---

## What Goes Where

### `notebooks/`
**Purpose**: Graded deliverables

Include:
- Complete EDA analysis
- Model training and evaluation
- Clear explanations
- Publication-quality visualizations
- Well-commented code

Exclude:
- Experimental notebooks
- Scratch work
- Development debugging
- Raw data

### `presentation/`
**Purpose**: Final deliverables

Include:
- PDF slides
- Markdown summary
- Generated figures
- Project findings

Exclude:
- Source (Rmd) files
- Working files
- Uncompiled documents

### `backend/`, `models/`, `docker/`
**Purpose**: Production code (demonstrates mastery)

Include:
- Clean, production-ready code
- Error handling
- Configuration
- Documentation

Exclude:
- Jupyter notebooks
- Test data
- Logs
- Temporary files

### `tests/`
**Purpose**: Quick validation (optional)

Include:
- Fast test scripts (< 2 min total)
- Self-contained tests
- Clear output

Exclude:
- Slow tests
- Real data files
- Integration tests

---

## Pre-Submission Checklist

### Notebooks
- [ ] `01_EDA.ipynb` is complete
- [ ] `02_Modeling.ipynb` is complete
- [ ] Both notebooks run without errors
- [ ] All cells have output
- [ ] Code is commented
- [ ] Visualizations have titles/labels

### Presentation
- [ ] `presentation.pdf` exists
- [ ] 5 slides present (overview, data, models, results, conclusion)
- [ ] All content from rubric covered
- [ ] Professional appearance
- [ ] File < 10 MB

### Documentation
- [ ] `SUMMARY.md` exists
- [ ] Key findings documented
- [ ] Limitations acknowledged
- [ ] Future work suggested
- [ ] Well-formatted

### General
- [ ] All files in correct directories
- [ ] No sensitive data exposed
- [ ] README.md up-to-date
- [ ] Code well-organized
- [ ] No error messages in notebooks

---

## If You Have Questions

- See [README.md](README.md) for overview
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [DEPLOYMENT.md](DEPLOYMENT.md) for how to run
- See [tests/README.md](tests/README.md) for testing

---

**Last Updated**: 2024
