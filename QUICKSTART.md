# Quick Start Guide - Music Feature Analysis

Get up and running in < 5 minutes.

---

## For Graders: View Deliverables

**Just want to grade the project?** Go directly to these files:

```bash
# Open graded notebooks
jupyter notebook notebooks/01_EDA.ipynb        # 15 points
jupyter notebook notebooks/02_Modeling.ipynb   # 5 points

# View presentation (submit day before)
open presentation/presentation.pdf             # 9 points

# Read project summary
open presentation/SUMMARY.md                   # 6 points
```

---

## For Students: Development Setup

### Prerequisites

- Python 3.9+
- Jupyter Notebook
- (Optional) Docker for deployment
- (Optional) MATLAB for FFT validation

### 1. Clone and Navigate

```bash
cd /Users/P2956632/Documents/CS\ 3120/Music_Feature_Analysis/
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

**Core requirements (for notebooks):**
```bash
pip install -r requirements.txt
```

**Model inference server:**
```bash
# app/ has its own dependencies
cd app/
pip install fastapi uvicorn torch numpy pandas scikit-learn librosa
cd ..
```

**Backend API (if separate):**
```bash
cd backend/
pip install -r requirements.txt
cd ..
```

### 4. Run Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to:
# - notebooks/01_EDA.ipynb (do this first)
# - notebooks/02_Modeling.ipynb (do this second)
```

---

## Verify Setup: Run Quick Tests

Prove that all components work (optional but recommended):

```bash
cd tests/
bash run_all_tests.sh
```

Expected output:
```
FFT validation test passed (10 sec)
Audio processing test passed (5 sec)
CNN architecture test passed (30 sec)
Bayesian optimization test passed (30 sec)

All tests completed in < 2 minutes
```

---

## Generate Presentation

### Option 1: Use RMarkdown (Recommended)

```bash
cd presentation/

# Install R dependencies (first time only)
Rscript -e "install.packages(c('rmarkdown', 'knitr'))"

# Compile slides to PDF
Rscript -e "rmarkdown::render('presentation.Rmd', output_format='pdf_document')"
```

### Option 2: Manual PDF Creation

1. Open `presentation/presentation.Rmd`
2. Copy content to Google Slides / PowerPoint
3. Export as PDF: `presentation/presentation.pdf`

---

## Run Model Inference Server

### Development Mode

```bash
cd app/
python main.py

# API runs at: http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Predict genre (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, ...]}'
```

**Example API call (PowerShell):**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get
```

**Example API call (curl):**
```bash
curl http://localhost:8000/api/v1/health
```

---

## Deploy with Docker (Optional)

### Build and Run

```bash
cd docker/
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- Backend API: http://localhost:8000
- Model Server: http://localhost:8001
- Health checks: Built-in monitoring

---

## Development Workflow

### 1. Work on EDA
```bash
jupyter notebook notebooks/01_EDA.ipynb

# Complete these sections:
# - Dataset overview
# - Feature extraction
# - Statistical analysis
# - Visualizations (5+)
# - Preprocessing strategy
```

### 2. Work on Modeling
```bash
jupyter notebook notebooks/02_Modeling.ipynb

# Complete these sections:
# - Model architecture
# - Training procedure
# - Evaluation metrics
# - Results
# - Limitations
```

### 3. Create Presentation
```bash
cd presentation/

# Edit presentation.Rmd
# Generate PDF
# Verify 5 slides present
```

### 4. Document Findings
```bash
# Edit presentation/SUMMARY.md
# Include:
# - Key findings
# - Model insights
# - Limitations
# - Future work
```

### 5. Final Check
```bash
cd tests/
bash run_all_tests.sh  # Verify everything works
```

---

## Submission Checklist

Before submitting, verify:

- [ ] `notebooks/01_EDA.ipynb` runs without errors
- [ ] `notebooks/02_Modeling.ipynb` runs without errors
- [ ] `presentation/presentation.pdf` exists (< 10 MB)
- [ ] `presentation/SUMMARY.md` is complete
- [ ] All code is commented
- [ ] All visualizations have titles/labels
- [ ] Limitations are documented
- [ ] README.md is up-to-date

---

## Troubleshooting

### Jupyter Won't Start
```bash
pip install --upgrade jupyter notebook
jupyter notebook --version
```

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Tests Fail
```bash
# Install test dependencies
pip install pytest numpy scipy librosa

# Run tests individually
cd tests/
python quick_fft_test.py
```

### Can't Generate PDF
```bash
# Option 1: Use online converter
# Upload .Rmd to https://pandoc.org/try/

# Option 2: Install Pandoc
brew install pandoc  # macOS
# or visit https://pandoc.org/installing.html
```

### Docker Issues
```bash
# Check Docker is running
docker --version

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## Additional Resources

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and grading info |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and data flow |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment guide |
| [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) | File structure guidelines |
| [tests/README.md](tests/README.md) | Testing documentation |

---

## Tips for Success

1. **Start with notebooks**: Complete `01_EDA.ipynb` before `02_Modeling.ipynb`
2. **Run cells sequentially**: Don't skip cells in notebooks
3. **Save often**: Jupyter autosaves, but manual save is safer
4. **Test early**: Run `tests/run_all_tests.sh` to catch issues
5. **Document as you go**: Add markdown cells explaining your work
6. **Check outputs**: Ensure all visualizations render correctly
7. **Review rubric**: See [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) for grading criteria

---

## Status Indicators

After setup, you should see:

[OK] Jupyter notebooks open  
[OK] All cells execute without errors  
[OK] Visualizations render correctly  
[OK] Tests pass (if run)  
[OK] Presentation PDF generated  
[OK] Summary document complete  

---

**Need help?** See [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) for detailed guidelines.

**Ready to submit?** Review checklist in [README.md](README.md#-submission-checklist).

---

**Last Updated**: 2024  
**Project Status**: Ready for Development
