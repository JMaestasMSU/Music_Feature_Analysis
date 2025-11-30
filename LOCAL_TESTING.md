# Local Testing Guide

This guide explains how to test all project components locally without GPU resources.

---

## Testing Strategy

### What Can Be Tested Locally

**Backend API** - REST endpoints, request handling  
**CNN Architecture** - Model instantiation, forward/backward pass (CPU)  
**Audio Processing** - Feature extraction pipeline  
**FFT Validation** - Spectral analysis (NumPy/SciPy, no MATLAB needed)  
**Integration** - Component interactions  

### What Requires GPU (GitHub Actions)

**Full Model Training** - Train CNN on complete dataset  
**Batch Inference** - Process 1000+ audio files  
**Hyperparameter Tuning** - Bayesian optimization  

---

## Quick Start - Local Testing

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Backend dependencies
cd backend/
pip install -r requirements.txt
cd ..

# Test dependencies
pip install pytest numpy scipy torch torchvision
```

### 2. Run All Tests (< 2 minutes)

```bash
cd tests/
bash run_all_tests.sh
```

**Expected Output**:
```
==========================================
Music Feature Analysis - Test Suite
==========================================

Running FFT Validation... PASSED
Running Audio Processing... PASSED
Running CNN Architecture (CPU)... PASSED
Running Bayesian Optimization... PASSED

==========================================
Test Summary
==========================================
Tests Passed: 4
Tests Failed: 0
Duration: 85s

All tests passed!
```

### 3. Run Individual Tests

```bash
cd tests/

# FFT validation (~10 sec)
python quick_fft_test.py

# Audio processing (~5 sec)
python quick_audio_processing_test.py

# CNN architecture (~30 sec)
python quick_cnn_test.py --cpu-only

# Bayesian optimization (~30 sec)
python quick_bayesian_test.py
```

---

## Component Testing Details

### Backend API Testing

```bash
cd backend/

# Install dependencies
pip install -r requirements.txt

# Run API tests
python test_api.py

# Start server manually
python app.py &

# Test endpoints
curl http://localhost:8000/api/v1/health
```

### Model Testing (CPU-Only)

```bash
cd tests/

# Force CPU mode
python quick_cnn_test.py --cpu-only

# What it tests:
# - Model instantiation
# - Forward pass
# - Backward pass (training)
# - Inference mode
```

**Note**: Training on CPU is ~10-100x slower than GPU, but functional.

### FFT Validation (No MATLAB Required)

```bash
cd tests/

python quick_fft_test.py

# What it tests:
# - Parseval's theorem (< 1% error)
# - Spectral centroid computation
# - Spectral rolloff computation
# - Windowing effects (Hann vs rectangular)
```

**Python FFT** (NumPy/SciPy) is equivalent to MATLAB for validation purposes.

---

## Docker Testing (Local)

### Build Containers

```bash
cd docker/

# Build backend
docker build -f Dockerfile.backend -t music-analysis-backend ..

# Build model server (CPU version)
docker build -f Dockerfile.model -t music-analysis-model ..
```

### Run with Docker Compose

```bash
cd docker/
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Test endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8001/health

# Stop
docker-compose down
```

---

## CI/CD with GitHub Actions

GitHub Actions runs automatically on push/PR:

### Workflow: `.github/workflows/test-production.yml`

**What it tests**:
1. **Backend API** - All endpoints, health checks
2. **Model (CPU)** - Architecture, forward/backward pass
3. **Integration** - Component interactions
4. **FFT Validation** - NumPy-based spectral analysis

**What it skips**:
- Full dataset training (too slow)
- GPU-specific tests (no GPU runners)
- Educational notebooks (not production code)

### Trigger CI Manually

```bash
# Push to trigger
git add .
git commit -m "Test changes"
git push origin main

# Or create PR
git checkout -b test-branch
git push origin test-branch
# Create PR on GitHub
```

### View Results

1. Go to GitHub repository
2. Click "Actions" tab
3. View workflow runs
4. Check test results

---

## Educational Components (Not Tested by CI)

### Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks
# - notebooks/01_EDA.ipynb
# - notebooks/02_Modeling.ipynb

# Run all cells manually
# Verify outputs
```

**Why not CI tested?**
- Jupyter notebooks are interactive
- Grading is manual
- Output depends on dataset presence

### Presentation

```bash
cd presentation/

# Edit slides
# presentation.Rmd → presentation.pdf

# Verify PDF generated correctly
open presentation.pdf
```

---

## Troubleshooting

### Tests Fail Locally

```bash
# Update dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+

# Run tests with verbose output
cd tests/
python quick_fft_test.py  # No --ci flag
```

### Docker Fails to Build

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check Docker version
docker --version
```

### Import Errors

```bash
# Install missing packages
pip install torch torchvision
pip install numpy scipy librosa scikit-learn
pip install fastapi uvicorn

# Verify installations
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

### PyTorch GPU Not Available (OK for Local Testing)

```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Expected: False (CPU-only)
# For local testing, this is fine

# To force CPU mode in tests:
python quick_cnn_test.py --cpu-only
```

---

## Performance Expectations

### Local (CPU-Only)

| Test | Duration | Notes |
|------|----------|-------|
| FFT Validation | ~10s | NumPy-based, fast |
| Audio Processing | ~5s | No actual files loaded |
| CNN Test | ~30s | Small test tensors |
| Bayesian Test | ~30s | Simplified search space |
| **Total** | **~90s** | All tests combined |

### GitHub Actions (CPU)

| Job | Duration | Notes |
|-----|----------|-------|
| Backend Tests | ~2-3 min | Install + test |
| Model Tests | ~3-4 min | Install PyTorch + test |
| Integration | ~2-3 min | Combined tests |
| **Total** | **~10 min** | Parallel execution |

### GPU Training (Not Local)

| Task | CPU Time | GPU Time |
|------|----------|----------|
| Single Epoch | ~30-60 min | ~3-5 min |
| Full Training (50 epochs) | ~25-50 hrs | ~2.5-4 hrs |
| Hyperparameter Search | Days | Hours |

---

## Pre-Submission Checklist

Before submitting project:

### Local Tests
- [ ] `bash tests/run_all_tests.sh` passes
- [ ] Backend API starts without errors
- [ ] Docker containers build successfully
- [ ] All notebooks run without errors

### GitHub Actions
- [ ] All CI workflows pass
- [ ] No test failures in Actions tab
- [ ] Docker images build successfully

### Deliverables
- [ ] `notebooks/01_EDA.ipynb` complete
- [ ] `notebooks/02_Modeling.ipynb` complete
- [ ] `presentation/presentation.pdf` generated
- [ ] `presentation/SUMMARY.md` complete

---

## Testing Philosophy

**Local testing validates**:
- Component logic correctness
- API endpoint behavior
- Model architecture validity
- Feature extraction pipeline

**GPU training validates**:
- Full dataset performance
- Convergence behavior
- Hyperparameter effectiveness

**Separation allows**:
- Fast iteration during development
- Catching bugs early (local)
- Comprehensive validation (CI)
- Cost-effective testing (no GPU needed locally)

---

**Next Steps**:
1. Run `bash tests/run_all_tests.sh` locally
2. Fix any failures
3. Push to GitHub
4. Check Actions tab for CI results
5. Deploy to GPU for full training (when ready)

---

**Questions?** See [README.md](README.md) for project overview or [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment.
