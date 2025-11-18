# Quick Tests - Music Feature Analysis

Fast validation tests that prove core components work (<2 minutes total).

## Running Tests

**Run all tests:**
```bash
cd tests/
bash run_all_tests.sh
```

**Run individual tests:**
```bash
python quick_fft_test.py              # FFT validation (~10 sec)
python quick_audio_processing_test.py # Audio processing (~5 sec)
python quick_cnn_test.py --cpu-only   # Neural network (~30 sec)
python quick_bayesian_test.py         # Optimization (~30 sec)
```

## Test Descriptions

### 1. FFT Validation (`quick_fft_test.py`)

**Purpose**: Validates FFT-based feature extraction

**Tests**:
- Parseval's theorem (time/freq energy equivalence)
- Spectral centroid computation
- Spectral rolloff calculation
- Windowing effects

**Pass criteria**: All assertions pass, error < 1%

---

### 2. Audio Processing (`quick_audio_processing_test.py`)

**Purpose**: Tests librosa-based feature extraction

**Tests**:
- Generate synthetic audio
- Extract MFCCs
- Compute spectral features
- Calculate temporal features

**Pass criteria**: Features extracted successfully

---

### 3. CNN Architecture (`quick_cnn_test.py`)

**Purpose**: Validates neural network can train

**Tests**:
- Model instantiation
- Forward pass
- Backward pass (training step)
- Inference mode

**Pass criteria**: All operations complete without errors

---

### 4. Bayesian Optimization (`quick_bayesian_test.py`)

**Purpose**: Tests hyperparameter optimization

**Tests**:
- Optimize synthetic objective function
- Convergence check

**Pass criteria**: Finds near-optimal solution

---

## CI Mode

For automated testing (less verbose):

```bash
python quick_fft_test.py --ci
python quick_cnn_test.py --ci --cpu-only
```

---

## Expected Runtime

| Test | Runtime | Purpose |
|------|---------|---------|
| FFT | ~10 sec | Numerical validation |
| Audio | ~5 sec | Feature extraction |
| CNN | ~30 sec | Neural network |
| Bayesian | ~30 sec | Optimization |
| **Total** | **< 2 min** | Complete validation |

---

## Troubleshooting

**ModuleNotFoundError**: Install dependencies
```bash
pip install -r ../requirements.txt
```

**CUDA errors**: Force CPU mode
```bash
python quick_cnn_test.py --cpu-only
```

**Slow tests**: Check CPU usage, close other apps

---

**Last Updated**: 2024
