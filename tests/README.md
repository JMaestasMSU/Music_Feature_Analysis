# Tests - Multi-Label CNN System

**Real tests for the actual components you built** (not toy smoke tests).

---

## Quick Start

```bash
cd tests/
bash run_all_tests.sh
```

This runs 3 test suites that validate your multi-label CNN system.

---

## Test Suites

### 1. Multi-Label CNN Architecture (`test_multilabel_cnn.py`)

Tests the **MultiLabelAudioCNN** class and trainer.

**What it tests:**
- Architecture scales linearly with genre count
- Multi-label output has correct shape
- Residual connections work
- Channel attention works
- Embedding extraction works
- Uses correct loss function

### 2. Data Augmentation (`test_augmentation.py`)

Tests **SpecAugment, Mixup**, and the data pipeline.

**What it tests:**
- SpecAugment masks correctly
- Mixup blends spectrograms
- Augmentations preserve shape
- Dataset applies augmentation
- DataLoaders work

### 3. Training Pipeline (`test_training.py`)

Tests the **MultiLabelTrainer** and training loop.

**What it tests:**
- Trainer initializes correctly
- Training and validation work
- Full training loop completes
- Predictions return correct format

---

## Runtime

**Total: ~25 seconds** (17 tests across 3 suites)

---

## Troubleshooting

**Missing PyTorch:**
```bash
source ../venv/bin/activate
```

**Missing dependencies:**
```bash
pip install torch numpy scikit-learn
```

---

**You now have real tests for your real code!**
