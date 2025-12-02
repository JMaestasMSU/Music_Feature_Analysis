# Data Processing Scripts

**Single source of truth**: `process_audio_files.py`

---

## Quick Start

```bash
# 1. Download FMA metadata (one time)
cd data/metadata/
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
cd ../../

# 2. Create features (one time)
python scripts/process_audio_files.py

# Done! File created: data/processed/ml_ready_features.pkl
```

---

## What It Does

1. Loads `data/metadata/features.csv` (pre-computed audio features)
2. Loads `data/metadata/tracks.csv` (genre labels)
3. Selects key feature groups (MFCC, spectral, temporal)
4. Merges features with genres
5. Cleans data (removes missing values)
6. Saves to `data/processed/ml_ready_features.pkl`

**Output**: ~8,000 tracks × ~500 features + genre labels

---

## This File Powers Everything

- `notebooks/01_EDA.ipynb` - Loads for exploratory analysis
- `notebooks/02_Modeling.ipynb` - Loads for model training
- `backend/app.py` - Loads for API predictions
- `tests/*.py` - Loads for validation

**No other data processing needed!**

---

## Troubleshooting

**"Features file not found"**
```bash
cd data/metadata/
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
```

**"Permission denied"**
```bash
chmod +x scripts/process_audio_files.py
```
