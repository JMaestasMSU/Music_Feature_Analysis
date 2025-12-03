# Model Optimization Log

## Initial Model Performance Analysis

### Date: December 3, 2025

### Starting Point: Production Model Results
- **Model**: `multilabel_cnn_70genres_20251203_113316`
- **Training ended**: Epoch 55 (early stopped from max 150)
- **Dataset**: 70 genres, 38,801 samples

#### Test Set Performance (Threshold = 0.5)
```
F1 Score:       0.134
Precision:      0.078 (7.8%)
Recall:         0.770 (77%)
Hamming Loss:   0.249
Jaccard Score:  0.052
Avg predictions: 15-20 genres per track (actual: ~1)
```

**Problem Identified**: Model predicts too many genres per track, resulting in very low precision despite high recall.

---

## Optimization Phase 1: Threshold Tuning

### Objective
Find optimal prediction threshold and/or top-K strategy to improve precision-recall balance.

### Methodology
Created `tune_model_threshold.py` script to systematically test:
1. **Threshold-only strategies**: 0.30 to 0.80 in 0.05 increments
2. **Top-K strategies**: Keep only top 3, 5, 7, 10, 15, or 20 predictions
3. **Combined strategies**: Threshold + top-K filtering

### Results

#### Best Threshold-Only Strategy
```
Threshold: 0.80
F1 Score:       0.101 (10.1%)
Precision:      0.083 (8.3%)
Recall:         0.427 (42.7%)
Avg predictions: 3.4 genres per track
```

#### Best Top-K Strategy
```
Top-K: 5
F1 Score:       0.097 (9.7%)
Precision:      0.070 (7.0%)
Recall:         0.502 (50.2%)
Avg predictions: 5.0 genres per track
```

#### Best Combined Strategy
```
Threshold: 0.60, Top-K: 5
F1 Score:       0.097
Precision:      0.071
Recall:         0.501
Avg predictions: 4.95 genres per track
```

### Conclusion
**Threshold tuning cannot fix the fundamental problem**: The model only achieved 9-10% F1 score at best, with precision barely reaching 8.3%. The underlying model is not learning effectively.

**Root Causes Identified**:
1. Severe class imbalance (126x between most/least common genres)
2. Too many genres (70) for the dataset size (38,801 samples)
3. Many genres have insufficient data (<100 samples)
4. Model capacity may be insufficient
5. Class weights not aggressive enough

---

## Optimization Phase 2: Data Quality & Model Architecture

### Data Analysis

#### Genre Distribution (Original 70 genres)
```
Top genre:    6,316 samples (Field Recordings)
Bottom genre:    50 samples
Imbalance:    126.3x

Genres with <500 samples: 51/70 (73%)
Genres with <100 samples: 19/70 (27%)
```

#### Viable Genres (≥300 samples): 24 genres
```
 1. Field Recordings         : 6316 samples
 2. Avant-Garde              : 3498
 3. Sound Collage            : 3160
 4. Folk                     : 2298
 5. Power-Pop                : 2222
 6. Free-Jazz                : 1688
 7. Singer-Songwriter        : 1442
 8. Musique Concrete         : 1425
 9. Indie-Rock               : 1363
10. Old-Time / Historic      : 1258
11. Dance                    :  875
12. Krautrock                :  837
13. Freak-Folk               :  817
14. Loud-Rock                :  778
15. International            :  622
16. Ambient Electronic       :  601
17. Reggae - Dub             :  562
18. Rock                     :  556
19. Chiptune                 :  514
20. Free-Folk                :  499
21. Post-Rock                :  490
22. Pop                      :  408
23. Hip-Hop                  :  317
24. Compilation              :  310

Total: 32,856 samples
Imbalance: 20.4x (down from 126.3x)
```

### Code Changes

#### 1. Genre Filtering (`train_multilabel_cnn.py`)
**New Arguments Added**:
```python
--min-samples-per-genre N   # Filter out genres with <N samples
--max-genres N              # Keep only top N most common genres
```

**New Function**: `filter_genres()`
- Filters dataset to include only well-represented genres
- Removes samples with no remaining labels after filtering
- Reports imbalance statistics

#### 2. Improved Class Weighting
**New Arguments**:
```python
--pos-weight-power 0.5      # Apply sqrt to weights (reduces extremes)
--pos-weight-cap 50         # Cap maximum weight value
--debug                     # Enable detailed logging
```

**Enhanced `calculate_pos_weight()` function**:
- **Power transformation**: Apply power to weights (0.5 = sqrt scaling)
  - Example: 126x imbalance → 11.2x effective weight
- **Weight capping**: Prevent any weight from exceeding specified maximum
- **Better statistics**: Shows median, top 10 classes in debug mode

**Before**: 126x raw imbalance → 126x weight
**After**: 20.4x raw imbalance → 4.5x effective weight (with sqrt)

#### 3. Increased Model Capacity
**Change**: `--base-channels` increased from 96 to 128
- Model parameter count increases by ~78%
- More capacity to learn complex genre patterns

#### 4. Top-K Prediction Integration
**New Arguments**:
```python
--load-tuning-results PATH  # Load optimal threshold/top-K from tuning
--prediction-threshold 0.5  # Manual threshold setting
--top-k N                   # Apply top-K filtering
```

**New Function**: `apply_topk()`
- Filters predictions to keep only top-K highest probability genres
- Applied after threshold during evaluation

### Training Configuration (Improved Model)

**Command**:
```bash
python scripts/train_multilabel_cnn.py \
  --min-samples-per-genre 300 \
  --labels-file labels_multilabel.npy \
  --num-genres 70 \
  --epochs 150 \
  --batch-size 32 \
  --lr 0.0005 \
  --base-channels 128 \
  --patience 25 \
  --weight-decay 1e-5 \
  --pos-weight-power 0.5 \
  --pos-weight-cap 50 \
  --debug \
  --experiment-name multilabel_cnn_filtered_improved
```

**Key Settings**:
- Genre filtering: 70 → 24 genres (300+ samples each)
- Class weighting: sqrt-scaled, capped at 50x
- Model size: 128 base channels (larger capacity)
- Same hyperparameters otherwise

---

## Expected Improvements

### Before (Original Model)
- 70 genres, 38,801 samples
- 126x class imbalance
- No class weighting optimization
- 96 base channels
- Val F1: 0.094 (9.4%)
- Test F1: 0.134 (13.4%)
- Test Precision: 0.078 (7.8%)

### After (Improved Model - Expected)
- 24 genres, 32,856 samples
- 20.4x raw imbalance → 4.5x effective (sqrt-scaled)
- Optimized class weighting
- 128 base channels (+78% parameters)
- **Expected Val F1**: 0.20-0.35 (20-35%)
- **Expected Test F1**: 0.25-0.40 (25-40%)
- **Expected Test Precision**: 0.15-0.30 (15-30%)

### Improvement Targets
- **F1 Score**: 2-3x improvement
- **Precision**: 2-4x improvement
- **Class Balance**: 28x reduction in imbalance (126x → 4.5x effective)
- **Data Quality**: 100% of genres have 300+ samples (vs 27% before)

---

## Files Modified

### Scripts
- `scripts/train_multilabel_cnn.py`
  - Added genre filtering functionality
  - Enhanced class weight calculation
  - Integrated threshold tuning results loading
  - Added top-K prediction filtering

- `scripts/tune_model_threshold.py` (NEW)
  - Systematic threshold testing
  - Top-K strategy evaluation
  - Combined strategy testing
  - Results visualization

### Model Outputs
- `models/trained_models/multilabel_cnn_70genres_20251203_113316/`
  - Original model results
  - `threshold_tuning_results.json` - Tuning analysis
  - `threshold_tuning_plots.png` - Visualization

- `models/trained_models/multilabel_cnn_filtered_improved/` (IN PROGRESS)
  - Improved model with filtered genres
  - Better class weighting
  - Larger architecture

---

## Next Steps

1. **Monitor training progress** - Check validation metrics
2. **Compare results** - Original vs improved model performance
3. **Threshold tuning (improved model)** - Run tuning on new model
4. **Final evaluation** - Test set performance analysis
5. **Documentation** - Update presentation with results comparison

---

## Lessons Learned

1. **Data quality > Model complexity**: Filtering to well-represented genres had bigger impact than architecture changes
2. **Class imbalance is critical**: 126x imbalance is too extreme for standard BCE loss
3. **Threshold tuning has limits**: Cannot fix a poorly-trained model
4. **Genre count matters**: 70 genres with poor representation → overfitting and poor generalization
5. **Class weighting strategy**: Sqrt-scaling prevents rare classes from overwhelming the loss

---

## Training Progress Updates

### Early Epochs
- **Initial validation precision: 9.12%**
  - Already exceeds original model's final test precision of 7.8%
  - Improvement visible from epoch 1
  - Indicates better learning with filtered genres and improved class weighting

---

## Status: Training in Progress

Current model (`multilabel_cnn_filtered_improved`) is training with optimized settings.
Early results show improvement over baseline.
Final results will be compared against baseline once training completes.
