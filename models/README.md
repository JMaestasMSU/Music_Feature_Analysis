# Models Directory

This directory contains machine learning models for music genre classification.

---

## Directory Structure

```
models/
├── genre_classifier.py       # Neural network model definition
├── model_utils.py             # Model save/load utilities
├── trained_models/            # Saved model weights
│   ├── genre_classifier_production.pt
│   ├── genre_classifier_production_scaler.pkl
│   └── genre_classifier_production_metadata.json
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Model Architecture

### GenreClassifier

Fully connected neural network for genre classification from audio features.

**Architecture:**
```
Input (20 features)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.3)
    ↓
Dense(32) + ReLU
    ↓
Dense(8) → Softmax
```

**Parameters:**
- Total parameters: ~18,000
- Trainable parameters: ~18,000
- Model size: ~70 KB

**Performance:**
- Test Accuracy: 78-82%
- Weighted F1-Score: 0.78-0.82
- Inference time: < 10ms per sample (CPU)

---

## Usage

### Training a Model

```python
from genre_classifier import GenreClassifier
import torch.optim as optim

# Create model
model = GenreClassifier(input_dim=20, num_classes=8)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train (see notebooks/02_Modeling.ipynb for full example)
```

### Saving a Trained Model

```python
from model_utils import save_model_for_production

# Save complete model package
paths = save_model_for_production(
    model=trained_model,
    scaler=feature_scaler,
    genre_names=genre_list,
    metrics={'accuracy': 0.82, 'f1': 0.81},
    save_dir='trained_models',
    model_name='genre_classifier_production'
)

print(f"Model saved to: {paths['model']}")
```

### Loading for Inference

```python
from genre_classifier import GenreClassifier, ModelWrapper
from model_utils import load_production_model

# Load model
model, scaler, genres, metadata = load_production_model(
    model_class=GenreClassifier,
    model_dir='trained_models',
    model_name='genre_classifier_production'
)

# Create wrapper for easy inference
wrapper = ModelWrapper(model, scaler, genres)

# Predict
import numpy as np
features = np.random.randn(20)  # Your audio features
result = wrapper.predict(features, top_k=3)

print(f"Predicted: {result['predicted_genre']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## Model Checkpointing

Use `ModelCheckpoint` for training with checkpoints:

```python
from model_utils import ModelCheckpoint

checkpoint_manager = ModelCheckpoint('trained_models')

# Save during training
checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    metadata={'epoch': epoch, 'loss': loss, 'accuracy': acc},
    filename=f'checkpoint_epoch_{epoch}.pt'
)

# Load best model
model, _, _, metadata = checkpoint_manager.load(
    model=GenreClassifier(),
    filename='checkpoint_best.pt'
)
```

---

## Model Utilities

### ModelWrapper

Convenience class for preprocessing and postprocessing:

```python
wrapper = ModelWrapper(model, scaler, genre_names)

# Single prediction
result = wrapper.predict(features, return_probs=True, top_k=3)

# Batch prediction
results = wrapper.predict_batch([features1, features2, features3])
```

### get_model_info()

Get detailed model information:

```python
from model_utils import get_model_info

info = get_model_info(model, input_shape=(1, 20))
print(f"Total parameters: {info['total_parameters']:,}")
print(f"Model size: {info['model_size_mb']:.2f} MB")
```

---

## Model Files

### Model Weights (.pt)

PyTorch state dict containing trained model weights.

**Contents:**
- `model_state_dict`: Model weights
- `model_config`: Model configuration (architecture params)

### Feature Scaler (.pkl)

Scikit-learn StandardScaler for feature normalization.

**Usage:**
```python
import pickle
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

normalized_features = scaler.transform(raw_features)
```

### Metadata (.json)

Model information and performance metrics.

**Contents:**
- `model_name`: Model identifier
- `genre_names`: List of genre labels
- `metrics`: Performance metrics (accuracy, F1, etc.)
- `timestamp`: When model was saved
- `model_architecture`: Model description

---

## Testing

Test model functionality:

```bash
# Test model creation and forward pass
python genre_classifier.py

# Test model utilities
python model_utils.py
```

---

## Model Versioning

When saving new models, use versioned filenames:

```python
save_model_for_production(
    model=model,
    scaler=scaler,
    genre_names=genres,
    metrics=metrics,
    model_name=f'genre_classifier_v{version}_{timestamp}'
)
```

---

## Integration with API

The `app/model.py` uses these models:

```python
from app.model import model_service

# Model is loaded on startup
result = model_service.predict(features)
```

---

## Production Checklist

Before deploying a model:

- [ ] Model trained on full dataset
- [ ] Cross-validation performed
- [ ] Test accuracy > 75%
- [ ] Scaler saved with model
- [ ] Metadata includes metrics
- [ ] Model tested with `model_utils.py`
- [ ] Integration tested with API
- [ ] Inference time < 50ms per sample

---

**See also:**
- [notebooks/02_Modeling.ipynb](../notebooks/02_Modeling.ipynb) - Model training
- [app/model.py](../app/model.py) - Model service integration
- [tests/quick_cnn_test.py](../tests/quick_cnn_test.py) - Model tests
