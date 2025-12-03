"""
Model utility functions for saving and loading trained models.
"""

import torch
import pickle
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Type
from sklearn.preprocessing import StandardScaler


def save_production_model(
    model,
    scaler: Optional[StandardScaler],
    genre_names: list,
    metadata: Dict[str, Any],
    model_dir: str,
    model_name: str
) -> None:
    """
    Save production model with all necessary components.

    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler (or None)
        genre_names: List of genre names
        metadata: Dictionary with training metrics and info
        model_dir: Directory to save model files
        model_name: Base name for model files
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = model_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    # Save scaler
    if scaler is not None:
        scaler_path = model_dir / f"{model_name}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to {scaler_path}")

    # Save genre names
    genre_path = model_dir / f"{model_name}_genres.json"
    with open(genre_path, 'w') as f:
        json.dump(genre_names, f, indent=2)
    print(f"Saved genre names to {genre_path}")

    # Save metadata
    metadata_path = model_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def load_production_model(
    model_class: Type,
    model_dir: str,
    model_name: str,
    device: str = 'cpu'
) -> Tuple[Any, Optional[StandardScaler], list, Dict[str, Any]]:
    """
    Load production model with all components.

    Args:
        model_class: Model class to instantiate
        model_dir: Directory containing model files
        model_name: Base name of model files
        device: Device to load model onto ('cpu' or 'cuda')

    Returns:
        Tuple of (model, scaler, genre_names, metadata)
    """
    model_dir = Path(model_dir)

    # Load metadata to get model config
    metadata_path = model_dir / f"{model_name}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load genre names
    genre_path = model_dir / f"{model_name}_genres.json"
    if not genre_path.exists():
        raise FileNotFoundError(f"Genre names file not found: {genre_path}")

    with open(genre_path, 'r') as f:
        genre_names = json.load(f)

    # Create model instance
    model_config = metadata.get('model_config', {})
    input_dim = model_config.get('input_dim', 20)
    num_classes = len(genre_names)

    model = model_class(input_dim=input_dim, num_classes=num_classes)

    # Load model weights
    model_path = model_dir / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load scaler if available
    scaler = None
    scaler_path = model_dir / f"{model_name}_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    return model, scaler, genre_names, metadata
