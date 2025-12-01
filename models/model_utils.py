"""
Utilities for saving, loading, and managing models.
"""

import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
import json


def save_production_model(
    model: torch.nn.Module,
    scaler: StandardScaler,
    genre_names: List[str],
    metrics: Dict[str, float],
    model_dir: str = 'trained_models',
    model_name: str = 'genre_classifier_production'
) -> None:
    """
    Save complete model package for production deployment.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        genre_names: List of genre names
        metrics: Dictionary of performance metrics
        model_dir: Directory to save model
        model_name: Base name for model files
    """
    # Create directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_file = model_path / f'{model_name}.pt'
    torch.save(model.state_dict(), model_file)
    print(f"✓ Saved model weights: {model_file}")
    
    # Save scaler
    scaler_file = model_path / f'{model_name}_scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_file}")
    
    # Save metadata
    metadata = {
        'genre_names': genre_names,
        'metrics': metrics,
        'input_dim': model.input_dim,
        'num_classes': model.num_classes,
        'model_architecture': str(model)
    }
    
    metadata_file = model_path / f'{model_name}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_file}")
    
    print(f"\n✓ Model package saved to: {model_path}/")


def load_production_model(
    model_class: type,
    model_dir: str = 'trained_models',
    model_name: str = 'genre_classifier_production',
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, StandardScaler, List[str], Dict]:
    """
    Load complete model package from disk.
    
    Args:
        model_class: Model class (e.g., GenreClassifier)
        model_dir: Directory containing model files
        model_name: Base name of model files
        device: Device to load model onto
    
    Returns:
        Tuple of (model, scaler, genre_names, metadata)
    """
    model_path = Path(model_dir)
    
    # Load metadata
    metadata_file = model_path / f'{model_name}_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    genre_names = metadata['genre_names']
    input_dim = metadata['input_dim']
    num_classes = metadata['num_classes']
    
    # Instantiate model
    model = model_class(input_dim=input_dim, num_classes=num_classes)
    
    # Load weights
    model_file = model_path / f'{model_name}.pt'
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    
    # Load scaler
    scaler_file = model_path / f'{model_name}_scaler.pkl'
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Model loaded from: {model_path}/")
    print(f"  Metrics: {metadata['metrics']}")
    
    return model, scaler, genre_names, metadata


if __name__ == '__main__':
    # Test save/load
    from genre_classifier import GenreClassifier
    
    print("Testing model save/load...")
    
    # Create dummy model
    model = GenreClassifier(input_dim=20, num_classes=8)
    scaler = StandardScaler()
    scaler.fit([[0]*20])  # Dummy fit
    
    genre_names = ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic', 'Hip-Hop', 'Folk', 'Experimental']
    metrics = {'accuracy': 0.78, 'f1_score': 0.77}
    
    # Save
    save_production_model(
        model=model,
        scaler=scaler,
        genre_names=genre_names,
        metrics=metrics,
        model_dir='test_models',
        model_name='test_model'
    )
    
    # Load
    loaded_model, loaded_scaler, loaded_genres, loaded_metadata = load_production_model(
        model_class=GenreClassifier,
        model_dir='test_models',
        model_name='test_model'
    )
    
    print(f"\n✓ Load successful!")
    print(f"  Genres: {loaded_genres}")
    print(f"  Metrics: {loaded_metadata['metrics']}")
