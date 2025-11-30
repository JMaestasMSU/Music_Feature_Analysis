"""
Model utilities for saving, loading, and managing trained models.
"""

import torch
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """
    Save and load model checkpoints with metadata.
    """
    
    def __init__(self, checkpoint_dir: str = 'trained_models'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        metadata: Optional[Dict] = None,
        filename: str = 'checkpoint.pt'
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state (optional)
            scaler: Feature scaler (optional)
            metadata: Additional metadata (metrics, config, etc.)
            filename: Checkpoint filename
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config() if hasattr(model, 'get_config') else {},
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add metadata if provided
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # Save model checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save scaler separately (pickle format)
        if scaler is not None:
            scaler_path = checkpoint_path.with_suffix('.scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    def load(
        self,
        model: torch.nn.Module,
        filename: str = 'checkpoint.pt',
        load_optimizer: bool = False,
        load_scaler: bool = False,
        device: str = 'cpu'
    ) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Dict]:
        """
        Load model checkpoint.
        
        Args:
            model: Model instance to load weights into
            filename: Checkpoint filename
            load_optimizer: Whether to load optimizer state
            load_scaler: Whether to load feature scaler
            device: Device to load model onto
        
        Returns:
            Tuple of (model, optimizer, scaler, metadata)
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        logger.info(f"Model weights loaded from {checkpoint_path}")
        
        # Load optimizer if requested
        optimizer = None
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            # Note: optimizer must be created externally with same parameters
            logger.warning("Optimizer state found but optimizer instance must be created externally")
        
        # Load scaler if requested
        scaler = None
        if load_scaler:
            scaler_path = checkpoint_path.with_suffix('.scaler.pkl')
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
        
        # Extract metadata
        metadata = checkpoint.get('metadata', {})
        
        return model, optimizer, scaler, metadata
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob('*.pt'))
        return [cp.name for cp in checkpoints]


def save_model_for_production(
    model: torch.nn.Module,
    scaler: Any,
    genre_names: list,
    metrics: Dict,
    save_dir: str = 'trained_models',
    model_name: str = 'genre_classifier_production'
) -> Dict[str, str]:
    """
    Save complete model package for production deployment.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        genre_names: List of genre labels
        metrics: Model performance metrics
        save_dir: Directory to save model package
        model_name: Base name for model files
    
    Returns:
        Dictionary of saved file paths
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save model weights
    model_path = save_path / f'{model_name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config() if hasattr(model, 'get_config') else {},
    }, model_path)
    paths['model'] = str(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = save_path / f'{model_name}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    paths['scaler'] = str(scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'genre_names': genre_names,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_architecture': str(model),
    }
    
    metadata_path = save_path / f'{model_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    paths['metadata'] = str(metadata_path)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return paths


def load_production_model(
    model_class,
    model_dir: str = 'trained_models',
    model_name: str = 'genre_classifier_production',
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, Any, list, Dict]:
    """
    Load complete model package for production use.
    
    Args:
        model_class: Model class to instantiate
        model_dir: Directory containing model files
        model_name: Base name of model files
        device: Device to load model onto
    
    Returns:
        Tuple of (model, scaler, genre_names, metadata)
    """
    model_path = Path(model_dir)
    
    # Load metadata first
    metadata_path = model_path / f'{model_name}_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    genre_names = metadata['genre_names']
    
    # Load model
    checkpoint_path = model_path / f'{model_name}.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model from config if available
    if 'model_config' in checkpoint and checkpoint['model_config']:
        model = model_class.from_config(checkpoint['model_config'])
    else:
        model = model_class()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load scaler
    scaler_path = model_path / f'{model_name}_scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info(f"Production model loaded from {model_dir}")
    
    return model, scaler, genre_names, metadata


def get_model_info(
    model: torch.nn.Module,
    input_shape: Tuple = (1, 20)
) -> Dict:
    """
    Get detailed information about a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape for testing
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape)
        output = model(dummy_input)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        'input_shape': input_shape,
        'output_shape': tuple(output.shape),
        'architecture': str(model),
    }
    
    return info


if __name__ == '__main__':
    # Example usage
    from genre_classifier import GenreClassifier
    
    # Create and save model
    model = GenreClassifier()
    checkpoint_manager = ModelCheckpoint()
    
    metadata = {
        'accuracy': 0.82,
        'f1_score': 0.81,
        'epoch': 45
    }
    
    checkpoint_manager.save(
        model=model,
        metadata=metadata,
        filename='test_checkpoint.pt'
    )
    
    # Load model
    loaded_model, _, _, loaded_metadata = checkpoint_manager.load(
        model=GenreClassifier(),
        filename='test_checkpoint.pt'
    )
    
    print("Model loaded successfully!")
    print(f"Metadata: {loaded_metadata}")
    
    # Get model info
    info = get_model_info(loaded_model)
    print(f"\nModel Info:")
    for key, value in info.items():
        if key != 'architecture':
            print(f"  {key}: {value}")
