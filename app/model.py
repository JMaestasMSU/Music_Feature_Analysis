"""
Model loading and inference for FastAPI application.
Handles model initialization, caching, and prediction.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.genre_classifier import GenreClassifier, ModelWrapper
from models.model_utils import load_production_model

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for managing model lifecycle and predictions.
    Implements singleton pattern for model caching.
    """
    
    _instance = None
    _model_wrapper: Optional[ModelWrapper] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model service."""
        if self._model_wrapper is None:
            self.load_model()
    
    def load_model(
        self,
        model_dir: str = 'trained_models',
        model_name: str = 'genre_classifier_production',
        device: Optional[str] = None
    ) -> None:
        """
        Load production model from disk.
        
        Args:
            model_dir: Directory containing model files
            model_name: Base name of model files
            device: Device to load model onto ('cpu' or 'cuda')
        """
        try:
            # Determine device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Loading model from {model_dir}/{model_name} on {device}")
            
            # Load model components
            model, scaler, genre_names, metadata = load_production_model(
                model_class=GenreClassifier,
                model_dir=model_dir,
                model_name=model_name,
                device=device
            )
            
            # Create model wrapper
            self._model_wrapper = ModelWrapper(
                model=model,
                scaler=scaler,
                genre_names=genre_names,
                device=device
            )
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model metrics: {metadata.get('metrics', {})}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fall back to untrained model for development
            logger.warning("Loading untrained model for development")
            self._load_fallback_model(device or 'cpu')
    
    def _load_fallback_model(self, device: str = 'cpu') -> None:
        """Load untrained model as fallback for development."""
        model = GenreClassifier()
        genre_names = [
            'Rock', 'Electronic', 'Hip-Hop', 'Classical',
            'Jazz', 'Folk', 'Pop', 'Experimental'
        ]
        
        self._model_wrapper = ModelWrapper(
            model=model,
            scaler=None,
            genre_names=genre_names,
            device=device
        )
        
        logger.warning("Using untrained model - predictions will be random!")
    
    def predict(
        self,
        features: np.ndarray,
        return_probs: bool = False,
        top_k: int = 3
    ) -> Dict:
        """
        Predict genre for given features.
        
        Args:
            features: Input features (n_features,) or (batch_size, n_features)
            return_probs: Whether to return all class probabilities
            top_k: Number of top predictions to return
        
        Returns:
            Prediction results dictionary
        """
        if self._model_wrapper is None:
            raise RuntimeError("Model not loaded")
        
        return self._model_wrapper.predict(
            features=features,
            return_probs=return_probs,
            top_k=top_k
        )
    
    def predict_batch(
        self,
        features_list: List[np.ndarray],
        return_probs: bool = False
    ) -> List[Dict]:
        """
        Predict genres for batch of features.
        
        Args:
            features_list: List of feature arrays
            return_probs: Whether to return all class probabilities
        
        Returns:
            List of prediction dictionaries
        """
        if self._model_wrapper is None:
            raise RuntimeError("Model not loaded")
        
        return self._model_wrapper.predict_batch(
            features_list=features_list,
            return_probs=return_probs
        )
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model."""
        if self._model_wrapper is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'device': self._model_wrapper.device,
            'num_classes': len(self._model_wrapper.genre_names),
            'genre_names': self._model_wrapper.genre_names,
            'input_features': self._model_wrapper.model.input_dim if hasattr(self._model_wrapper.model, 'input_dim') else 20,
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_wrapper is not None


# Global model service instance
model_service = ModelService()


def get_model_service() -> ModelService:
    """Get global model service instance."""
    return model_service


if __name__ == '__main__':
    # Test model service
    service = get_model_service()
    
    # Get model info
    info = service.get_model_info()
    print(f"Model info: {info}")
    
    # Test prediction
    test_features = np.random.randn(20)
    result = service.predict(test_features, return_probs=True, top_k=3)
    
    print(f"\nPrediction result:")
    print(f"  Predicted genre: {result['predicted_genre']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"\n  Top 3 predictions:")
    for pred in result['top_predictions']:
        print(f"    {pred['genre']}: {pred['confidence']:.4f}")
