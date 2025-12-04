"""
Unified prediction service supporting both feature-based and CNN-based models.
Abstracts single-label (GenreClassifier) and multi-label (MultiLabelAudioCNN) inference.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Unified service for genre prediction supporting multiple model types.
    
    Supports:
    - single-label: Feature-based GenreClassifier (8 genres, 20 features)
    - multi-label: Spectrogram-based MultiLabelAudioCNN (24+ genres, 128x128 mel spec)
    """
    
    def __init__(
        self,
        model_type: str = "multi-label",
        model_dir: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize prediction service.
        
        Args:
            model_type: "single-label" or "multi-label"
            model_dir: Path to model directory
            device: Torch device (defaults to cuda if available)
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = None
        self.genre_names = []
        self.config = {}
        
        if self.model_dir:
            self._load_model()
    
    def _load_model(self):
        """Load model based on type."""
        if self.model_type == "single-label":
            self._load_feature_based_model()
        elif self.model_type == "multi-label":
            self._load_cnn_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_feature_based_model(self):
        """Load GenreClassifier (feature-based, single-label)."""
        import sys
        import pickle
        # Add models directory to path
        # self.model_dir is models/ (the models directory itself for feature-based)
        # So it's already the models directory
        models_dir = self.model_dir if self.model_dir.name == "models" else self.model_dir.parent / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))
        from genre_classifier import GenreClassifier
        from model_utils import load_production_model
        
        logger.info(f"Loading feature-based model from {self.model_dir}")
        
        # Check if model files exist
        model_path = self.model_dir / "ml_ready_features.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Feature-based model not found at {model_path}. "
                "Train a GenreClassifier model first or use the multi-label CNN model."
            )
        
        # Load production model
        model, scaler, genre_names, metadata = load_production_model(
            model_class=GenreClassifier,
            model_dir=str(self.model_dir),
            model_name="ml_ready_features",
            device=str(self.device)
        )
        
        self.model = model
        self.scaler = scaler
        self.genre_names = genre_names
        self.config = metadata
        
        logger.info(f"Loaded GenreClassifier: {len(genre_names)} genres")
    
    def _load_cnn_model(self):
        """Load MultiLabelAudioCNN (spectrogram-based, multi-label)."""
        import sys
        import yaml
        # Add models directory to path (from backend/services/ -> project_root/models/)
        # self.model_dir is models/trained_models/multilabel_cnn_filtered_improved
        # So go up 3 levels to project root, then add models
        project_root = self.model_dir.parent.parent.parent
        models_dir = project_root / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))
        from cnn_model import MultiLabelAudioCNN
        
        logger.info(f"Loading CNN model from {self.model_dir}")
        
        # Check required files exist
        config_path = self.model_dir / "config.yaml"
        checkpoint_path = self.model_dir / "best_model.pt"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load genre names (try filtered first, then regular)
        genre_path = self.model_dir / "genre_names_filtered.json"
        if not genre_path.exists():
            genre_path = self.model_dir / "genre_names.json"
        
        if not genre_path.exists():
            raise FileNotFoundError(f"No genre names file found in {self.model_dir}")
        
        with open(genre_path, 'r') as f:
            self.genre_names = json.load(f)
        
        # Create model
        self.model = MultiLabelAudioCNN(
            num_genres=len(self.genre_names),
            base_channels=self.config['base_channels'],
            use_attention=self.config['use_attention']
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded MultiLabelAudioCNN: {len(self.genre_names)} genres, "
                   f"epoch {checkpoint['epoch']}, F1={checkpoint['val_f1']:.3f}")
    
    def predict_from_features(
        self,
        features: np.ndarray,
        top_k: int = 3,
        return_probs: bool = False
    ) -> Dict:
        """
        Predict from hand-crafted features (single-label model only).
        
        Args:
            features: Feature array (20,) or (batch, 20)
            top_k: Number of top predictions
            return_probs: Whether to return all probabilities
            
        Returns:
            Prediction dictionary
        """
        if self.model_type != "single-label":
            raise ValueError("predict_from_features only supported for single-label models")
        
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1)
        
        probs_np = probs.cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probs_np)[-top_k:][::-1]
        top_predictions = [
            {
                'genre': self.genre_names[idx],
                'confidence': float(probs_np[idx])
            }
            for idx in top_indices
        ]
        
        result = {
            'predicted_genre': self.genre_names[top_indices[0]],
            'confidence': float(probs_np[top_indices[0]]),
            'top_predictions': top_predictions
        }
        
        if return_probs:
            result['all_probabilities'] = {
                genre: float(prob)
                for genre, prob in zip(self.genre_names, probs_np)
            }
        
        return result
    
    def predict_from_spectrogram(
        self,
        spectrogram: np.ndarray,
        threshold: float = 0.5,
        top_k: int = 5
    ) -> Dict:
        """
        Predict from mel spectrogram.
        
        Args:
            spectrogram: Mel spectrogram [128, 128]
            threshold: Confidence threshold for multi-label
            top_k: Number of top predictions
            
        Returns:
            Prediction dictionary
        """
        if self.model_type == "single-label":
            raise ValueError("predict_from_spectrogram not supported for single-label models. Use predict_from_features instead.")
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        # Validate spectrogram shape
        if spectrogram.shape != (128, 128):
            raise ValueError(f"Expected spectrogram shape (128, 128), got {spectrogram.shape}")
        
        try:
            # Convert to tensor [1, 1, 128, 128]
            with torch.no_grad():
                spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
                outputs = self.model(spec_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error in CNN forward pass: {e}")
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        # Get top predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                'genre': self.genre_names[idx],
                'confidence': float(probs[idx])
            }
            for idx in top_indices
        ]
        
        result = {
            'predicted_genre': self.genre_names[top_indices[0]],
            'confidence': float(probs[top_indices[0]]),
            'top_predictions': top_predictions
        }
        
        return result
    
    def predict_multi_label(
        self,
        spectrogram: np.ndarray,
        threshold: float = 0.3
    ) -> List[Dict[str, float]]:
        """
        Get all genres above threshold (multi-label only).
        
        Args:
            spectrogram: Mel spectrogram [128, 128]
            threshold: Minimum confidence threshold
            
        Returns:
            List of {genre, confidence} for predictions above threshold
        """
        if self.model_type != "multi-label":
            raise ValueError("predict_multi_label only supported for multi-label models")
        
        with torch.no_grad():
            spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
            outputs = self.model(spec_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Filter by threshold
        above_threshold = [
            {
                'genre': self.genre_names[i],
                'confidence': float(probs[i])
            }
            for i in range(len(probs))
            if probs[i] >= threshold
        ]
        
        # Sort by confidence
        above_threshold.sort(key=lambda x: x['confidence'], reverse=True)
        
        return above_threshold
    
    def get_all_probabilities(
        self,
        input_data: np.ndarray,
        input_type: str = "spectrogram"
    ) -> Dict[str, float]:
        """
        Get probabilities for all genres.
        
        Args:
            input_data: Either features (20,) or spectrogram (128, 128)
            input_type: "features" or "spectrogram"
            
        Returns:
            Dict mapping genre name to probability
        """
        if input_type == "features":
            if self.model_type != "single-label":
                raise ValueError("Features input only supported for single-label models")
            
            # Handle single sample
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Scale and predict
            if self.scaler is not None:
                input_data = self.scaler.transform(input_data)
            
            features_tensor = torch.FloatTensor(input_data).to(self.device)
            
            with torch.no_grad():
                logits = self.model(features_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        elif input_type == "spectrogram":
            if self.model_type != "multi-label":
                raise ValueError("Spectrogram input only supported for multi-label models")
            
            with torch.no_grad():
                spec_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0).to(self.device)
                outputs = self.model(spec_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        return {
            self.genre_names[i]: float(probs[i])
            for i in range(len(probs))
        }
    
    @property
    def num_genres(self) -> int:
        """Get number of genres."""
        return len(self.genre_names)
    
    @property
    def model_info(self) -> Dict:
        """Get model information."""
        info = {
            'model_type': self.model_type,
            'num_genres': self.num_genres,
            'genres': self.genre_names,
            'device': str(self.device)
        }
        
        if self.model_type == "single-label":
            info.update({
                'input_type': 'features',
                'input_dim': 20,
                'architecture': 'GenreClassifier',
                'task': 'single-label classification'
            })
            if self.config:
                info['metadata'] = self.config
        
        elif self.model_type == "multi-label":
            info.update({
                'input_type': 'spectrogram',
                'input_shape': [128, 128],
                'architecture': 'MultiLabelAudioCNN',
                'task': 'multi-label classification',
                'experiment_name': self.config.get('experiment_name', 'unknown'),
                'base_channels': self.config.get('base_channels', 'unknown'),
                'use_attention': self.config.get('use_attention', 'unknown')
            })
        
        return info
