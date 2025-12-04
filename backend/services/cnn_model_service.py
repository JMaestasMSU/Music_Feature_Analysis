"""
CNN Model Service - Handles model loading and inference for genre classification.
"""

import torch
import numpy as np
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import sys

logger = logging.getLogger(__name__)

class CNNModelService:
    """Service for CNN-based genre prediction."""

    def __init__(self, model_dir: Path):
        """
        Initialize CNN model service.

        Args:
            model_dir: Path to model directory containing config.yaml, genre_names.json, and best_model.pt
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.config = None
        self.genre_names = None
        self.device = None

        self._load_model()

    def _load_model(self):
        """Load model, config, and genre names."""
        logger.info(f"Loading CNN model from {self.model_dir}")

        # Load config
        config_path = self.model_dir / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load genre names
        genre_path = self.model_dir / "genre_names.json"
        with open(genre_path, 'r') as f:
            self.genre_names = json.load(f)

        logger.info(f"Model: {self.config['experiment_name']}, Genres: {len(self.genre_names)}")

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Import model class
        project_root = self.model_dir.parent.parent.parent
        sys.path.append(str(project_root / "models"))
        from cnn_model import MultiLabelAudioCNN

        # Create model
        self.model = MultiLabelAudioCNN(
            num_genres=len(self.genre_names),
            base_channels=self.config['base_channels'],
            use_attention=self.config['use_attention']
        )

        # Load checkpoint
        checkpoint_path = self.model_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(self.device)

        logger.info(f"Model loaded: epoch {checkpoint['epoch']}, val_F1={checkpoint['val_f1']:.3f}")

    def predict(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Predict genre probabilities from spectrogram.

        Args:
            spectrogram: Mel spectrogram [128, 128]

        Returns:
            Probability array for each genre
        """
        with torch.no_grad():
            # Convert to tensor [1, 1, 128, 128]
            spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)

            # Forward pass
            outputs = self.model(spec_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        return probs

    def get_top_predictions(
        self,
        probabilities: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Get top K genre predictions.

        Args:
            probabilities: Probability array for each genre
            top_k: Number of top predictions to return

        Returns:
            List of {genre, confidence} dicts sorted by confidence
        """
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions = [
            {
                "genre": self.genre_names[idx],
                "confidence": float(probabilities[idx])
            }
            for idx in sorted_indices
        ]

        return predictions

    def get_predictions_above_threshold(
        self,
        probabilities: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict[str, any]]:
        """
        Get all predictions above confidence threshold.

        Args:
            probabilities: Probability array for each genre
            threshold: Minimum confidence threshold

        Returns:
            List of {genre, confidence} dicts for predictions above threshold
        """
        above_threshold = [
            {
                "genre": self.genre_names[i],
                "confidence": float(probabilities[i])
            }
            for i in range(len(probabilities))
            if probabilities[i] >= threshold
        ]

        # Sort by confidence descending
        above_threshold.sort(key=lambda x: x['confidence'], reverse=True)

        return above_threshold

    def get_all_probabilities(self, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Get probabilities for all genres.

        Args:
            probabilities: Probability array for each genre

        Returns:
            Dict mapping genre name to probability
        """
        return {
            self.genre_names[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

    @property
    def num_genres(self) -> int:
        """Get number of genres."""
        return len(self.genre_names)

    @property
    def model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            "experiment_name": self.config['experiment_name'],
            "num_genres": self.num_genres,
            "base_channels": self.config['base_channels'],
            "use_attention": self.config['use_attention'],
            "device": str(self.device),
            "genres": self.genre_names
        }
