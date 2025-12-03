import httpx
import numpy as np
from typing import Dict, List, Optional
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Unified model loader supporting both local and remote inference.

    Modes:
    - Remote: HTTP-based inference via model server (default)
    - Local: Direct PyTorch inference (fallback or when specified)
    """

    def __init__(self,
                 model_service_url: str = "http://localhost:8001",
                 timeout: float = 30.0,
                 local_model_path: Optional[str] = None,
                 device: str = "cpu"):
        self.model_service_url = model_service_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.device = torch.device(device)

        # Local model support
        self.local_model = None
        self.local_model_path = local_model_path
        self.model_metadata = None

        if local_model_path:
            self._load_local_model(local_model_path)

        self.genre_labels = self._default_genre_labels()

    def _default_genre_labels(self) -> List[str]:
        """Default genre labels (can be overridden by model metadata)"""
        return [
            'Blues', 'Classical', 'Country', 'Disco', 'Electronic',
            'Folk', 'Funk', 'Hip-Hop', 'Jazz', 'Metal',
            'Pop', 'Reggae', 'Rock', 'Soul'
        ]

    def _load_local_model(self, model_path: str):
        """Load PyTorch model for local inference."""
        try:
            from models.cnn_model import MultiLabelAudioCNN

            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract model configuration
            num_genres = checkpoint.get('num_genres', 50)
            self.threshold = checkpoint.get('threshold', 0.5)

            # Initialize model
            self.local_model = MultiLabelAudioCNN(
                num_genres=num_genres,
                input_channels=1,
                base_channels=64,
                use_attention=True
            )

            # Load weights
            self.local_model.load_state_dict(checkpoint['model_state_dict'])
            self.local_model.to(self.device)
            self.local_model.eval()

            self.model_metadata = {
                'num_genres': num_genres,
                'threshold': self.threshold,
                'epoch': checkpoint.get('epoch', 'unknown'),
                'val_f1': checkpoint.get('val_f1', 'unknown')
            }

            logger.info(f"Loaded local model: {model_path}")
            logger.info(f"Model metadata: {self.model_metadata}")

        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if model server is available"""
        try:
            response = await self.client.get(f"{self.model_service_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Model server health check failed: {e}")
            return False
    
    def _predict_local(self, spectrogram: np.ndarray) -> Dict:
        """
        Make prediction using local PyTorch model.

        Args:
            spectrogram: Spectrogram array

        Returns:
            Prediction dictionary with multi-label support
        """
        if self.local_model is None:
            raise ValueError("No local model loaded")

        # Prepare input tensor
        if spectrogram.ndim == 2:
            spectrogram = spectrogram[np.newaxis, np.newaxis, :, :]
        elif spectrogram.ndim == 3:
            spectrogram = spectrogram[np.newaxis, :, :, :]

        input_tensor = torch.FloatTensor(spectrogram).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.local_model(input_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        # Multi-label predictions (threshold-based)
        predictions = (probabilities > self.threshold).astype(int)
        predicted_indices = np.nonzero(predictions == 1)[0]

        # Build response
        result = {
            'type': 'multi-label',
            'threshold': float(self.threshold),
            'predicted_genres': [
                {
                    'genre': self.genre_labels[idx] if idx < len(self.genre_labels) else f'Genre_{idx}',
                    'probability': float(probabilities[idx]),
                    'index': int(idx)
                }
                for idx in predicted_indices
            ],
            'all_probabilities': {
                (self.genre_labels[i] if i < len(self.genre_labels) else f'Genre_{i}'): float(probabilities[i])
                for i in range(len(probabilities))
            },
            'top_5': [
                {
                    'genre': self.genre_labels[idx] if idx < len(self.genre_labels) else f'Genre_{idx}',
                    'probability': float(probabilities[idx]),
                    'index': int(idx)
                }
                for idx in np.argsort(probabilities)[::-1][:5]
            ],
            'model_metadata': self.model_metadata
        }

        return result

    async def predict(self, spectrogram: np.ndarray, force_local: bool = False) -> Dict:
        """
        Make prediction via remote server or local model.

        Args:
            spectrogram: Spectrogram array
            force_local: Force local inference even if remote server available

        Returns:
            Prediction dictionary with genre, confidence, etc.
        """
        # Use local model if available and requested, or if remote fails
        if force_local and self.local_model:
            return self._predict_local(spectrogram)

        # Try remote first
        try:
            # Ensure correct shape
            if spectrogram.ndim == 2:
                spectrogram = np.expand_dims(spectrogram, axis=0)
            if spectrogram.ndim == 3:
                spectrogram = np.expand_dims(spectrogram, axis=0)

            # Prepare request
            payload = {
                'spectrogram': spectrogram[0].tolist() if spectrogram.shape[0] == 1 else spectrogram.tolist(),
                'metadata': {}
            }

            # Call remote server
            response = await self.client.post(
                f"{self.model_service_url}/predict",
                json=payload
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prediction failed: {response.text}")
                raise Exception("Model server prediction failed")

        except Exception as e:
            logger.warning(f"Remote prediction failed: {e}")

            # Fallback to local if available
            if self.local_model:
                logger.info("Falling back to local inference")
                return self._predict_local(spectrogram)
            else:
                logger.error("No local model available for fallback")
                raise
    
    async def batch_predict(self, spectrograms: List[np.ndarray]) -> List[Dict]:
        """Batch predictions"""
        try:
            payloads = [
                {
                    'spectrogram': spec.tolist(),
                    'metadata': {}
                }
                for spec in spectrograms
            ]
            
            response = await self.client.post(
                f"{self.model_service_url}/batch-predict",
                json=payloads
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception("Batch prediction failed")
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
