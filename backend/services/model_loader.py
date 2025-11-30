import httpx
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Remote model server client (HTTP-based)"""
    
    def __init__(self, model_service_url: str = "http://localhost:8001", timeout: float = 30.0):
        self.model_service_url = model_service_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.genre_labels = self._default_genre_labels()
    
    def _default_genre_labels(self) -> List[str]:
        """Default genre labels"""
        return [
            'Blues', 'Classical', 'Country', 'Disco', 'Electronic',
            'Folk', 'Funk', 'Hip-Hop', 'Jazz', 'Metal',
            'Pop', 'Reggae', 'Rock', 'Soul'
        ]
    
    async def health_check(self) -> bool:
        """Check if model server is available"""
        try:
            response = await self.client.get(f"{self.model_service_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Model server health check failed: {e}")
            return False
    
    async def predict(self, spectrogram: np.ndarray) -> Dict:
        """
        Make prediction via remote model server
        
        Args:
            spectrogram: (128, 216) numpy array
        
        Returns:
            Prediction dictionary with genre, confidence, etc.
        """
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
            logger.error(f"Prediction error: {e}")
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
