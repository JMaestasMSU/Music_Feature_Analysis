from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch

from model import ModelLoader

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Music Analysis Model Server",
    description="GPU-accelerated CNN inference for music genre classification",
    version="1.0.0"
)

# Initialize model loader
model_loader = ModelLoader(
    path=str(Path(__file__).parent.parent / "trained_models" / "cnn_best_model.pt"),
    backend="torch"
)
model_loader.load()


class PredictionRequest(BaseModel):
    """Spectrogram input for prediction"""
    spectrogram: List[List[float]]  # (128, 216) spectrogram
    metadata: Optional[Dict] = None


class PredictionResponse(BaseModel):
    """Model prediction response"""
    predicted_genre: str
    confidence: float
    top_3_predictions: List[tuple]
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check model server health"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.model is not None,
        device=model_loader.device
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make prediction on spectrogram
    
    Input: spectrogram as list of lists (128, 216)
    Output: genre prediction with confidence scores
    """
    try:
        # Convert to numpy
        spec_array = np.array(request.spectrogram, dtype=np.float32)
        
        # Validate shape
        if spec_array.shape != (128, 216):
            raise ValueError(f"Expected shape (128, 216), got {spec_array.shape}")
        
        # Get prediction
        result = model_loader.predict(spec_array)
        
        logger.info(f"Prediction: {result['predicted_genre']} ({result['confidence']:.2%})")
        
        return PredictionResponse(
            predicted_genre=result['predicted_genre'],
            confidence=result['confidence'],
            top_3_predictions=result['top_3_predictions'],
            all_probabilities=result['all_probabilities']
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(requests: List[PredictionRequest]) -> Dict:
    """
    Batch predictions for multiple spectrograms
    """
    try:
        results = []
        
        for req in requests:
            spec_array = np.array(req.spectrogram, dtype=np.float32)
            result = model_loader.predict(spec_array)
            results.append({
                'success': True,
                'prediction': result
            })
        
        return {
            'batch_size': len(requests),
            'successful': len(results),
            'results': results
        }
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info() -> Dict:
    """Get model information"""
    return {
        'backend': 'torch',
        'model_loaded': model_loader.model is not None,
        'device': model_loader.device,
        'genres': model_loader.genre_labels if hasattr(model_loader, 'genre_labels') else []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
