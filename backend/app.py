"""
Production FastAPI Backend for Music Genre Classification
Handles audio file uploads, feature extraction, and genre prediction.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, cast
import tempfile
import os
import pickle

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.feature_extraction import extract_features, features_to_array
from models.genre_classifier import GenreClassifier, ModelWrapper
from models.model_utils import load_production_model


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration."""
    APP_NAME = "Music Genre Classification API"
    VERSION = "1.0.0"
    MODEL_DIR = "../models/trained_models"
    MODEL_NAME = "genre_classifier_production"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".au", ".flac", ".ogg"}


config = Config()


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for direct feature prediction."""
    features: List[float] = Field(..., min_length=20, max_length=20)
    return_probs: bool = False
    top_k: int = Field(default=3, ge=1, le=8)


class TopPrediction(BaseModel):
    """Single top prediction."""
    genre: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_genre: str
    confidence: float
    top_predictions: List[TopPrediction]
    all_probabilities: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    app_name: str
    version: str
    model_loaded: bool
    device: str


class AudioAnalysisResponse(BaseModel):
    """Response for audio file analysis."""
    filename: str
    duration: float
    features: Dict[str, Any]
    prediction: PredictionResponse


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description="REST API for music genre classification using neural networks",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State
# ============================================================================

model_wrapper: Optional[ModelWrapper] = None


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_wrapper
    
    print(f"Starting {config.APP_NAME} v{config.VERSION}")
    
    # SIMPLIFIED: Always use same feature file
    features_pkl = Path("../data/processed/ml_ready_features.pkl")
    
    if not features_pkl.exists():
        print(f"⚠️  WARNING: {features_pkl} not found")
        print(f"   Run: python scripts/create_features.py")
        print(f"   Using untrained model...")
        
        # Load untrained model
        model = GenreClassifier(input_dim=20, num_classes=8)
        genre_names = ['Rock', 'Electronic', 'Hip-Hop', 'Classical', 'Jazz', 'Folk', 'Pop', 'Experimental']
        
        model_wrapper = ModelWrapper(model=model, scaler=None, genre_names=genre_names, device='cpu')
        return
    
    # Load real data
    with open(features_pkl, 'rb') as f:
        df = pickle.load(f)
    
    genre_names = sorted(df['genre'].unique())
    
    # Try loading trained model
    try:
        model, scaler, _, _ = load_production_model(
            model_class=GenreClassifier,
            model_dir='../models/trained_models',
            model_name='genre_classifier_production',
            device='cpu'
        )
        print(f"✓ Loaded trained model")
    except:
        print(f"⚠️  Trained model not found, using untrained")
        model = GenreClassifier(input_dim=len(df.columns)-1, num_classes=len(genre_names))
        scaler = None
    
    model_wrapper = ModelWrapper(model=model, scaler=scaler, genre_names=genre_names, device='cpu')
    print(f"✓ Model ready: {len(genre_names)} genres")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print(f"Shutting down {config.APP_NAME}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {config.APP_NAME}",
        "version": config.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_wrapper is not None else "unhealthy",
        app_name=config.APP_NAME,
        version=config.VERSION,
        model_loaded=model_wrapper is not None,
        device=model_wrapper.device if model_wrapper else "unknown"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_features(request: PredictionRequest):
    """
    Predict genre from pre-extracted features.
    
    Args:
        request: PredictionRequest with 20 audio features
    
    Returns:
        PredictionResponse with genre prediction and confidence
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to numpy array
        features = np.array(request.features)
        
        # Predict
        result = model_wrapper.predict(
            features=features,
            return_probs=request.return_probs,
            top_k=request.top_k
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/analyze-audio", response_model=AudioAnalysisResponse, tags=["Audio Analysis"])
async def analyze_audio_file(
    file: UploadFile = File(...),
    return_probs: bool = False,
    top_k: int = 3
):
    """
    Analyze uploaded audio file and predict genre.
    
    Args:
        file: Audio file (mp3, wav, au, flac, ogg)
        return_probs: Whether to return all genre probabilities
        top_k: Number of top predictions to return
    
    Returns:
        AudioAnalysisResponse with features and prediction
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {config.ALLOWED_EXTENSIONS}"
        )
    
    # Validate file size
    file_content = await file.read()
    if len(file_content) > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {config.MAX_FILE_SIZE / 1024 / 1024} MB"
        )
    
    # Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # Extract features
        features_dict = extract_features(tmp_path, sr=22050, duration=30.0)
        
        if features_dict is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed")
        
        # Convert to array
        feature_array = features_to_array(features_dict)
        
        # Predict
        prediction_result = model_wrapper.predict(
            features=feature_array,
            return_probs=return_probs,
            top_k=top_k
        )
        
        # Clean up
        os.unlink(tmp_path)
        
        return AudioAnalysisResponse(
            filename=file.filename,
            duration=features_dict.get('duration', 30.0),
            features=features_dict,
            prediction=PredictionResponse(**prediction_result)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")


@app.post("/batch-predict", tags=["Batch Operations"])
async def batch_predict(requests: List[PredictionRequest]):
    """
    Batch prediction for multiple feature sets.
    
    Args:
        requests: List of PredictionRequest objects
    
    Returns:
        List of PredictionResponse objects
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100")
    
    try:
        results = []
        for req in requests:
            features = np.array(req.features)
            result = model_wrapper.predict(
                features=features,
                return_probs=req.return_probs,
                top_k=req.top_k
            )
            results.append(PredictionResponse(**result))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/genres", tags=["Model"])
async def get_genres():
    """Get list of supported genres."""
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "genres": model_wrapper.genre_names,
        "count": len(model_wrapper.genre_names)
    }


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about loaded model."""
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        'status': 'loaded',
        'device': model_wrapper.device,
        'num_classes': len(model_wrapper.genre_names),
        'genre_names': model_wrapper.genre_names,
        'input_features': model_wrapper.model.input_dim if hasattr(model_wrapper.model, 'input_dim') else 20,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Music Genre Classification API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"\nStarting {config.APP_NAME} v{config.VERSION}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health\n")
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
