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
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.feature_extraction import extract_features, features_to_array
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
    features: List[float] = Field(
        ...,
        min_length=20,
        max_length=20,
        description="Array of 20 audio features (13 MFCCs + spectral centroid + spectral rolloff + ZCR + RMS energy + 3 chroma features)",
        examples=[[0.1, -0.5, 0.3, 0.2, -0.1, 0.4, 0.0, -0.2, 0.1, 0.3, -0.4, 0.2, 0.1, 1500.5, 3200.1, 0.05, 0.02, 0.3, 0.4, 0.3]]
    )
    return_probs: bool = Field(
        default=False,
        description="Whether to return probabilities for all genres"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Number of top predictions to return (1-8)"
    )


class TopPrediction(BaseModel):
    """Single top prediction."""
    genre: str = Field(description="Genre name")
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"genre": "Rock", "confidence": 0.85}
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_genre: str = Field(description="Most likely genre")
    confidence: float = Field(description="Confidence of top prediction (0.0-1.0)", ge=0.0, le=1.0)
    top_predictions: List[TopPrediction] = Field(description="Top K genre predictions ranked by confidence")
    all_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probabilities for all genres (only if return_probs=true)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_genre": "Rock",
                    "confidence": 0.85,
                    "top_predictions": [
                        {"genre": "Rock", "confidence": 0.85},
                        {"genre": "Pop", "confidence": 0.10},
                        {"genre": "Electronic", "confidence": 0.03}
                    ],
                    "all_probabilities": {
                        "Rock": 0.85,
                        "Pop": 0.10,
                        "Electronic": 0.03,
                        "Hip-Hop": 0.01,
                        "Classical": 0.005,
                        "Jazz": 0.003,
                        "Folk": 0.001,
                        "Experimental": 0.001
                    }
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}

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
# Global State
# ============================================================================

model_wrapper: Optional[ModelWrapper] = None


# ============================================================================
# Lifespan Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global model_wrapper

    # Startup
    print(f"Starting {config.APP_NAME} v{config.VERSION}")

    # SIMPLIFIED: Always use same feature file
    features_pkl = Path("../data/processed/ml_ready_features.pkl")

    if not features_pkl.exists():
        print(f"  WARNING: {features_pkl} not found")
        print(f"   Run: python scripts/create_features.py")
        print(f"   Using untrained model...")

        # Load untrained model
        model = GenreClassifier(input_dim=20, num_classes=8)
        genre_names = ['Rock', 'Electronic', 'Hip-Hop', 'Classical', 'Jazz', 'Folk', 'Pop', 'Experimental']

        model_wrapper = ModelWrapper(model=model, scaler=None, genre_names=genre_names, device='cpu')
    else:
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
            print(f" Loaded trained model")
        except:
            print(f"  Trained model not found, using untrained")
            model = GenreClassifier(input_dim=len(df.columns)-1, num_classes=len(genre_names))
            scaler = None

        model_wrapper = ModelWrapper(model=model, scaler=scaler, genre_names=genre_names, device='cpu')
        print(f" Model ready: {len(genre_names)} genres")

    yield

    # Shutdown
    print(f"Shutting down {config.APP_NAME}")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description="""
## Music Genre Classification API

A production-ready REST API for music genre classification using deep learning.

### Features
* **Feature-based Prediction**: Classify music from pre-extracted audio features
* **Audio File Analysis**: Upload audio files for automatic feature extraction and classification
* **Batch Processing**: Process multiple samples in a single request
* **Model Information**: Access model metadata and supported genres

### Supported Audio Formats
MP3, WAV, AU, FLAC, OGG

### Model Architecture
* Deep neural network trained on audio features
* 20 input features (MFCCs, spectral features, chroma, etc.)
* 8 music genres supported

### Quick Start
1. Check API health: `GET /health`
2. View supported genres: `GET /genres`
3. Classify audio: `POST /analyze-audio` or `POST /predict`

### Documentation
* **Swagger UI**: [/docs](/docs)
* **ReDoc**: [/redoc](/redoc)
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Music Genre Classification API",
        "url": "https://github.com/yourusername/Music_Feature_Analysis",
    },
    license_info={
        "name": "MIT License",
    }
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
# API Endpoints
# ============================================================================

@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Get API information and navigation links"
)
async def root():
    """
    # Welcome to Music Genre Classification API

    Returns basic API information and links to documentation.
    """
    return {
        "message": f"Welcome to {config.APP_NAME}",
        "version": config.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check API health status and model availability"
)
async def health_check():
    """
    # Health Check Endpoint

    Returns the current health status of the API, including:
    - Overall API status
    - Model loading status
    - Compute device being used (CPU/CUDA)

    **Use this endpoint** to verify the API is running before making predictions.
    """
    return HealthResponse(
        status="healthy" if model_wrapper is not None else "unhealthy",
        app_name=config.APP_NAME,
        version=config.VERSION,
        model_loaded=model_wrapper is not None,
        device=model_wrapper.device if model_wrapper else "unknown"
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict Genre from Features",
    description="Classify music genre using pre-extracted audio features",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_genre": "Rock",
                        "confidence": 0.85,
                        "top_predictions": [
                            {"genre": "Rock", "confidence": 0.85},
                            {"genre": "Pop", "confidence": 0.10},
                            {"genre": "Electronic", "confidence": 0.03}
                        ]
                    }
                }
            }
        },
        422: {"description": "Validation error - invalid features"},
        503: {"description": "Model not loaded"}
    }
)
async def predict_from_features(request: PredictionRequest):
    """
    # Predict Genre from Audio Features

    Classify music genre using 20 pre-extracted audio features.

    ## Required Features (in order):
    1-13. **MFCC coefficients** (Mel-frequency cepstral coefficients)
    14. **Spectral centroid** - brightness of the sound
    15. **Spectral rolloff** - shape of the signal
    16. **Zero crossing rate** - percussiveness
    17. **RMS energy** - loudness
    18-20. **Chroma features** - harmonic content (3 values)

    ## Example Usage:
    ```python
    import requests

    features = [0.1, -0.5, 0.3, ..., 0.4, 0.3]  # 20 features
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "features": features,
            "return_probs": True,
            "top_k": 3
        }
    )
    ```

    ## Parameters:
    - **features**: Array of exactly 20 float values
    - **return_probs**: Set to `true` to get probabilities for all genres
    - **top_k**: Number of top predictions to return (1-8)
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


@app.post(
    "/analyze-audio",
    response_model=AudioAnalysisResponse,
    tags=["Audio Analysis"],
    summary="Analyze Audio File",
    description="Upload an audio file for automatic feature extraction and genre classification",
    responses={
        200: {"description": "Analysis complete with genre prediction"},
        400: {"description": "Invalid file type or missing filename"},
        413: {"description": "File too large (max 50MB)"},
        500: {"description": "Feature extraction or prediction failed"},
        503: {"description": "Model not loaded"}
    }
)
async def analyze_audio_file(
    file: UploadFile = File(..., description="Audio file to analyze"),
    return_probs: bool = False,
    top_k: int = 3
):
    """
    # Analyze Audio File and Predict Genre

    Upload an audio file and get automatic genre classification.

    ## Process:
    1. **Upload** audio file (MP3, WAV, AU, FLAC, OGG)
    2. **Extract** 20 audio features automatically using librosa
    3. **Predict** genre using trained neural network
    4. **Return** prediction results and extracted features

    ## Supported Formats:
    - MP3
    - WAV
    - AU
    - FLAC
    - OGG

    ## Constraints:
    - **Max file size**: 50 MB
    - **Duration**: First 30 seconds analyzed
    - **Sample rate**: Resampled to 22,050 Hz

    ## Example Usage (curl):
    ```bash
    curl -X POST "http://localhost:8000/analyze-audio" \\
      -F "file=@song.mp3" \\
      -F "return_probs=true" \\
      -F "top_k=3"
    ```

    ## Example Usage (Python):
    ```python
    import requests

    with open("song.mp3", "rb") as f:
        response = requests.post(
            "http://localhost:8000/analyze-audio",
            files={"file": f},
            params={"return_probs": True, "top_k": 3}
        )
    ```
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


@app.post(
    "/batch-predict",
    tags=["Batch Operations"],
    summary="Batch Genre Prediction",
    description="Predict genres for multiple audio feature sets in a single request",
    responses={
        200: {"description": "Batch prediction successful"},
        400: {"description": "Batch size exceeds limit (max 100)"},
        503: {"description": "Model not loaded"}
    }
)
async def batch_predict(requests: List[PredictionRequest]):
    """
    # Batch Genre Prediction

    Classify multiple audio samples efficiently in a single API call.

    ## Benefits:
    - **Reduced latency**: Single network call for multiple predictions
    - **Efficient processing**: Batch inference optimization
    - **Bulk analysis**: Process entire playlists or albums

    ## Constraints:
    - **Maximum batch size**: 100 samples per request
    - Each sample requires exactly 20 features

    ## Example Usage:
    ```python
    import requests

    batch = [
        {
            "features": [0.1, -0.5, ..., 0.3],  # Song 1
            "return_probs": False,
            "top_k": 1
        },
        {
            "features": [0.2, -0.3, ..., 0.4],  # Song 2
            "return_probs": False,
            "top_k": 1
        }
    ]

    response = requests.post(
        "http://localhost:8000/batch-predict",
        json=batch
    )
    ```
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


@app.get(
    "/genres",
    tags=["Model Information"],
    summary="Get Supported Genres",
    description="Returns the list of music genres that the model can classify"
)
async def get_genres():
    """
    # Get Supported Genres

    Returns all music genres that the model can classify.

    ## Response:
    - **genres**: Array of genre names
    - **count**: Total number of genres

    ## Example Response:
    ```json
    {
        "genres": [
            "Rock",
            "Electronic",
            "Hip-Hop",
            "Classical",
            "Jazz",
            "Folk",
            "Pop",
            "Experimental"
        ],
        "count": 8
    }
    ```
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "genres": model_wrapper.genre_names,
        "count": len(model_wrapper.genre_names)
    }


@app.get(
    "/model/info",
    tags=["Model Information"],
    summary="Get Model Information",
    description="Returns detailed information about the loaded model"
)
async def get_model_info():
    """
    # Get Model Information

    Returns metadata and configuration details about the loaded classification model.

    ## Response Fields:
    - **status**: Model loading status
    - **device**: Compute device (CPU or CUDA)
    - **num_classes**: Number of genre classes
    - **genre_names**: List of supported genres
    - **input_features**: Number of input features expected (20)

    ## Example Response:
    ```json
    {
        "status": "loaded",
        "device": "cpu",
        "num_classes": 8,
        "genre_names": ["Rock", "Electronic", "Hip-Hop", "Classical", "Jazz", "Folk", "Pop", "Experimental"],
        "input_features": 20
    }
    ```
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        'status': 'loaded',
        'device': model_wrapper.device,
        'num_classes': len(model_wrapper.genre_names),
        'genre_names': model_wrapper.genre_names,
        'input_features': model_wrapper.model.input_dim if hasattr(model_wrapper.model, 'input_dim') else 20,
    }


# Add new endpoint for multi-label prediction
@app.post("/api/v1/analysis/predict-multilabel")
async def predict_multilabel(file: UploadFile = File(...), threshold: float = 0.3):
    """
    Multi-label genre classification endpoint.
    
    Args:
        file: Audio file (WAV, MP3, FLAC)
        threshold: Probability threshold for genre inclusion (default: 0.3)
    
    Returns:
        List of genres with probabilities above threshold
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Extract features (spectrogram)
        # TODO: Implement feature extraction
        features = extract_spectrogram(temp_path)
        
        # Call model server for prediction
        # TODO: Implement model server call
        predictions = call_model_server(features, mode="multilabel")
        
        # Filter by threshold
        results = []
        for genre, prob in predictions.items():
            if prob >= threshold:
                results.append({"genre": genre, "probability": float(prob)})
        
        # Sort by probability
        results.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "status": "success",
            "file": file.filename,
            "threshold": threshold,
            "genres": results,
            "num_genres": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
