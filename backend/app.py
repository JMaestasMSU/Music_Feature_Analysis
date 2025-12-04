"""
Unified FastAPI backend for genre prediction.
Supports both single-label (feature-based) and multi-label (CNN-based) models.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import sys
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from services.prediction_service import PredictionService
from services.feature_service import FeatureService
from services.matlab_interface import MATLABInterface

# ============================================================================
# Configuration
# ============================================================================

# Model directories
CNN_MODEL_DIR = PROJECT_ROOT / "models" / "trained_models" / "multilabel_cnn_filtered_improved"
FEATURE_MODEL_DIR = PROJECT_ROOT / "models"  # Contains ml_ready_features.pt

# Audio processing config
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 30
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# ============================================================================
# Response Models
# ============================================================================

class GenrePrediction(BaseModel):
    genre: str
    confidence: float

class TopPrediction(BaseModel):
    genre: str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_genre: str
    confidence: float
    top_predictions: List[TopPrediction]

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    device: str
    available_models: List[str]

class AudioAnalysisResponse(BaseModel):
    filename: str
    duration: float
    predicted_genre: str
    confidence: float
    top_predictions: List[TopPrediction]
    processing_time_ms: int
    all_probabilities: Optional[Dict[str, float]] = None

class BatchPredictionRequest(BaseModel):
    features_list: List[List[float]] = Field(..., description="List of feature arrays (each 20 elements)")
    return_probs: bool = False
    top_k: int = 3

class MultiLabelPredictionResponse(BaseModel):
    predictions: List[GenrePrediction]
    threshold: float
    count: int
    processing_time_ms: int

class ModelInfoResponse(BaseModel):
    model_type: str
    architecture: str
    num_genres: int
    genres: List[str]
    device: str
    additional_info: Dict[str, Any]

# ============================================================================
# Global Services (loaded at startup)
# ============================================================================

feature_service = None
cnn_prediction_service = None
feature_prediction_service = None
matlab_interface = None

def load_services():
    """Load services at startup."""
    global feature_service, cnn_prediction_service, feature_prediction_service, matlab_interface

    print("\n" + "="*80)
    print("LOADING UNIFIED GENRE CLASSIFICATION SERVICES")
    print("="*80)

    # Initialize feature extraction service
    feature_service = FeatureService(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    print(f"Feature Service: sample_rate={SAMPLE_RATE}, n_mels={N_MELS}, n_fft={N_FFT}")

    # Initialize CNN prediction service (multi-label)
    try:
        cnn_prediction_service = PredictionService(
            model_type="multi-label",
            model_dir=CNN_MODEL_DIR
        )
        print(f"CNN Model: {cnn_prediction_service.num_genres} genres, device={cnn_prediction_service.device}")
        print(f"  Architecture: {cnn_prediction_service.model_info.get('architecture', 'unknown')}")
    except Exception as e:
        import traceback
        logger.error(f"Failed to load CNN model: {e}")
        logger.error(traceback.format_exc())
        cnn_prediction_service = None

    # Initialize feature-based prediction service (single-label)
    try:
        feature_prediction_service = PredictionService(
            model_type="single-label",
            model_dir=FEATURE_MODEL_DIR
        )
        print(f"Feature Model: {feature_prediction_service.num_genres} genres, device={feature_prediction_service.device}")
        print(f"  Architecture: {feature_prediction_service.model_info.get('architecture', 'unknown')}")
    except Exception as e:
        logger.warning(f"Failed to load feature-based model: {e}")
        feature_prediction_service = None

    # Initialize MATLAB interface (optional)
    try:
        matlab_interface = MATLABInterface()
        matlab_available = matlab_interface.validate_matlab_available()
        if matlab_available:
            print(f"MATLAB Interface: Available (FFT validation enabled)")
        else:
            print(f"MATLAB Interface: Not available (using NumPy fallback)")
    except Exception as e:
        logger.warning(f"MATLAB interface initialization failed: {e}")
        matlab_interface = None

    print("="*80 + "\n")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Music Genre Classification API (CNN)",
    description="Multi-label genre prediction using CNN on spectrograms",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load services on startup."""
    load_services()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    available_models = []
    if cnn_prediction_service:
        available_models.append("multi-label")
    if feature_prediction_service:
        available_models.append("single-label")
    
    matlab_status = "unavailable"
    if matlab_interface:
        matlab_status = "available" if matlab_interface.validate_matlab_available() else "fallback"
    
    return {
        "message": "Unified Music Genre Classification API",
        "version": "2.0.0",
        "available_models": available_models,
        "matlab_fft": matlab_status,
        "endpoints": {
            "predict_features": "POST /predict",
            "analyze_audio": "POST /analyze-audio",
            "predict_multilabel": "POST /api/v1/analysis/predict-multilabel",
            "batch_predict": "POST /batch-predict",
            "genres": "GET /genres",
            "model_info": "GET /model/info",
            "health": "GET /health",
            "matlab_status": "GET /matlab/status",
            "fft_analysis": "POST /api/v1/analysis/fft-analysis",
            "validate_features": "POST /api/v1/analysis/validate-features",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    models_loaded = []
    if cnn_prediction_service:
        models_loaded.append("multi-label")
    if feature_prediction_service:
        models_loaded.append("single-label")
    
    device = "unknown"
    if cnn_prediction_service:
        device = str(cnn_prediction_service.device)
    elif feature_prediction_service:
        device = str(feature_prediction_service.device)
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        app_name="Music Genre Classification API",
        version="2.0.0",
        model_loaded=len(models_loaded) > 0,
        device=device,
        available_models=models_loaded
    )

@app.get("/genres")
async def get_genres(model_type: str = Query("multi-label", description="Model type: single-label or multi-label")):
    """Get list of supported genres for specified model."""
    if model_type == "single-label" and feature_prediction_service:
        return {
            "model_type": "single-label",
            "genres": feature_prediction_service.genre_names,
            "count": feature_prediction_service.num_genres
        }
    elif model_type == "multi-label" and cnn_prediction_service:
        return {
            "model_type": "multi-label",
            "genres": cnn_prediction_service.genre_names,
            "count": cnn_prediction_service.num_genres
        }
    else:
        raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not available")

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(model_type: str = Query("multi-label", description="Model type: single-label or multi-label")):
    """Get model information for specified model."""
    if model_type == "single-label" and feature_prediction_service:
        info = feature_prediction_service.model_info
        return ModelInfoResponse(
            model_type=info['model_type'],
            architecture=info['architecture'],
            num_genres=info['num_genres'],
            genres=info['genres'],
            device=info['device'],
            additional_info={k: v for k, v in info.items() if k not in ['model_type', 'architecture', 'num_genres', 'genres', 'device']}
        )
    elif model_type == "multi-label" and cnn_prediction_service:
        info = cnn_prediction_service.model_info
        return ModelInfoResponse(
            model_type=info['model_type'],
            architecture=info['architecture'],
            num_genres=info['num_genres'],
            genres=info['genres'],
            device=info['device'],
            additional_info={k: v for k, v in info.items() if k not in ['model_type', 'architecture', 'num_genres', 'genres', 'device']}
        )
    else:
        raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not available")

# ============================================================================
# Main Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_from_features(
    features: List[float],
    return_probs: bool = False,
    top_k: int = 3
):
    """
    Predict genre from hand-crafted audio features (single-label model).
    
    Args:
        features: 20-dimensional feature vector
        return_probs: Whether to return all class probabilities
        top_k: Number of top predictions to return
        
    Returns:
        PredictionResponse with top genre predictions
    """
    if not feature_prediction_service:
        raise HTTPException(status_code=503, detail="Feature-based model not available")
    
    if len(features) != 20:
        raise HTTPException(status_code=400, detail="Expected 20 features, got {len(features)}")
    
    try:
        features_array = np.array(features)
        result = feature_prediction_service.predict_from_features(
            features_array,
            top_k=top_k,
            return_probs=return_probs
        )
        
        top_preds = [TopPrediction(**p) for p in result['top_predictions']]
        
        response = PredictionResponse(
            predicted_genre=result['predicted_genre'],
            confidence=result['confidence'],
            top_predictions=top_preds
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    model_type: str = Query("multi-label", description="Model type: single-label or multi-label"),
    return_probs: bool = False,
    top_k: int = 5
):
    """
    Analyze audio file and return genre predictions.
    Supports both single-label (feature-based) and multi-label (CNN) models.

    Args:
        file: Audio file (MP3, WAV, FLAC, etc.)
        model_type: "single-label" or "multi-label"
        return_probs: Whether to return all genre probabilities
        top_k: Number of top predictions

    Returns:
        AudioAnalysisResponse with predictions
    """
    start_time = time.time()

    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.au')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Validate requested model is available
    if model_type == "single-label" and not feature_prediction_service:
        raise HTTPException(status_code=503, detail="Single-label model not loaded")
    if model_type == "multi-label" and not cnn_prediction_service:
        raise HTTPException(status_code=503, detail="Multi-label CNN model not loaded")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")
            tmp.write(content)
            tmp_path = tmp.name

        metadata = feature_service.get_audio_metadata(tmp_path)
        
        if model_type == "single-label" and feature_prediction_service:
            features = feature_service.extract_hand_crafted_features(tmp_path)
            result = feature_prediction_service.predict_from_features(features, top_k=top_k, return_probs=return_probs)
            
        elif model_type == "multi-label" and cnn_prediction_service:
            mel_spec = feature_service.extract_spectrogram(tmp_path, duration=DURATION)
            result = cnn_prediction_service.predict_from_spectrogram(mel_spec, top_k=top_k)
            
            if return_probs:
                result['all_probabilities'] = cnn_prediction_service.get_all_probabilities(mel_spec, input_type="spectrogram")
        else:
            raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not available")

        top_preds = [TopPrediction(**p) for p in result['top_predictions']]
        response = AudioAnalysisResponse(
            filename=metadata['filename'],
            duration=metadata['duration'],
            predicted_genre=result['predicted_genre'],
            confidence=result['confidence'],
            top_predictions=top_preds,
            processing_time_ms=int((time.time() - start_time) * 1000),
            all_probabilities=result.get('all_probabilities')
        )

        Path(tmp_path).unlink()
        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = str(e) or "Unknown error occurred"
        logger.error(f"Error analyzing audio: {error_msg}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing audio: {error_msg}")

@app.post("/api/v1/analysis/predict-multilabel", response_model=MultiLabelPredictionResponse)
async def predict_multilabel(
    file: UploadFile = File(...),
    threshold: float = 0.3
):
    """
    Multi-label prediction endpoint (returns all genres above threshold).

    Args:
        file: Audio file
        threshold: Minimum confidence threshold

    Returns:
        All genres above threshold with confidences
    """
    if not cnn_prediction_service:
        raise HTTPException(status_code=503, detail="Multi-label model not available")
    
    start_time = time.time()

    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.au')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")
            tmp.write(content)
            tmp_path = tmp.name

        mel_spec = feature_service.extract_spectrogram(tmp_path, duration=DURATION)
        above_threshold = cnn_prediction_service.predict_multi_label(mel_spec, threshold=threshold)
        
        Path(tmp_path).unlink()

        predictions = [GenrePrediction(**p) for p in above_threshold]
        return MultiLabelPredictionResponse(
            predictions=predictions,
            threshold=threshold,
            count=len(predictions),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )

    except Exception as e:
        logger.error(f"Error in multi-label prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction from multiple feature arrays (single-label model only).
    
    Args:
        request: BatchPredictionRequest with features_list, return_probs, top_k
        
    Returns:
        List of predictions for each feature array
    """
    if not feature_prediction_service:
        raise HTTPException(status_code=503, detail="Feature-based model not available")
    
    if len(request.features_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 samples per batch")
    
    try:
        results = []
        for i, features in enumerate(request.features_list):
            if len(features) != 20:
                raise HTTPException(status_code=400, detail=f"Sample {i}: Expected 20 features, got {len(features)}")
            
            features_array = np.array(features)
            result = feature_prediction_service.predict_from_features(
                features_array,
                top_k=request.top_k,
                return_probs=request.return_probs
            )
            
            results.append(result)
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ============================================================================
# MATLAB FFT Validation Endpoints
# ============================================================================

class FFTAnalysisResponse(BaseModel):
    spectral_centroid: float
    spectral_spread: float
    spectral_rolloff: float
    time_domain_energy: float
    freq_domain_energy: float
    source: str
    filename: str
    duration: float
    processing_time_ms: int

class FFTComparisonResponse(BaseModel):
    fft_analysis: Dict[str, Any]
    ml_features: Dict[str, Any]
    correlation: Dict[str, Any]
    filename: str

@app.get("/matlab/status")
async def matlab_status():
    """
    Check MATLAB availability status.
    
    Returns:
        MATLAB status and version information
    """
    if not matlab_interface:
        return {
            "available": False,
            "status": "MATLAB interface not initialized"
        }
    
    is_available = matlab_interface.validate_matlab_available()
    
    return {
        "available": is_available,
        "status": "MATLAB ready" if is_available else "MATLAB not found (using NumPy fallback)",
        "fallback_available": True
    }

@app.post("/api/v1/analysis/fft-analysis", response_model=FFTAnalysisResponse)
async def fft_analysis(file: UploadFile = File(...)):
    """
    Perform FFT spectral analysis on audio file using MATLAB (or NumPy fallback).
    
    Computes:
    - Spectral centroid (center of mass in frequency domain)
    - Spectral spread (dispersion around centroid)
    - Spectral rolloff (95th percentile frequency)
    - Parseval's theorem validation (time vs frequency domain energy)
    
    Args:
        file: Audio file (MP3, WAV, FLAC, etc.)
        
    Returns:
        FFT analysis results with spectral features
    """
    if not matlab_interface:
        raise HTTPException(status_code=503, detail="MATLAB interface not available")
    
    start_time = time.time()
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.au')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load audio
        audio, sr = feature_service.load_audio(tmp_path)
        metadata = feature_service.get_audio_metadata(tmp_path)
        
        # Run FFT analysis via MATLAB (or NumPy fallback)
        fft_results = matlab_interface.run_fft_analysis(audio, sr)
        
        Path(tmp_path).unlink()
        
        return FFTAnalysisResponse(
            spectral_centroid=fft_results['spectral_centroid'],
            spectral_spread=fft_results['spectral_spread'],
            spectral_rolloff=fft_results['spectral_rolloff'],
            time_domain_energy=fft_results['time_domain_energy'],
            freq_domain_energy=fft_results['freq_domain_energy'],
            source=fft_results['source'],
            filename=metadata['filename'],
            duration=metadata['duration'],
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
    except Exception as e:
        logger.error(f"FFT analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"FFT analysis error: {str(e)}")

@app.post("/api/v1/analysis/validate-features", response_model=FFTComparisonResponse)
async def validate_features(file: UploadFile = File(...)):
    """
    Validate ML-extracted features against MATLAB FFT analysis.
    
    Compares:
    - Hand-crafted features (librosa) vs MATLAB FFT features
    - Spectral centroid correlation
    - Energy validation (Parseval's theorem)
    
    Args:
        file: Audio file (MP3, WAV, FLAC, etc.)
        
    Returns:
        Comparison of FFT and ML features with correlation metrics
    """
    if not matlab_interface:
        raise HTTPException(status_code=503, detail="MATLAB interface not available")
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.au')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load audio
        audio, sr = feature_service.load_audio(tmp_path)
        
        # Extract hand-crafted features
        features_array = feature_service.extract_hand_crafted_features(tmp_path)
        
        # Create feature dict (simplified - map to known indices)
        ml_features = {
            'spectral_centroid': float(features_array[2]) if len(features_array) > 2 else None,
            'spectral_rolloff': float(features_array[6]) if len(features_array) > 6 else None,
            'zero_crossing_rate': float(features_array[8]) if len(features_array) > 8 else None
        }
        
        # Run FFT analysis
        fft_results = matlab_interface.run_fft_analysis(audio, sr)
        
        # Compare features
        comparison = matlab_interface.validate_fft_vs_ml_features(fft_results, ml_features)
        
        Path(tmp_path).unlink()
        
        return FFTComparisonResponse(
            fft_analysis=comparison['fft'],
            ml_features=comparison['ml'],
            correlation=comparison['correlation'],
            filename=file.filename
        )
        
    except Exception as e:
        logger.error(f"Feature validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature validation error: {str(e)}")

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    print("\nStarting Music Genre Classification API (CNN)...")
    print("Access API docs at: http://localhost:8000/docs\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
