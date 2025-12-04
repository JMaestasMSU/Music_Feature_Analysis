"""
Simple FastAPI backend for CNN-based genre prediction.
Uses the trained multilabel CNN model with service layer architecture.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
from pathlib import Path
from typing import List
import time
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from services.audio_processor import AudioProcessor
from services.cnn_model_service import CNNModelService

# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = PROJECT_ROOT / "models" / "trained_models" / "multilabel_cnn_filtered_improved"
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

class PredictionResponse(BaseModel):
    predictions: List[GenrePrediction]
    top_k: int
    threshold: float
    processing_time_ms: int

# ============================================================================
# Global Services (loaded at startup)
# ============================================================================

audio_processor = None
model_service = None

def load_services():
    """Load services at startup."""
    global audio_processor, model_service

    print("\n" + "="*80)
    print("LOADING SERVICES")
    print("="*80)

    # Initialize audio processor
    audio_processor = AudioProcessor(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT
    )
    print(f"Audio Processor: sample_rate={SAMPLE_RATE}, n_mels={N_MELS}, n_fft={N_FFT}")

    # Initialize model service
    model_service = CNNModelService(model_dir=MODEL_DIR)
    print(f"Model Service: {model_service.model_info['experiment_name']}")
    print(f"Genres: {model_service.num_genres}")
    print(f"Device: {model_service.device}")

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
    """Root endpoint."""
    return {
        "message": "Music Genre Classification API (CNN)",
        "version": "2.0.0",
        "model": model_service.model_info['experiment_name'] if model_service else "Not loaded",
        "genres": model_service.num_genres if model_service else 0,
        "endpoints": {
            "predict": "/api/v1/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_service is not None,
        "device": str(model_service.device) if model_service else "unknown"
    }

@app.get("/genres")
async def get_genres():
    """Get list of supported genres."""
    return {
        "genres": model_service.genre_names,
        "count": model_service.num_genres
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    return model_service.model_info

# ============================================================================
# Main Prediction Endpoints
# ============================================================================

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    top_k: int = 5
):
    """
    Predict genres for uploaded audio file.

    Args:
        file: Audio file (MP3, WAV, FLAC, etc.)
        threshold: Minimum confidence threshold (0.0-1.0)
        top_k: Number of top predictions to return

    Returns:
        PredictionResponse with top genre predictions
    """
    start_time = time.time()

    # Validate file
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.au')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")

            tmp.write(content)
            tmp_path = tmp.name

        # Process audio to spectrogram using AudioProcessor service
        mel_spec_db = audio_processor.process_audio_for_cnn(tmp_path, duration=DURATION)

        # Predict using CNNModelService
        probabilities = model_service.predict(mel_spec_db)

        # Get top predictions using service method
        top_predictions = model_service.get_top_predictions(probabilities, top_k=top_k)

        # Convert to response format
        predictions = [
            GenrePrediction(
                genre=pred["genre"],
                confidence=pred["confidence"]
            )
            for pred in top_predictions
        ]

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)

        # Clean up temp file
        Path(tmp_path).unlink()

        return PredictionResponse(
            predictions=predictions,
            top_k=top_k,
            threshold=threshold,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/api/v1/analysis/analyze-audio")  # Alias
async def analyze_audio(
    file: UploadFile = File(...),
    return_probs: bool = False,
    top_k: int = 5
):
    """
    Analyze audio file and return predictions (alternative endpoint format).

    Args:
        file: Audio file
        return_probs: Whether to return probabilities for all genres
        top_k: Number of top predictions

    Returns:
        Analysis with predictions
    """
    start_time = time.time()

    # Validate file
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.au')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")

            tmp.write(content)
            tmp_path = tmp.name

        # Process audio to spectrogram
        mel_spec_db = audio_processor.process_audio_for_cnn(tmp_path, duration=DURATION)

        # Predict
        probabilities = model_service.predict(mel_spec_db)

        # Get top predictions
        top_predictions = model_service.get_top_predictions(probabilities, top_k=top_k)

        # Format response in old API style
        response = {
            "predicted_genre": top_predictions[0]["genre"] if top_predictions else "Unknown",
            "confidence": top_predictions[0]["confidence"] if top_predictions else 0.0,
            "top_predictions": top_predictions,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

        if return_probs:
            # Return all probabilities using service method
            response["all_probabilities"] = model_service.get_all_probabilities(probabilities)

        # Clean up temp file
        Path(tmp_path).unlink()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/api/v1/analysis/predict-multilabel")
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
    start_time = time.time()

    # Validate file
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.au')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 50MB)")

            tmp.write(content)
            tmp_path = tmp.name

        # Process audio to spectrogram
        mel_spec_db = audio_processor.process_audio_for_cnn(tmp_path, duration=DURATION)

        # Predict
        probabilities = model_service.predict(mel_spec_db)

        # Get predictions above threshold using service method
        above_threshold = model_service.get_predictions_above_threshold(probabilities, threshold=threshold)

        # Clean up temp file
        Path(tmp_path).unlink()

        return {
            "predictions": above_threshold,
            "threshold": threshold,
            "count": len(above_threshold),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

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
