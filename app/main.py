"""
Model Inference Server - FastAPI Application
Simplified version for inference only.
"""

import sys
from pathlib import Path
from typing import List, Dict
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from app.model import get_model_service

from .config import settings
from .logging_config import get_logger
from preprocessing.feature_extraction import extract_features, features_to_array
import tempfile
import os

logger = get_logger(name=settings.app_name)

app = FastAPI(
    title="Music Genre Classification Inference Server",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    label: int
    score: float


@app.on_event("startup")
def on_startup():
    logger.info("starting application", extra={"app": settings.app_name})
    # load model
    from .model import ModelLoader

    loader = ModelLoader(path=settings.model_path, backend=settings.model_backend)
    loader.load()
    app.state.model_loader = loader


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.debug("predict called", extra={"features_len": len(req.features)})
    if not req.features:
        logger.warning("empty features in predict")
        raise HTTPException(status_code=400, detail="No features provided")
    loader = getattr(app.state, "model_loader", None)
    if loader is None:
        resp_data = {"label": 0, "score": 0.5}
    else:
        resp_data = loader.predict(req.features)
    # Ensure types are explicit and keep line length under limits
    resp = PredictResponse(
        label=int(resp_data["label"]),
        score=float(resp_data["score"]),
    )
    logger.info("prediction", extra={"label": resp.label, "score": resp.score})
    return resp


@app.post("/admin/load-model")
def admin_load_model(token: str):
    # simple token auth (replace with more secure auth in prod)
    from .model import ModelLoader

    admin_token = getattr(settings, "admin_token", None)
    if not admin_token or token != admin_token:
        raise HTTPException(status_code=403, detail="forbidden")
    loader = ModelLoader(path=settings.model_path, backend=settings.model_backend)
    loader.load()
    app.state.model_loader = loader
    return {"status": "reloaded"}


@app.post("/api/v1/predict")
async def predict_genre(audio_file: UploadFile):
    """Predict genre from uploaded audio file."""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await audio_file.read())
        tmp_path = tmp.name

    try:
        # Extract features
        features = extract_features(tmp_path)
        feature_vector = features_to_array(features)

        # Predict with model
        loader = getattr(app.state, "model_loader", None)
        if loader is None:
            prediction = {"predicted_genre": "unknown", "confidence": 0.0}
        else:
            prediction = loader.predict(feature_vector)

        return {
            "genre": prediction["predicted_genre"],
            "confidence": prediction["confidence"],
            "features": features,  # Optional: return extracted features
        }

    finally:
        os.unlink(tmp_path)  # Clean up temp file


@app.get("/health")
async def health():
    """Health check endpoint."""
    service = get_model_service()
    info = service.get_model_info()
    
    return {
        "status": "healthy" if service.is_loaded else "unhealthy",
        "model_loaded": service.is_loaded,
        "device": info.get('device', 'unknown')
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
