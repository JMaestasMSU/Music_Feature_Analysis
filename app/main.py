from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from .config import settings
from .logging_config import get_logger

logger = get_logger(name=settings.app_name)

app = FastAPI(title="Music Feature Analysis - Inference")
# instrumentator for Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
except Exception:
    logger.info("prometheus instrumentator not available")


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
