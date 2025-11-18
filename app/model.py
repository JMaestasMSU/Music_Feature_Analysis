from typing import Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, path: str = "", backend: str = "sklearn"):
        self.path = path
        self.backend = backend
        self.model: Optional[Any] = None

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            logger.warning(
                "model path not provided or does not exist, using dummy model"
                )
            self.model = None
            return
        try:
            if self.backend == "sklearn":
                import joblib

                self.model = joblib.load(self.path)
            elif self.backend == "torch":
                import torch

                # assume a torch.save-ed module or state_dict
                self.model = torch.load(self.path, map_location="cpu")
            else:
                logger.error("unknown backend %s", self.backend)
                self.model = None
        except Exception as e:
            logger.exception("failed to load model: %s", e)
            self.model = None

    def predict(self, features) -> Any:
        if self.model is None:
            # dummy prediction
            return {"label": 0, "score": 0.5}
        if self.backend == "sklearn":
            y = self.model.predict([features])
            proba = None
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba([features]).max()
            return {
                "label": int(y[0]),
                "score": float(proba) if proba is not None else 1.0
                }
        if self.backend == "torch":
            import torch

            self.model.eval()
            with torch.no_grad():
                x = torch.tensor([features], dtype=torch.float32)
                out = self.model(x)
                if isinstance(out, torch.Tensor):
                    pred = int(out.argmax(dim=-1).item())
                    score = float(out.softmax(dim=-1).max().item())
                    return {"label": pred, "score": score}
                return {"label": 0, "score": 0.5}
