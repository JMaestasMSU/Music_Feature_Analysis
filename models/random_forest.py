"""Random Forest baseline trainer stub."""
from typing import Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_model(X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Tuple[Any, float]:
    """Train a RandomForestClassifier and return model and dummy score.

    This is a minimal implementation to verify wiring. Replace cross-val with
    stratified k-fold evaluation later.
    """
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    return model, float(score)
