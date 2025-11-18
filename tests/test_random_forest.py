import numpy as np
from models.random_forest import train_model


def test_train_model_basic():
    # tiny synthetic dataset
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.9]])
    y = np.array([0, 1, 0])
    model, score = train_model(X, y)
    assert score >= 0.0
    assert hasattr(model, 'predict')
