from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert r.status_code == 200
    data = r.json()
    assert "label" in data and "score" in data
