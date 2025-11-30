"""
Quick API Integration Test
Proves FastAPI backend + Model Server communication works
Requires both services running (docker-compose up)
"""

import requests
import json
import time

print("\n" + "="*70)
print("QUICK API TEST - Backend Integration")
print("="*70)

BACKEND_URL = "http://localhost:8000"
MODEL_SERVER_URL = "http://localhost:8001"

# Test 1: Backend health
print("\n1. Testing Backend Health Check...")
try:
    response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=2)
    if response.status_code == 200:
        print(f"   Backend healthy: {response.json()}")
    else:
        print(f"   Backend error: {response.status_code}")
except Exception as e:
    print(f"   Backend not running: {e}")

# Test 2: Model Server health
print("\n2. Testing Model Server Health Check...")
try:
    response = requests.get(f"{MODEL_SERVER_URL}/health", timeout=2)
    if response.status_code == 200:
        print(f"   Model server healthy: {response.json()}")
    else:
        print(f"   Model server error: {response.status_code}")
except Exception as e:
    print(f"   Model server not running: {e}")

# Test 3: API endpoints
print("\n3. Testing API Endpoints...")
endpoints = [
    ("GET", "/", "API Info"),
    ("GET", "/api/v1/analysis/features", "Feature List"),
    ("GET", "/api/v1/analysis/fft-validation", "FFT Info"),
]

for method, endpoint, name in endpoints:
    try:
        if method == "GET":
            response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=2)
        
        if response.status_code == 200:
            print(f"   {name}: OK")
        else:
            print(f"   {name}: {response.status_code}")
    except Exception as e:
        print(f"   {name}: {e}")

# Test 4: Model prediction endpoint (if available)
print("\n4. Testing Model Prediction...")
try:
    # Synthetic spectrogram
    import numpy as np
    spec = np.random.randn(128, 216).tolist()
    
    payload = {
        "spectrogram": spec,
        "metadata": {}
    }
    
    response = requests.post(
        f"{MODEL_SERVER_URL}/predict",
        json=payload,
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Prediction: {result['predicted_genre']}")
        print(f"     Confidence: {result['confidence']:.2%}")
    else:
        print(f"   Prediction error: {response.status_code}")
except Exception as e:
    print(f"   Model server not ready: {e}")

print("\n" + "="*70)
print(" API TEST COMPLETE")
print("="*70 + "\n")
