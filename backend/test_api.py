"""
API Testing Script for Music Genre Classification Backend
Tests all endpoints with synthetic data.
"""

import requests
import json
import numpy as np
from pathlib import Path
import sys

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test health check endpoint."""
    print("\n[1/6] Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Health check passed")
            print(f"    Status: {data.get('status')}")
            print(f"    Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"  ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_predict_endpoint():
    """Test prediction endpoint."""
    print("\n[2/6] Testing predict endpoint...")
    
    try:
        # Generate synthetic features
        features = np.random.randn(20).tolist()
        
        payload = {
            "features": features,
            "return_probs": True,
            "top_k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Prediction successful")
            print(f"    Predicted genre: {data.get('predicted_genre')}")
            print(f"    Confidence: {data.get('confidence'):.4f}")
            print(f"    Top 3 predictions: {[p['genre'] for p in data.get('top_predictions', [])]}")
            return True
        else:
            print(f"  ✗ Prediction failed: {response.status_code}")
            print(f"    Response: {response.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_batch_predict_endpoint():
    """Test batch prediction endpoint."""
    print("\n[3/6] Testing batch predict endpoint...")
    
    try:
        # Generate batch of synthetic features
        batch_size = 5
        requests_data = [
            {
                "features": np.random.randn(20).tolist(),
                "return_probs": False,
                "top_k": 1
            }
            for _ in range(batch_size)
        ]
        
        response = requests.post(
            f"{BASE_URL}/batch-predict",
            json=requests_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Batch prediction successful")
            print(f"    Processed {len(data)} samples")
            print(f"    Predictions: {[d.get('predicted_genre') for d in data]}")
            return True
        else:
            print(f"  ✗ Batch prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_genres_endpoint():
    """Test get genres endpoint."""
    print("\n[4/6] Testing genres endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/genres")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Genres retrieved successfully")
            print(f"    Total genres: {data.get('count')}")
            print(f"    Genres: {data.get('genres')}")
            return True
        else:
            print(f"  ✗ Get genres failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_model_info_endpoint():
    """Test model info endpoint."""
    print("\n[5/6] Testing model info endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Model info retrieved successfully")
            print(f"    Status: {data.get('status')}")
            print(f"    Device: {data.get('device')}")
            print(f"    Num classes: {data.get('num_classes')}")
            return True
        else:
            print(f"  ✗ Get model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid input."""
    print("\n[6/6] Testing error handling...")
    
    try:
        # Send invalid payload (wrong number of features)
        payload = {
            "features": [1, 2, 3],  # Only 3 features instead of 20
            "return_probs": False,
            "top_k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload
        )
        
        if response.status_code in [400, 422, 500]:
            print(f"  ✓ Error handling works correctly")
            print(f"    Status code: {response.status_code}")
            print(f"    Error message: {response.json().get('detail', 'Unknown error')}")
            return True
        else:
            print(f"  ✗ Expected error response, got: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all API tests."""
    print("=" * 60)
    print("API TESTING - Music Genre Classification Backend")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}")
    print("Make sure the API server is running: python backend/app.py")
    print()
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_predict_endpoint,
        test_batch_predict_endpoint,
        test_genres_endpoint,
        test_model_info_endpoint,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All API tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
