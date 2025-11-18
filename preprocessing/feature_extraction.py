"""Minimal feature extraction stubs used by the project.

These functions are intentionally lightweight so importing the package doesn't
require heavy optional dependencies at test time.
"""
from typing import Dict, Any
import numpy as np


def extract_features(path: str) -> Dict[str, Any]:
    """Extract a minimal set of placeholder features from an audio file path.

    This is a stub used for scaffolding and tests. Replace with librosa-based
    extraction in development.
    """
    # Placeholder deterministic features for testing
    dummy_mfcc = np.zeros((13,))
    spectral_centroid = 0.0
    zcr = 0.0
    chroma = np.zeros((12,))

    return {
        "mfcc": dummy_mfcc,
        "spectral_centroid": spectral_centroid,
        "zcr": zcr,
        "chroma": chroma,
    }
