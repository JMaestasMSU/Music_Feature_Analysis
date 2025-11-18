"""
Audio feature extraction for music genre classification.
Uses librosa for real audio processing.
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional


# Provides reusable functions:
def extract_features(
    audio_path: str,
    sr: int = 22050,
    duration: Optional[float] = 30.0
) -> Dict[str, Any]:
    """
    Extract comprehensive audio features from file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate for audio loading
        duration: Duration to load (seconds), None for full file
    
    Returns:
        Dictionary of extracted features:
        - mfcc: MFCC coefficients (13,)
        - spectral_centroid: Mean spectral centroid
        - spectral_rolloff: Mean spectral rolloff
        - zcr: Mean zero crossing rate
        - chroma: Mean chroma features (12,)
        - rms_energy: Mean RMS energy
    """
    # Load audio
    y, actual_sr = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Extract features
    features = {}
    
    # MFCCs (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=actual_sr, n_mfcc=13)
    features['mfcc'] = np.mean(mfcc, axis=1)  # Average over time
    
    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr)
    features['spectral_centroid'] = float(np.mean(centroid))
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=actual_sr)
    features['spectral_rolloff'] = float(np.mean(rolloff))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr'] = float(np.mean(zcr))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=actual_sr)
    features['chroma'] = np.mean(chroma, axis=1)  # Average over time
    
    # RMS energy
    rms = librosa.feature.rms(y=y)
    features['rms_energy'] = float(np.mean(rms))
    
    return features


def extract_features_batch(
    audio_paths: list[str],
    sr: int = 22050,
    duration: Optional[float] = 30.0,
    verbose: bool = True
) -> list[Dict[str, Any]]:
    """
    Extract features from multiple audio files.
    
    Args:
        audio_paths: List of paths to audio files
        sr: Sample rate
        duration: Duration to load per file
        verbose: Print progress
    
    Returns:
        List of feature dictionaries
    """
    features_list = []
    
    for i, path in enumerate(audio_paths):
        try:
            features = extract_features(path, sr=sr, duration=duration)
            features_list.append(features)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(audio_paths)} files")
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            features_list.append(None)
    
    return features_list


def features_to_array(features: Dict[str, Any]) -> np.ndarray:
    """
    Convert feature dictionary to flat numpy array for ML models.
    
    Args:
        features: Dictionary from extract_features()
    
    Returns:
        1D numpy array of all features concatenated
    """
    feature_vector = []
    
    # MFCCs (13 values)
    feature_vector.extend(features['mfcc'])
    
    # Scalar features
    feature_vector.append(features['spectral_centroid'])
    feature_vector.append(features['spectral_rolloff'])
    feature_vector.append(features['zcr'])
    feature_vector.append(features['rms_energy'])
    
    # Chroma (12 values)
    feature_vector.extend(features['chroma'])
    
    return np.array(feature_vector)
