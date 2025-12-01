"""
Quick Audio Processing Test
Tests audio feature extraction pipeline with synthetic audio.
No real audio files required.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.feature_extraction import extract_features, features_to_array


def parse_args():
    parser = argparse.ArgumentParser(description='Quick Audio Processing Test')
    parser.add_argument('--ci', action='store_true', help='CI mode')
    return parser.parse_args()


def generate_synthetic_audio(duration=1.0, sr=22050):
    """Generate synthetic audio signal."""
    t = np.arange(0, duration, 1/sr)
    
    # Mix of sine waves (simulate music)
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 1320 * t)   # E6 note
    )
    
    # Add some noise
    noise = np.random.randn(len(signal)) * 0.05
    signal = signal + noise
    
    return signal, sr


def test_feature_extraction(verbose=True):
    """Test 1: Can extract features from synthetic audio."""
    if verbose:
        print("Test 1: Feature extraction from synthetic audio...", end=" ")
    
    try:
        # Generate synthetic audio
        audio, sr = generate_synthetic_audio(duration=2.0)
        
        # Save to temporary file
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name
        
        # Extract features
        features = extract_features(tmp_path, sr=sr, duration=2.0)
        
        # Validate features
        assert 'mfcc' in features, "Missing MFCC features"
        assert 'spectral_centroid' in features, "Missing spectral centroid"
        assert 'spectral_rolloff' in features, "Missing spectral rolloff"
        assert 'zcr' in features, "Missing zero crossing rate"
        assert 'chroma' in features, "Missing chroma features"
        assert 'rms_energy' in features, "Missing RMS energy"
        
        # Check dimensions
        assert len(features['mfcc']) == 13, f"Expected 13 MFCCs, got {len(features['mfcc'])}"
        assert len(features['chroma']) == 12, f"Expected 12 chroma bins, got {len(features['chroma'])}"
        
        # Clean up
        import os
        os.unlink(tmp_path)
        
        if verbose:
            print("[OK] PASSED")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False


def test_features_to_array(verbose=True):
    """Test 2: Can convert features to array."""
    if verbose:
        print("Test 2: Feature conversion to array...", end=" ")
    
    try:
        # Create dummy features
        features = {
            'mfcc': np.random.randn(13),
            'spectral_centroid': 2000.0,
            'spectral_rolloff': 7000.0,
            'zcr': 0.1,
            'rms_energy': 0.05,
            'chroma': np.random.randn(12)
        }
        
        # Convert to array
        feature_array = features_to_array(features)
        
        # Validate
        expected_length = 13 + 1 + 1 + 1 + 1 + 12  # 29 features
        assert len(feature_array) == expected_length, \
            f"Expected {expected_length} features, got {len(feature_array)}"
        
        assert feature_array.ndim == 1, "Feature array should be 1D"
        
        if verbose:
            print(f"[OK] PASSED (vector length: {len(feature_array)})")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False


def test_batch_processing(verbose=True):
    """Test 3: Can process batch of audio files."""
    if verbose:
        print("Test 3: Batch processing...", end=" ")
    
    try:
        import tempfile
        import soundfile as sf
        
        # Generate multiple audio files
        temp_files = []
        for i in range(3):
            audio, sr = generate_synthetic_audio(duration=1.0)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, sr)
                temp_files.append(tmp.name)
        
        # Batch extract (simulated - call extract_features for each)
        features_list = []
        for path in temp_files:
            features = extract_features(path, sr=sr, duration=1.0)
            features_list.append(features)
        
        # Validate
        assert len(features_list) == 3, "Should have extracted features from 3 files"
        
        for features in features_list:
            assert 'mfcc' in features
            assert 'spectral_centroid' in features
        
        # Clean up
        import os
        for path in temp_files:
            os.unlink(path)
        
        if verbose:
            print(f"[OK] PASSED (processed {len(features_list)} files)")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False


def main():
    args = parse_args()
    verbose = not args.ci
    
    if verbose:
        print("=" * 60)
        print("Quick Audio Processing Test")
        print("=" * 60)
        print()
    
    # Run tests
    tests = [
        test_feature_extraction,
        test_features_to_array,
        test_batch_processing
    ]
    
    results = []
    for test in tests:
        try:
            result = test(verbose=verbose)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"[EXCEPTION] {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    if verbose:
        print()
        print("=" * 60)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 60)
    
    if passed == total:
        if verbose:
            print("\n[OK] All audio processing tests passed!")
        sys.exit(0)
    else:
        if verbose:
            print(f"\n[FAIL] {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
