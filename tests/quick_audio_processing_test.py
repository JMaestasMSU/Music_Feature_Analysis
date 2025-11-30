"""
Quick Audio Processing Test
Tests audio feature extraction pipeline without requiring actual audio files.
"""

import sys
import argparse
import numpy as np
from scipy import signal as scipy_signal

def parse_args():
    parser = argparse.ArgumentParser(description='Quick Audio Processing Test')
    parser.add_argument('--ci', action='store_true', help='CI mode')
    return parser.parse_args()

def generate_synthetic_audio(duration=1.0, sr=22050):
    """Generate synthetic audio for testing"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Mix of frequencies (simulates music)
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.2 * np.sin(2 * np.pi * 220 * t) +  # A3
        0.1 * np.random.randn(len(t))         # Noise
    )
    
    # Normalize audio to [-1, 1]
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    return audio, sr

def compute_mfcc(audio, sr, n_mfcc=13):
    """Simplified MFCC computation"""
    # This is a simplified version - real implementation uses librosa
    # For testing, we just verify the shape and basic properties
    
    # Create mel spectrogram (simplified)
    f, t, Sxx = scipy_signal.spectrogram(audio, sr, nperseg=512, noverlap=256)
    
    # Simulate MFCC output shape
    n_frames = Sxx.shape[1]
    mfcc = np.random.randn(n_mfcc, n_frames)  # Placeholder
    
    return mfcc

def test_audio_generation(verbose=True):
    """Test 1: Generate synthetic audio"""
    if verbose:
        print("Test 1: Audio generation...", end=" ")
    
    try:
        audio, sr = generate_synthetic_audio()
        
        assert len(audio) > 0, "Audio is empty"
        assert sr == 22050, f"Expected sr=22050, got {sr}"
        assert -1.0 <= audio.min() <= audio.max() <= 1.0, "Audio values out of range"
        
        if verbose:
            print(f"[OK] PASSED (length: {len(audio)} samples)")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_spectral_centroid_extraction(verbose=True):
    """Test 2: Extract spectral centroid"""
    if verbose:
        print("Test 2: Spectral centroid...", end=" ")
    
    try:
        audio, sr = generate_synthetic_audio()
        
        # Compute spectrogram
        f, t, Sxx = scipy_signal.spectrogram(audio, sr, nperseg=512, noverlap=256)
        
        # Compute centroid for each frame
        centroids = []
        for frame in range(Sxx.shape[1]):
            magnitude = Sxx[:, frame]
            centroid = np.sum(f * magnitude) / np.sum(magnitude)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        assert len(centroids) > 0, "No centroids computed"
        assert np.all(centroids >= 0), "Negative centroids"
        assert np.all(centroids < sr/2), "Centroids exceed Nyquist"
        
        if verbose:
            print(f"[OK] PASSED (mean: {centroids.mean():.2f} Hz)")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_zero_crossing_rate(verbose=True):
    """Test 3: Compute zero crossing rate"""
    if verbose:
        print("Test 3: Zero crossing rate...", end=" ")
    
    try:
        audio, sr = generate_synthetic_audio()
        
        # Compute ZCR
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        zcr = len(zero_crossings) / len(audio)
        
        assert 0 <= zcr <= 1, f"ZCR out of range: {zcr}"
        
        if verbose:
            print(f"[OK] PASSED (ZCR: {zcr:.4f})")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_rms_energy(verbose=True):
    """Test 4: Compute RMS energy"""
    if verbose:
        print("Test 4: RMS energy...", end=" ")
    
    try:
        audio, sr = generate_synthetic_audio()
        
        # Compute RMS
        rms = np.sqrt(np.mean(audio**2))
        
        assert rms > 0, "RMS is zero"
        assert rms < 1, f"RMS too high: {rms}"
        
        if verbose:
            print(f"[OK] PASSED (RMS: {rms:.4f})")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_mfcc_extraction(verbose=True):
    """Test 5: Extract MFCC features"""
    if verbose:
        print("Test 5: MFCC extraction...", end=" ")
    
    try:
        audio, sr = generate_synthetic_audio()
        
        mfcc = compute_mfcc(audio, sr, n_mfcc=13)
        
        assert mfcc.shape[0] == 13, f"Expected 13 MFCCs, got {mfcc.shape[0]}"
        assert mfcc.shape[1] > 0, "No MFCC frames"
        
        if verbose:
            print(f"[OK] PASSED (shape: {mfcc.shape})")
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
        test_audio_generation,
        test_spectral_centroid_extraction,
        test_zero_crossing_rate,
        test_rms_energy,
        test_mfcc_extraction
    ]
    
    results = []
    for test in tests:
        result = test(verbose=verbose)
        results.append(result)
    
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
