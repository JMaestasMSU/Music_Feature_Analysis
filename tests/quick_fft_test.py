"""
Quick FFT Validation Test
Validates FFT-based feature extraction and Parseval's theorem.
Pure NumPy/SciPy implementation (no MATLAB required).
"""

import sys
import argparse
import numpy as np
from scipy import signal

def parse_args():
    parser = argparse.ArgumentParser(description='Quick FFT Test')
    parser.add_argument('--ci', action='store_true', help='CI mode')
    return parser.parse_args()

def test_parseval_theorem(verbose=True):
    """Test 1: Validate Parseval's theorem"""
    if verbose:
        print("Test 1: Parseval's theorem...", end=" ")
    
    try:
        # Generate test signal
        fs = 44100
        duration = 1
        t = np.arange(0, duration, 1/fs)
        test_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        
        # Time-domain energy
        time_energy = np.sum(test_signal**2)
        
        # Frequency-domain energy
        fft_mag = np.abs(np.fft.fft(test_signal))
        freq_energy = np.sum(fft_mag**2) / len(test_signal)
        
        # Calculate error
        error = abs(time_energy - freq_energy) / time_energy * 100
        
        assert error < 1, f"Parseval error too high: {error:.4f}%"
        
        if verbose:
            print(f"[OK] PASSED (error: {error:.6f}%)")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_spectral_centroid(verbose=True):
    """Test 2: Compute spectral centroid"""
    if verbose:
        print("Test 2: Spectral centroid...", end=" ")
    
    try:
        fs = 44100
        duration = 1
        t = np.arange(0, duration, 1/fs)
        
        # Low-frequency signal
        low_freq_signal = np.sin(2 * np.pi * 100 * t)
        fft_mag = np.abs(np.fft.fft(low_freq_signal))
        freqs = np.fft.fftfreq(len(low_freq_signal), 1/fs)
        
        # Positive frequencies only
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        fft_mag_pos = fft_mag[pos_mask]
        
        # Centroid
        centroid = np.sum(fft_mag_pos * freqs_pos) / np.sum(fft_mag_pos)
        
        # Should be close to 100 Hz
        assert 90 < centroid < 110, f"Expected ~100 Hz, got {centroid:.2f} Hz"
        
        if verbose:
            print(f"[OK] PASSED (centroid: {centroid:.2f} Hz)")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_spectral_rolloff(verbose=True):
    """Test 3: Compute spectral rolloff"""
    if verbose:
        print("Test 3: Spectral rolloff...", end=" ")
    
    try:
        fs = 44100
        duration = 1
        t = np.arange(0, duration, 1/fs)
        
        # Test signal
        test_signal = np.sin(2 * np.pi * 1000 * t)
        fft_mag = np.abs(np.fft.fft(test_signal))
        freqs = np.fft.fftfreq(len(test_signal), 1/fs)
        
        # Positive frequencies
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        fft_mag_pos = fft_mag[pos_mask]
        
        # Rolloff (95th percentile)
        cumsum_power = np.cumsum(fft_mag_pos)
        total_power = cumsum_power[-1]
        rolloff_idx = np.where(cumsum_power >= 0.95 * total_power)[0][0]
        rolloff_freq = freqs_pos[rolloff_idx]
        
        # Should be close to 1000 Hz for pure sine wave
        assert rolloff_freq > 900, f"Rolloff too low: {rolloff_freq:.2f} Hz"
        
        if verbose:
            print(f"[OK] PASSED (rolloff: {rolloff_freq:.2f} Hz)")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_windowing_effects(verbose=True):
    """Test 4: Compare windowing effects"""
    if verbose:
        print("Test 4: Windowing effects...", end=" ")
    
    try:
        fs = 44100
        duration = 1
        t = np.arange(0, duration, 1/fs)
        
        # Two sine waves
        test_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        
        # Rectangular window (no window)
        fft_rect = np.abs(np.fft.fft(test_signal))
        
        # Hann window
        window_hann = signal.windows.hann(len(test_signal))
        fft_hann = np.abs(np.fft.fft(test_signal * window_hann))
        
        # Both should have similar peak locations
        peaks_rect = signal.find_peaks(fft_rect[:len(fft_rect)//2], height=1000)[0]
        peaks_hann = signal.find_peaks(fft_hann[:len(fft_hann)//2], height=500)[0]
        
        assert len(peaks_rect) >= 2, "Not enough peaks in rectangular window"
        assert len(peaks_hann) >= 2, "Not enough peaks in Hann window"
        
        if verbose:
            print(f"[OK] PASSED (peaks found: rect={len(peaks_rect)}, hann={len(peaks_hann)})")
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
        print("Quick FFT Validation Test")
        print("=" * 60)
        print()
    
    # Run tests
    tests = [
        test_parseval_theorem,
        test_spectral_centroid,
        test_spectral_rolloff,
        test_windowing_effects
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
            print("\n[OK] All FFT tests passed!")
        sys.exit(0)
    else:
        if verbose:
            print(f"\n[FAIL] {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == '__main__':
    main()