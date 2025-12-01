"""
FFT Validation (Python Alternative to MATLAB)
Validates FFT-based feature extraction and Parseval's theorem.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def validate_parsevals_theorem():
    """Test Parseval's theorem: time-domain energy == frequency-domain energy."""
    print("=" * 70)
    print("FFT VALIDATION - Parseval's Theorem")
    print("=" * 70)
    
    # Generate test signal
    fs = 44100  # Sample rate
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
    
    print(f"\nTime-domain energy:   {time_energy:.6f}")
    print(f"Frequency-domain energy: {freq_energy:.6f}")
    print(f"Error:               {error:.6f}%")
    
    if error < 1:
        print("Result: PASSED (error < 1%)")
    else:
        print("Result: FAILED (error >= 1%)")
    
    return error < 1


def validate_spectral_centroid():
    """Validate spectral centroid computation."""
    print("\n" + "=" * 70)
    print("SPECTRAL CENTROID VALIDATION")
    print("=" * 70)
    
    fs = 44100
    duration = 1
    t = np.arange(0, duration, 1/fs)
    
    # Low-frequency signal (100 Hz)
    low_freq_signal = np.sin(2 * np.pi * 100 * t)
    
    # Compute FFT
    fft_mag = np.abs(np.fft.fft(low_freq_signal))
    freqs = np.fft.fftfreq(len(low_freq_signal), 1/fs)
    
    # Positive frequencies only
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    fft_mag_pos = fft_mag[pos_mask]
    
    # Compute spectral centroid
    centroid = np.sum(fft_mag_pos * freqs_pos) / np.sum(fft_mag_pos)
    
    print(f"\nExpected centroid: ~100 Hz")
    print(f"Computed centroid: {centroid:.2f} Hz")
    
    passed = 90 <= centroid <= 110
    if passed:
        print("Result: PASSED (within ±10 Hz)")
    else:
        print("Result: FAILED (outside expected range)")
    
    return passed


def visualize_fft_analysis():
    """Create FFT analysis visualizations."""
    print("\n" + "=" * 70)
    print("FFT ANALYSIS VISUALIZATION")
    print("=" * 70)
    
    fs = 44100
    duration = 1
    t = np.arange(0, duration, 1/fs)
    
    # Test signal with multiple frequencies
    test_signal = (
        np.sin(2 * np.pi * 440 * t) +
        0.5 * np.sin(2 * np.pi * 880 * t) +
        0.3 * np.sin(2 * np.pi * 1320 * t)
    )
    
    # Compute FFT
    fft_result = np.fft.fft(test_signal)
    fft_mag = np.abs(fft_result)
    fft_freq = np.fft.fftfreq(len(test_signal), 1/fs)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time domain
    axes[0].plot(t[:1000], test_signal[:1000])
    axes[0].set_title('Time Domain Signal', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Frequency domain (positive frequencies only)
    pos_mask = fft_freq >= 0
    axes[1].plot(fft_freq[pos_mask][:5000], fft_mag[pos_mask][:5000])
    axes[1].set_title('Frequency Domain (FFT)', fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 2000])
    
    plt.tight_layout()
    plt.savefig('fft_validation_plot.png', dpi=300)
    print("\n✓ Visualization saved: fft_validation_plot.png")
    plt.show()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("RUNNING FFT VALIDATION SUITE")
    print("=" * 70)
    
    results = []
    
    # Run validations
    results.append(validate_parsevals_theorem())
    results.append(validate_spectral_centroid())
    
    # Create visualization
    visualize_fft_analysis()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All validations passed!")
    else:
        print(f"\n✗ {total - passed} validation(s) failed")
