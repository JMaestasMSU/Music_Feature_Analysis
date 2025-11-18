import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional
import logging
import numpy as np
from scipy.io import savemat, loadmat

logger = logging.getLogger(__name__)

class MATLABInterface:
    """Interface with MATLAB for spectral analysis validation"""
    
    def __init__(self, matlab_path: str = "matlab", matlab_dir: Optional[Path] = None):
        self.matlab_path = matlab_path
        self.matlab_dir = matlab_dir or Path(__file__).parent.parent.parent / "matlab"
        self.timeout = 60
    
    def validate_matlab_available(self) -> bool:
        """Check if MATLAB is available in system"""
        try:
            result = subprocess.run([self.matlab_path, "-version"], capture_output=True, timeout=5)
            is_available = result.returncode == 0
            logger.info(f"MATLAB availability: {is_available}")
            return is_available
        except Exception as e:
            logger.warning(f"MATLAB not available: {e}")
            return False
    
    def run_fft_analysis(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Run FFT spectral analysis via MATLAB
        
        Returns:
            Dictionary with FFT results (spectral centroid, rolloff, spread, etc.)
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Save audio data
                audio_file = tmpdir_path / "audio_data.mat"
                savemat(str(audio_file), {'audio': audio, 'fs': sample_rate})
                
                # Create MATLAB script
                script_file = tmpdir_path / "run_fft_analysis.m"
                output_file = tmpdir_path / "fft_results.mat"
                
                script_content = f"""
cd('{self.matlab_dir}');
addpath(genpath('{self.matlab_dir}'));

% Load audio
load('{audio_file}');

% Compute FFT
N = length(audio);
fft_mag = abs(fft(audio));
freq = (0:N-1) * fs / N;
positive_freq_idx = 1:floor(N/2);
freq_positive = freq(positive_freq_idx);
fft_mag_pos = fft_mag(positive_freq_idx);

% Spectral centroid
centroid = sum(fft_mag_pos .* freq_positive) / sum(fft_mag_pos);

% Spectral spread
spread = sqrt(sum(fft_mag_pos .* (freq_positive - centroid).^2) / sum(fft_mag_pos));

% Spectral rolloff (95th percentile)
cumsum_power = cumsum(fft_mag_pos);
total_power = cumsum_power(end);
rolloff_idx = find(cumsum_power >= 0.95 * total_power, 1);
if isempty(rolloff_idx)
    rolloff = freq_positive(end);
else
    rolloff = freq_positive(rolloff_idx);
end

% Parseval's theorem validation
time_energy = sum(audio.^2) / length(audio);
freq_energy = sum(fft_mag.^2) / (length(audio)^2);

% Save results
save('{output_file}', 'centroid', 'spread', 'rolloff', 'time_energy', 'freq_energy', 'fft_mag_pos', 'freq_positive');
exit;
"""
                
                with open(script_file, 'w') as f:
                    f.write(script_content)
                
                # Run MATLAB
                logger.info("Running FFT analysis via MATLAB...")
                result = subprocess.run(
                    [self.matlab_path, "-batch", f"run '{script_file}'"],
                    capture_output=True,
                    timeout=self.timeout,
                    cwd=str(tmpdir_path)
                )
                
                if result.returncode != 0:
                    logger.warning(f"MATLAB error: {result.stderr.decode()}")
                    return self._fallback_fft_analysis(audio, sample_rate)
                
                # Load results
                if output_file.exists():
                    results = loadmat(str(output_file))
                    
                    fft_results = {
                        'spectral_centroid': float(results['centroid'][0, 0]),
                        'spectral_spread': float(results['spread'][0, 0]),
                        'spectral_rolloff': float(results['rolloff'][0, 0]),
                        'time_domain_energy': float(results['time_energy'][0, 0]),
                        'freq_domain_energy': float(results['freq_energy'][0, 0]),
                        'source': 'MATLAB'
                    }
                    
                    logger.info(f"FFT analysis complete: {fft_results}")
                    return fft_results
        
        except subprocess.TimeoutExpired:
            logger.warning("MATLAB FFT analysis timed out, using fallback")
            return self._fallback_fft_analysis(audio, sample_rate)
        except Exception as e:
            logger.warning(f"MATLAB FFT analysis failed: {e}, using fallback")
            return self._fallback_fft_analysis(audio, sample_rate)
    
    def _fallback_fft_analysis(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Fallback FFT analysis using NumPy (when MATLAB unavailable)"""
        try:
            logger.info("Using fallback NumPy FFT analysis")
            
            N = len(audio)
            fft_mag = np.abs(np.fft.fft(audio))
            freq = np.arange(N) * sample_rate / N
            positive_freq_idx = slice(0, N // 2)
            freq_positive = freq[positive_freq_idx]
            fft_mag_pos = fft_mag[positive_freq_idx]
            
            # Spectral centroid
            centroid = np.sum(fft_mag_pos * freq_positive) / np.sum(fft_mag_pos)
            
            # Spectral spread
            spread = np.sqrt(np.sum(fft_mag_pos * (freq_positive - centroid)**2) / np.sum(fft_mag_pos))
            
            # Spectral rolloff
            cumsum_power = np.cumsum(fft_mag_pos)
            rolloff_idx = np.where(cumsum_power >= 0.95 * cumsum_power[-1])[0]
            rolloff = freq_positive[rolloff_idx[0]] if len(rolloff_idx) > 0 else freq_positive[-1]
            
            # Parseval's theorem
            time_energy = np.sum(audio**2) / len(audio)
            freq_energy = np.sum(fft_mag**2) / len(audio)
            
            return {
                'spectral_centroid': float(centroid),
                'spectral_spread': float(spread),
                'spectral_rolloff': float(rolloff),
                'time_domain_energy': float(time_energy),
                'freq_domain_energy': float(freq_energy),
                'source': 'NumPy (fallback)'
            }
        except Exception as e:
            logger.error(f"Fallback FFT analysis failed: {e}")
            raise
    
    def validate_fft_vs_ml_features(self, fft_results: Dict, ml_features: Dict) -> Dict:
        """Compare FFT-based features with ML-extracted features"""
        try:
            comparison = {
                'fft': fft_results,
                'ml': ml_features,
                'correlation': {}
            }
            
            # Simple correlation check for common features
            if 'spectral_centroid' in fft_results and 'spectral_centroid' in ml_features:
                fft_centroid = fft_results['spectral_centroid']
                ml_centroid = ml_features.get('spectral_centroid', fft_centroid)
                
                # Percentage difference
                pct_diff = abs(fft_centroid - ml_centroid) / max(abs(fft_centroid), 1e-8) * 100
                comparison['correlation']['centroid_pct_diff'] = float(pct_diff)
            
            return comparison
        except Exception as e:
            logger.error(f"FFT vs ML validation failed: {e}")
            raise
