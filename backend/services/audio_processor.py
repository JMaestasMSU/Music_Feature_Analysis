import numpy as np
import librosa
import librosa.feature
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Process audio files and extract features for ML analysis"""
    
    def __init__(self, sample_rate: int = 44100, n_mels: int = 128, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
    
    def load_audio(self, file_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            logger.info(f"Loaded audio: {file_path} (shape: {y.shape}, sr: {sr})")
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-spectrogram"""
        try:
            # Mel-spectrogram
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Log scale
            S_db = librosa.power_to_db(S, ref=np.max)
            
            logger.info(f"Spectrogram shape: {S_db.shape}")
            return S_db
        except Exception as e:
            logger.error(f"Failed to compute spectrogram: {e}")
            raise
    
    def compute_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute hand-crafted spectral features"""
        try:
            features = {}
            
            # Spectral centroid
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)))
            
            # Spectral rolloff
            features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)))
            
            # Zero crossing rate
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            
            # MFCC (first 13 coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = float(np.mean(mfcc))
            features['mfcc_std'] = float(np.std(mfcc))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features['chroma_mean'] = float(np.mean(chroma))
            
            # Temporal features
            features['rms_energy'] = float(np.mean(librosa.feature.rms(y=audio)))
            
            logger.info(f"Computed {len(features)} spectral features")
            return features
        except Exception as e:
            logger.error(f"Failed to compute spectral features: {e}")
            raise
    
    def normalize_spectrogram(self, spectrogram: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize spectrogram for CNN input"""
        try:
            if method == "minmax":
                min_val = spectrogram.min()
                max_val = spectrogram.max()
                normalized = (spectrogram - min_val) / (max_val - min_val + 1e-8)
            elif method == "zscore":
                mean_val = spectrogram.mean()
                std_val = spectrogram.std()
                normalized = (spectrogram - mean_val) / (std_val + 1e-8)
            else:
                normalized = spectrogram
            
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize spectrogram: {e}")
            raise
    
    def resize_spectrogram(self, spectrogram: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize spectrogram to CNN input shape"""
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_factors = (
                target_shape[0] / spectrogram.shape[0],
                target_shape[1] / spectrogram.shape[1]
            )
            
            resized = zoom(spectrogram, zoom_factors, order=1)
            logger.info(f"Resized spectrogram to {resized.shape}")
            return resized
        except Exception as e:
            logger.error(f"Failed to resize spectrogram: {e}")
            raise
    
    def process_audio_for_cnn(self, file_path: str, duration: int = 30) -> np.ndarray:
        """
        Process audio file for CNN model inference.
        Generates 128x128 mel spectrogram matching training parameters.

        Args:
            file_path: Path to audio file
            duration: Duration in seconds (default 30)

        Returns:
            128x128 mel spectrogram in dB scale
        """
        try:
            logger.info(f"Processing audio for CNN: {file_path}")

            # Load audio with fixed duration
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)

            # Pad or trim to exact duration
            target_length = self.sample_rate * duration
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            # Create mel spectrogram with exact training parameters
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Resize to 128x128 if needed
            if mel_spec_db.shape[1] != 128:
                from scipy.ndimage import zoom
                zoom_factor = 128 / mel_spec_db.shape[1]
                mel_spec_db = zoom(mel_spec_db, (1, zoom_factor))

            logger.info(f"Generated CNN spectrogram: {mel_spec_db.shape}")
            return mel_spec_db

        except Exception as e:
            logger.error(f"Failed to process audio for CNN: {e}")
            raise

    def process_audio_file(self, file_path: str, target_shape: Tuple[int, int] = (128, 216)) -> Dict:
        """End-to-end audio processing pipeline"""
        try:
            logger.info(f"Processing audio file: {file_path}")

            # Load audio
            audio, sr = self.load_audio(file_path)

            # Compute spectrogram
            spectrogram = self.compute_spectrogram(audio)

            # Compute spectral features
            spectral_features = self.compute_spectral_features(audio)

            # Normalize
            normalized_spec = self.normalize_spectrogram(spectrogram)

            # Resize to CNN input
            resized_spec = self.resize_spectrogram(normalized_spec, target_shape)

            return {
                'audio': audio,
                'sample_rate': sr,
                'spectrogram': spectrogram,
                'spectrogram_normalized': normalized_spec,
                'spectrogram_cnn': resized_spec,
                'spectral_features': spectral_features,
                'shape': resized_spec.shape
            }
        except Exception as e:
            logger.error(f"Audio processing pipeline failed: {e}")
            raise


# Utility function for batch processing
def process_batch(file_paths: list, processor: Optional[AudioProcessor] = None) -> list:
    """Process multiple audio files"""
    if processor is None:
        processor = AudioProcessor()
    
    results = []
    for file_path in file_paths:
        try:
            result = processor.process_audio_file(file_path)
            results.append({'file': file_path, 'success': True, 'data': result})
        except Exception as e:
            results.append({'file': file_path, 'success': False, 'error': str(e)})
    
    return results
