"""
Unified feature extraction service for audio analysis.
Supports both hand-crafted features and spectrogram generation.
"""

import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FeatureService:
    """
    Service for extracting audio features.
    
    Supports:
    - hand-crafted features: 20-dimensional feature vector for GenreClassifier
    - spectrograms: 128x128 mel spectrogram for MultiLabelAudioCNN
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: Optional[int] = None
    ):
        """
        Initialize feature extraction service.
        
        Args:
            sample_rate: Target sample rate for audio loading
            n_mels: Number of mel bands for spectrogram
            n_fft: FFT window size
            hop_length: Hop length (defaults to n_fft // 4)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
    
    def load_audio(
        self,
        file_path: str,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with librosa.
        
        Args:
            file_path: Path to audio file
            duration: Duration in seconds (None for full file)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            logger.info(f"Loaded audio: {file_path} (shape: {y.shape}, sr: {sr})")
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def extract_hand_crafted_features(self, file_path: str) -> np.ndarray:
        """
        Extract 20-dimensional hand-crafted features for GenreClassifier.
        
        Features extracted:
        - Chroma STFT (mean)
        - RMS (mean)
        - Spectral Centroid (mean, var)
        - Spectral Bandwidth (mean, var)
        - Rolloff (mean, var)
        - Zero Crossing Rate (mean, var)
        - Harmony (mean, var)
        - Perceptr (mean, var)
        - Tempo
        - MFCCs 1-5 (mean)
        
        Args:
            file_path: Path to audio file
            
        Returns:
            20-dimensional feature vector
        """
        try:
            logger.info(f"Extracting hand-crafted features from {file_path}")
            
            # Load audio (full file for feature extraction)
            y, sr = self.load_audio(file_path)
            
            features = []
            
            # Chroma STFT
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            features.append(np.mean(chroma_stft))
            
            # RMS
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms))
            
            # Spectral Centroid
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spec_cent))
            features.append(np.var(spec_cent))
            
            # Spectral Bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(spec_bw))
            features.append(np.var(spec_bw))
            
            # Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(rolloff))
            features.append(np.var(rolloff))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr))
            features.append(np.var(zcr))
            
            # Harmony and Perceptr
            harmony, perceptr = librosa.effects.hpss(y)
            features.append(np.mean(harmony))
            features.append(np.var(harmony))
            features.append(np.mean(perceptr))
            features.append(np.var(perceptr))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # MFCCs (first 5 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
            for i in range(5):
                features.append(np.mean(mfcc[i]))
            
            feature_vector = np.array(features)
            logger.info(f"Extracted {len(feature_vector)} hand-crafted features")
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Failed to extract hand-crafted features: {e}")
            raise
    
    def extract_spectrogram(
        self,
        file_path: str,
        duration: int = 30,
        target_shape: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Extract mel spectrogram for CNN model.
        
        Args:
            file_path: Path to audio file
            duration: Duration in seconds
            target_shape: Target spectrogram shape (default 128x128)
            
        Returns:
            Mel spectrogram in dB scale with target shape
        """
        try:
            logger.info(f"Extracting spectrogram from {file_path}")
            
            # Load audio with fixed duration
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            
            # Pad or trim to exact duration
            target_length = self.sample_rate * duration
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to target shape if needed
            if mel_spec_db.shape != target_shape:
                mel_spec_db = self._resize_spectrogram(mel_spec_db, target_shape)
            
            logger.info(f"Generated spectrogram: {mel_spec_db.shape}")
            return mel_spec_db
            
        except Exception as e:
            logger.error(f"Failed to extract spectrogram: {e}")
            raise
    
    def _resize_spectrogram(
        self,
        spectrogram: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize spectrogram to target shape using interpolation.
        
        Args:
            spectrogram: Input spectrogram
            target_shape: Target (height, width)
            
        Returns:
            Resized spectrogram
        """
        from scipy.ndimage import zoom
        
        zoom_factors = (
            target_shape[0] / spectrogram.shape[0],
            target_shape[1] / spectrogram.shape[1]
        )
        
        resized = zoom(spectrogram, zoom_factors, order=1)
        logger.debug(f"Resized spectrogram from {spectrogram.shape} to {resized.shape}")
        
        return resized
    
    def get_audio_metadata(self, file_path: str) -> Dict:
        """
        Extract basic audio metadata.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with duration, sample_rate, channels
        """
        try:
            y, sr = self.load_audio(file_path)
            
            duration = len(y) / sr
            
            metadata = {
                'filename': Path(file_path).name,
                'duration': float(duration),
                'sample_rate': int(sr),
                'samples': len(y),
                'channels': 1  # librosa loads as mono
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise
    
    def validate_audio_file(
        self,
        file_path: str,
        max_duration: float = 600.0,
        supported_formats: Tuple[str, ...] = ('.mp3', '.wav', '.flac', '.ogg', '.au')
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate audio file format and duration.
        
        Args:
            file_path: Path to audio file
            max_duration: Maximum allowed duration in seconds
            supported_formats: Tuple of supported file extensions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        # Check format
        if not file_path.suffix.lower() in supported_formats:
            return False, f"Unsupported format: {file_path.suffix}. Supported: {supported_formats}"
        
        # Check duration
        try:
            metadata = self.get_audio_metadata(str(file_path))
            if metadata['duration'] > max_duration:
                return False, f"Audio too long: {metadata['duration']:.1f}s (max {max_duration}s)"
        except Exception as e:
            return False, f"Failed to validate audio: {str(e)}"
        
        return True, None


# Utility function for batch processing
def process_batch(
    file_paths: list,
    feature_type: str = "spectrogram",
    feature_service: Optional[FeatureService] = None
) -> list:
    """
    Process multiple audio files.
    
    Args:
        file_paths: List of audio file paths
        feature_type: "features" or "spectrogram"
        feature_service: FeatureService instance (creates new if None)
        
    Returns:
        List of results with file, success, and data/error
    """
    if feature_service is None:
        feature_service = FeatureService()
    
    results = []
    for file_path in file_paths:
        try:
            if feature_type == "features":
                data = feature_service.extract_hand_crafted_features(file_path)
            elif feature_type == "spectrogram":
                data = feature_service.extract_spectrogram(file_path)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            results.append({
                'file': file_path,
                'success': True,
                'data': data
            })
        except Exception as e:
            results.append({
                'file': file_path,
                'success': False,
                'error': str(e)
            })
    
    return results
