"""
Audio Data Augmentation for Music Classification

Provides various augmentation techniques to increase dataset diversity
and improve model generalization.
"""

import numpy as np
import torch
import librosa
from typing import Optional, Tuple


class SpectrogramAugmentation:
    """
    Augmentation techniques for spectrograms (time-frequency representations).

    Methods:
    - Time masking: Mask random time segments
    - Frequency masking: Mask random frequency bands
    - Time warping: Non-linear time stretching
    - Mixup: Blend two spectrograms together
    - SpecAugment: Combined time/freq masking (proven effective)
    """

    def __init__(self, sr=22050):
        self.sr = sr

    def time_mask(self, spectrogram: np.ndarray, max_mask_time: int = 20) -> np.ndarray:
        """
        Mask a random segment of time frames.

        Args:
            spectrogram: Input spectrogram (freq_bins, time_steps)
            max_mask_time: Maximum number of consecutive time steps to mask

        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.copy()
        _, time_steps = spec.shape

        mask_time = np.random.randint(0, min(max_mask_time, time_steps // 4))
        start = np.random.randint(0, time_steps - mask_time)

        spec[:, start:start + mask_time] = 0
        return spec

    def frequency_mask(self, spectrogram: np.ndarray, max_mask_freq: int = 20) -> np.ndarray:
        """
        Mask a random band of frequencies.

        Args:
            spectrogram: Input spectrogram (freq_bins, time_steps)
            max_mask_freq: Maximum number of consecutive frequency bins to mask

        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.copy()
        freq_bins, _ = spec.shape

        mask_freq = np.random.randint(0, min(max_mask_freq, freq_bins // 4))
        start = np.random.randint(0, freq_bins - mask_freq)

        spec[start:start + mask_freq, :] = 0
        return spec

    def spec_augment(self, spectrogram: np.ndarray,
                     num_time_masks: int = 2,
                     num_freq_masks: int = 2,
                     max_time_mask: int = 20,
                     max_freq_mask: int = 20) -> np.ndarray:
        """
        Apply SpecAugment (proven technique from Google).
        Combines multiple time and frequency masks.

        Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
                   for Automatic Speech Recognition", 2019
        """
        spec = spectrogram.copy()

        # Apply multiple time masks
        for _ in range(num_time_masks):
            spec = self.time_mask(spec, max_time_mask)

        # Apply multiple frequency masks
        for _ in range(num_freq_masks):
            spec = self.frequency_mask(spec, max_freq_mask)

        return spec

    def mixup(self, spec1: np.ndarray, spec2: np.ndarray,
              alpha: float = 0.2) -> Tuple[np.ndarray, float]:
        """
        Mixup augmentation: Blend two spectrograms.

        Args:
            spec1: First spectrogram
            spec2: Second spectrogram
            alpha: Beta distribution parameter (lower = less mixing)

        Returns:
            Mixed spectrogram, mixing coefficient lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        mixed = lam * spec1 + (1 - lam) * spec2
        return mixed, lam

    def add_noise(self, spectrogram: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add Gaussian noise to spectrogram."""
        noise = np.random.randn(*spectrogram.shape) * noise_factor
        return spectrogram + noise

    def time_shift(self, spectrogram: np.ndarray, shift_max: int = 20) -> np.ndarray:
        """
        Shift spectrogram in time (circular shift).

        Args:
            spectrogram: Input spectrogram
            shift_max: Maximum shift in time steps
        """
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(spectrogram, shift, axis=1)


class AudioAugmentation:
    """
    Augmentation techniques for raw audio waveforms.
    Apply BEFORE converting to spectrograms for more realistic augmentations.
    """

    def __init__(self, sr=22050):
        self.sr = sr

    def pitch_shift(self, audio: np.ndarray, n_steps: Optional[int] = None) -> np.ndarray:
        """
        Shift audio pitch without changing tempo.

        Args:
            audio: Audio waveform
            n_steps: Number of semitones to shift (random if None)
        """
        if n_steps is None:
            n_steps = np.random.randint(-2, 3)  # ±2 semitones

        if n_steps == 0:
            return audio

        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, audio: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """
        Change tempo without changing pitch.

        Args:
            audio: Audio waveform
            rate: Stretch factor (random if None, 1.0 = no change)
        """
        if rate is None:
            rate = np.random.uniform(0.9, 1.1)  # ±10% tempo change

        if rate == 1.0:
            return audio

        return librosa.effects.time_stretch(audio, rate=rate)

    def add_background_noise(self, audio: np.ndarray,
                            noise_factor: float = 0.005) -> np.ndarray:
        """Add white noise to audio."""
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise

    def dynamic_range_compression(self, audio: np.ndarray,
                                  threshold: float = 0.1) -> np.ndarray:
        """
        Apply dynamic range compression (reduces loud parts, boosts quiet parts).
        Simulates different recording conditions.
        """
        compressed = audio.copy()
        mask = np.abs(compressed) > threshold
        compressed[mask] = threshold + (compressed[mask] - threshold) * 0.5
        return compressed

    def random_gain(self, audio: np.ndarray,
                   min_gain: float = 0.8,
                   max_gain: float = 1.2) -> np.ndarray:
        """Apply random volume scaling."""
        gain = np.random.uniform(min_gain, max_gain)
        return audio * gain


class AudioDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset with on-the-fly augmentation support.

    Supports both single-label and multi-label classification.
    """

    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray,
                 augment: bool = True, multi_label: bool = False):
        """
        Args:
            spectrograms: Array of spectrograms (N, H, W)
            labels: Array of labels (N,) for single-label or (N, C) for multi-label
            augment: Apply augmentation during training
            multi_label: Whether this is multi-label classification
        """
        self.spectrograms = spectrograms
        self.labels = labels
        self.augment = augment
        self.multi_label = multi_label
        self.spec_aug = SpectrogramAugmentation()

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx].copy()
        label = self.labels[idx].copy()

        # Apply augmentation if enabled
        if self.augment:
            # Randomly apply augmentations (50% chance each)
            if np.random.rand() > 0.5:
                spec = self.spec_aug.spec_augment(spec)

            if np.random.rand() > 0.5:
                spec = self.spec_aug.add_noise(spec)

            if np.random.rand() > 0.5:
                spec = self.spec_aug.time_shift(spec)

        # Add channel dimension
        spec = spec[np.newaxis, :, :]

        # Convert to tensors
        spec_tensor = torch.FloatTensor(spec)

        if self.multi_label:
            label_tensor = torch.FloatTensor(label)
        else:
            label_tensor = torch.LongTensor([label]).squeeze()

        return spec_tensor, label_tensor


def create_dataloaders(spectrograms: np.ndarray,
                      labels: np.ndarray,
                      train_idx: np.ndarray,
                      val_idx: np.ndarray,
                      test_idx: np.ndarray,
                      batch_size: int = 32,
                      multi_label: bool = False,
                      num_workers: int = 0) -> Tuple:
    """
    Create PyTorch DataLoaders with augmentation support.

    Args:
        spectrograms: All spectrograms
        labels: All labels
        train_idx, val_idx, test_idx: Split indices
        batch_size: Batch size
        multi_label: Multi-label classification
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets (augment only training)
    train_dataset = AudioDataset(
        spectrograms[train_idx],
        labels[train_idx],
        augment=True,
        multi_label=multi_label
    )

    val_dataset = AudioDataset(
        spectrograms[val_idx],
        labels[val_idx],
        augment=False,
        multi_label=multi_label
    )

    test_dataset = AudioDataset(
        spectrograms[test_idx],
        labels[test_idx],
        augment=False,
        multi_label=multi_label
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
