#!/usr/bin/env python3
"""
Test Data Augmentation

Tests SpecAugment, Mixup, and other augmentation techniques.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from models.audio_augmentation import (
    SpectrogramAugmentation,
    AudioDataset,
    create_dataloaders
)


def test_spec_augment():
    """Test 1: SpecAugment masks time/frequency"""
    print("\n" + "="*70)
    print("TEST 1: SpecAugment")
    print("="*70)

    aug = SpectrogramAugmentation()
    spec = np.random.randn(128, 128) + 5.0  # Non-zero spectrogram

    # Apply SpecAugment
    augmented = aug.spec_augment(spec, num_time_masks=2, num_freq_masks=2)

    # Check that masking happened (should have zeros)
    has_zeros = np.any(augmented == 0)
    original_energy = np.sum(spec**2)
    augmented_energy = np.sum(augmented**2)

    print(f"  Original energy:  {original_energy:.2f}")
    print(f"  Augmented energy: {augmented_energy:.2f}")
    print(f"  Has masked regions: {has_zeros}")

    if has_zeros and augmented_energy < original_energy:
        print("  PASSED: SpecAugment masks correctly")
        return True
    else:
        print("  FAILED: SpecAugment didn't mask")
        return False


def test_mixup():
    """Test 2: Mixup blends two spectrograms"""
    print("\n" + "="*70)
    print("TEST 2: Mixup")
    print("="*70)

    aug = SpectrogramAugmentation()

    spec1 = np.ones((128, 128)) * 10.0
    spec2 = np.ones((128, 128)) * 20.0

    mixed, lambda_val = aug.mixup(spec1, spec2, alpha=0.2)

    expected_value = lambda_val * 10.0 + (1 - lambda_val) * 20.0
    actual_value = mixed.mean()

    print(f"  Lambda: {lambda_val:.3f}")
    print(f"  Expected mean: {expected_value:.2f}")
    print(f"  Actual mean:   {actual_value:.2f}")
    print(f"  Difference:    {abs(expected_value - actual_value):.4f}")

    if abs(expected_value - actual_value) < 0.1:
        print("  PASSED: Mixup blends correctly")
        return True
    else:
        print("  FAILED: Mixup blending incorrect")
        return False


def test_time_frequency_masking():
    """Test 3: Time and frequency masks are different"""
    print("\n" + "="*70)
    print("TEST 3: Time vs Frequency Masking")
    print("="*70)

    aug = SpectrogramAugmentation()
    spec = np.ones((128, 256))  # Use 256 time steps for more reliable masking

    # Apply multiple times to ensure we get non-zero masks
    time_masked = spec.copy()
    freq_masked = spec.copy()

    # Apply masks until we get actual masking (avoid zero-length masks)
    for _ in range(10):
        time_masked = aug.time_mask(time_masked, max_mask_time=30)
        freq_masked = aug.frequency_mask(freq_masked, max_mask_freq=30)

    # Check that masking happened
    time_has_zeros = np.any(time_masked == 0)
    freq_has_zeros = np.any(freq_masked == 0)

    # Check the pattern of masking
    # Time masking should create vertical stripes (all frequencies affected at certain times)
    # Frequency masking should create horizontal stripes (all times affected at certain frequencies)

    # For time mask: check if any column is fully masked
    time_fully_masked_cols = np.any((time_masked == 0).all(axis=0))

    # For freq mask: check if any row is fully masked
    freq_fully_masked_rows = np.any((freq_masked == 0).all(axis=1))

    print(f"  Time mask has zeros: {time_has_zeros}")
    print(f"  Freq mask has zeros: {freq_has_zeros}")
    print(f"  Time mask creates full columns: {time_fully_masked_cols}")
    print(f"  Freq mask creates full rows:    {freq_fully_masked_rows}")

    if time_has_zeros and freq_has_zeros and time_fully_masked_cols and freq_fully_masked_rows:
        print("  PASSED: Masking dimensions correct")
        return True
    else:
        print("  FAILED: Masking dimensions incorrect")
        return False


def test_augmentation_preserves_shape():
    """Test 4: Augmentations preserve spectrogram shape"""
    print("\n" + "="*70)
    print("TEST 4: Shape Preservation")
    print("="*70)

    aug = SpectrogramAugmentation()
    original_shape = (128, 256)
    spec = np.random.randn(*original_shape)

    augmentations = {
        'SpecAugment': lambda: aug.spec_augment(spec),
        'Time mask': lambda: aug.time_mask(spec),
        'Freq mask': lambda: aug.frequency_mask(spec),
        'Noise': lambda: aug.add_noise(spec),
        'Time shift': lambda: aug.time_shift(spec),
    }

    all_correct = True
    for name, aug_func in augmentations.items():
        result = aug_func()
        if result.shape != original_shape:
            print(f"  {name}: {result.shape} != {original_shape}")
            all_correct = False
        else:
            print(f"  {name}: {result.shape}")

    if all_correct:
        print("  PASSED: All augmentations preserve shape")
        return True
    else:
        print("  FAILED: Some augmentations change shape")
        return False


def test_dataset_with_augmentation():
    """Test 5: AudioDataset applies augmentation"""
    print("\n" + "="*70)
    print("TEST 5: Dataset Augmentation")
    print("="*70)

    # Create dummy data
    spectrograms = np.random.randn(10, 128, 128)
    labels = np.random.randint(0, 2, (10, 5)).astype(np.float32)

    # Dataset with augmentation
    dataset_aug = AudioDataset(spectrograms, labels, augment=True, multi_label=True)

    # Dataset without augmentation
    dataset_no_aug = AudioDataset(spectrograms, labels, augment=False, multi_label=True)

    # Get same sample twice with augmentation (should be different due to randomness)
    spec1, _ = dataset_aug[0]
    spec2, _ = dataset_aug[0]

    # Get same sample twice without augmentation (should be identical)
    spec3, _ = dataset_no_aug[0]
    spec4, _ = dataset_no_aug[0]

    aug_different = not torch.allclose(spec1, spec2)
    no_aug_same = torch.allclose(spec3, spec4)

    print(f"  With augmentation - samples different: {aug_different}")
    print(f"  Without augmentation - samples same:   {no_aug_same}")

    if aug_different and no_aug_same:
        print("  PASSED: Augmentation works in dataset")
        return True
    else:
        print("  FAILED: Augmentation not working properly")
        return False


def test_dataloader_creation():
    """Test 6: create_dataloaders works"""
    print("\n" + "="*70)
    print("TEST 6: DataLoader Creation")
    print("="*70)

    # Create dummy data
    n_samples = 100
    spectrograms = np.random.randn(n_samples, 128, 128)
    labels = np.random.randint(0, 2, (n_samples, 10)).astype(np.float32)

    indices = np.arange(n_samples)
    train_idx = indices[:60]
    val_idx = indices[60:80]
    test_idx = indices[80:]

    # Create dataloaders
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            spectrograms=spectrograms,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=16,
            multi_label=True,
            num_workers=0
        )

        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")

        # Try to get a batch
        batch_x, batch_y = next(iter(train_loader))
        print(f"  Batch shape:   {batch_x.shape}, {batch_y.shape}")

        print("  PASSED: DataLoaders created successfully")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DATA AUGMENTATION TEST SUITE")
    print("="*70)
    print("Testing SpecAugment, Mixup, and data pipeline\n")

    tests = [
        test_spec_augment,
        test_mixup,
        test_time_frequency_masking,
        test_augmentation_preserves_shape,
        test_dataset_with_augmentation,
        test_dataloader_creation,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {sum(results)} / {len(results)}")
    print(f"Failed: {len(results) - sum(results)} / {len(results)}")

    if all(results):
        print("\n ALL TESTS PASSED - Data augmentation works correctly!")
        return 0
    else:
        print("\n SOME TESTS FAILED - Check output above")
        return 1


if __name__ == '__main__':
    exit(main())
