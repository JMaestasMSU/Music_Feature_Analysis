#!/usr/bin/env python3
"""
Test Training Pipeline

Tests that the training script and trainer work correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import tempfile
from models.cnn_model import MultiLabelAudioCNN, MultiLabelTrainer
from models.audio_augmentation import create_dataloaders


def test_trainer_initialization():
    """Test 1: Trainer initializes correctly"""
    print("\n" + "="*70)
    print("TEST 1: Trainer Initialization")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=10)
    trainer = MultiLabelTrainer(model, device='cpu', learning_rate=0.001)

    print(f"  Device: {trainer.device}")
    print(f"  Optimizer: {type(trainer.optimizer).__name__}")
    print(f"  Loss: {type(trainer.criterion).__name__}")
    print(f"  Scheduler: {type(trainer.scheduler).__name__}")

    has_required_attrs = all([
        hasattr(trainer, 'model'),
        hasattr(trainer, 'optimizer'),
        hasattr(trainer, 'criterion'),
        hasattr(trainer, 'scheduler'),
        hasattr(trainer, 'history'),
    ])

    if has_required_attrs:
        print("  PASSED: Trainer initialized correctly")
        return True
    else:
        print("  FAILED: Missing required attributes")
        return False


def test_single_training_step():
    """Test 2: Can perform single training step"""
    print("\n" + "="*70)
    print("TEST 2: Single Training Step")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=5)
    trainer = MultiLabelTrainer(model, device='cpu')

    # Create minimal dataset
    specs = np.random.randn(10, 128, 128)
    labels = np.random.randint(0, 2, (10, 5)).astype(np.float32)
    indices = np.arange(10)

    train_loader, _, _ = create_dataloaders(
        specs, labels, indices[:7], indices[7:9], indices[9:],
        batch_size=4, multi_label=True, num_workers=0
    )

    try:
        loss = trainer.train_epoch(train_loader)
        print(f"  Training loss: {loss:.4f}")

        if loss > 0 and not np.isnan(loss):
            print("  PASSED: Training step completed")
            return True
        else:
            print("  FAILED: Invalid loss value")
            return False

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_validation_step():
    """Test 3: Can perform validation"""
    print("\n" + "="*70)
    print("TEST 3: Validation Step")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=5)
    trainer = MultiLabelTrainer(model, device='cpu')

    # Create minimal dataset
    specs = np.random.randn(10, 128, 128)
    labels = np.random.randint(0, 2, (10, 5)).astype(np.float32)
    indices = np.arange(10)

    _, val_loader, _ = create_dataloaders(
        specs, labels, indices[:7], indices[7:9], indices[9:],
        batch_size=4, multi_label=True, num_workers=0
    )

    try:
        val_loss, val_f1, val_precision, val_recall = trainer.validate(val_loader)

        print(f"  Val loss:      {val_loss:.4f}")
        print(f"  Val F1:        {val_f1:.4f}")
        print(f"  Val precision: {val_precision:.4f}")
        print(f"  Val recall:    {val_recall:.4f}")

        metrics_valid = all([
            val_loss > 0,
            0 <= val_f1 <= 1,
            0 <= val_precision <= 1,
            0 <= val_recall <= 1,
        ])

        if metrics_valid:
            print("  PASSED: Validation completed")
            return True
        else:
            print("  FAILED: Invalid metrics")
            return False

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_loop():
    """Test 4: Can train for multiple epochs"""
    print("\n" + "="*70)
    print("TEST 4: Full Training Loop (3 epochs)")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=3, base_channels=16)  # Small for speed
    trainer = MultiLabelTrainer(model, device='cpu')

    # Create dataset
    n = 20
    specs = np.random.randn(n, 128, 128)
    labels = np.random.randint(0, 2, (n, 3)).astype(np.float32)
    indices = np.arange(n)

    train_loader, val_loader, _ = create_dataloaders(
        specs, labels, indices[:14], indices[14:18], indices[18:],
        batch_size=4, multi_label=True, num_workers=0
    )

    try:
        # Use cross-platform temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            history = trainer.train(
                train_loader, val_loader,
                epochs=3, patience=10,
                save_path=tmp_path
            )

            print(f"  Training completed")
            print(f"  Epochs trained: {len(history['train_loss'])}")
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")

            # Check history has required keys
            required_keys = ['train_loss', 'val_loss', 'val_f1', 'val_precision', 'val_recall']
            has_all_keys = all(key in history for key in required_keys)

            if has_all_keys and len(history['train_loss']) > 0:
                print("  PASSED: Training loop completed")
                return True
            else:
                print("  FAILED: History incomplete")
                return False
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Test 5: Can make predictions"""
    print("\n" + "="*70)
    print("TEST 5: Prediction")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=5)
    trainer = MultiLabelTrainer(model, device='cpu')

    # Create test data
    specs = np.random.randn(8, 128, 128)
    labels = np.random.randint(0, 2, (8, 5)).astype(np.float32)
    indices = np.arange(8)

    _, _, test_loader = create_dataloaders(
        specs, labels, indices[:5], indices[5:6], indices[6:],
        batch_size=4, multi_label=True, num_workers=0
    )

    try:
        predictions, probabilities = trainer.predict(test_loader, threshold=0.5)

        print(f"  Predictions shape:    {predictions.shape}")
        print(f"  Probabilities shape:  {probabilities.shape}")
        print(f"  Prediction range:     [{predictions.min()}, {predictions.max()}]")
        print(f"  Probability range:    [{probabilities.min():.3f}, {probabilities.max():.3f}]")

        shape_correct = predictions.shape == probabilities.shape == (2, 5)
        predictions_binary = set(np.unique(predictions)) <= {0, 1}
        probs_in_range = (probabilities >= 0).all() and (probabilities <= 1).all()

        if shape_correct and predictions_binary and probs_in_range:
            print("  PASSED: Predictions correct")
            return True
        else:
            print("  FAILED: Invalid predictions")
            return False

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("TRAINING PIPELINE TEST SUITE")
    print("="*70)
    print("Testing MultiLabelTrainer and training loop\n")

    tests = [
        test_trainer_initialization,
        test_single_training_step,
        test_validation_step,
        test_full_training_loop,
        test_prediction,
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
        print("\n ALL TESTS PASSED - Training pipeline works correctly!")
        return 0
    else:
        print("\n SOME TESTS FAILED - Check output above")
        return 1


if __name__ == '__main__':
    exit(main())
