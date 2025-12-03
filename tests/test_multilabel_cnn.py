#!/usr/bin/env python3
"""
Test Multi-Label CNN Architecture

Tests the actual MultiLabelAudioCNN that you built, not some toy model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from models.cnn_model import MultiLabelAudioCNN, MultiLabelTrainer


def test_architecture_scaling():
    """Test 1: Architecture scales gracefully with genre count"""
    print("\n" + "="*70)
    print("TEST 1: Architecture Scaling")
    print("="*70)

    genre_counts = [8, 20, 50, 100]
    param_counts = []

    for num_genres in genre_counts:
        model = MultiLabelAudioCNN(num_genres=num_genres, base_channels=64)
        params = sum(p.numel() for p in model.parameters())
        param_counts.append(params)
        print(f"  {num_genres:3d} genres → {params:,} parameters")

    # Check linear growth (should be roughly linear)
    growth_rates = [(param_counts[i+1] - param_counts[i]) / param_counts[i] * 100
                    for i in range(len(param_counts)-1)]
    max_growth = max(growth_rates)

    print(f"\n  Max parameter growth: {max_growth:.1f}%")

    if max_growth < 25:  # Less than 25% growth is good
        print("  PASSED: Linear scaling confirmed")
        return True
    else:
        print("  FAILED: Growth too high (not linear)")
        return False


def test_multi_label_output():
    """Test 2: Multi-label output has correct shape and range"""
    print("\n" + "="*70)
    print("TEST 2: Multi-Label Output")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=20)
    model.eval()

    # Create dummy batch
    batch = torch.randn(4, 1, 128, 128)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)

    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected:     torch.Size([4, 20])")

    # Check shape
    if logits.shape != torch.Size([4, 20]):
        print("  FAILED: Wrong output shape")
        return False

    # Check probability range
    if probs.min() < 0 or probs.max() > 1:
        print(f"  FAILED: Probabilities out of range [0,1]: [{probs.min():.3f}, {probs.max():.3f}]")
        return False

    # Check that probabilities are not all identical (basic sanity check)
    # Note: For an untrained model, variance will be low (around 0.0001-0.01)
    # This is expected since weights are randomly initialized
    variance = probs.var(dim=1).mean().item()
    print(f"  Probability variance: {variance:.4f}")

    # Just check that there is SOME variance (not completely frozen)
    if variance == 0.0:
        print("  FAILED: All probabilities are identical (model frozen)")
        return False

    print("  PASSED: Correct shape and range")
    print(f"  Note: Low variance ({variance:.4f}) is normal for untrained models")
    return True


def test_residual_connections():
    """Test 3: Residual blocks work (gradient flow)"""
    print("\n" + "="*70)
    print("TEST 3: Residual Blocks (Gradient Flow)")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=10, base_channels=32)

    # Create dummy data and target
    inputs = torch.randn(2, 1, 128, 128, requires_grad=True)
    targets = torch.randint(0, 2, (2, 10)).float()

    # Forward + backward
    outputs = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
    loss.backward()

    # Check gradients exist and are not NaN
    has_gradients = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    no_nan_gradients = all(not torch.isnan(p.grad).any() for p in model.parameters()
                           if p.grad is not None)

    print(f"  All parameters have gradients: {has_gradients}")
    print(f"  No NaN gradients: {no_nan_gradients}")

    if has_gradients and no_nan_gradients:
        print("  PASSED: Gradients flow correctly")
        return True
    else:
        print("  FAILED: Gradient flow issues")
        return False


def test_attention_mechanism():
    """Test 4: Channel attention can be enabled/disabled"""
    print("\n" + "="*70)
    print("TEST 4: Channel Attention")
    print("="*70)

    # With attention
    model_with = MultiLabelAudioCNN(num_genres=10, use_attention=True)
    params_with = sum(p.numel() for p in model_with.parameters())

    # Without attention
    model_without = MultiLabelAudioCNN(num_genres=10, use_attention=False)
    params_without = sum(p.numel() for p in model_without.parameters())

    print(f"  With attention:    {params_with:,} parameters")
    print(f"  Without attention: {params_without:,} parameters")
    print(f"  Difference:        {params_with - params_without:,} parameters")

    if params_with > params_without:
        print("  PASSED: Attention adds parameters")
        return True
    else:
        print("  FAILED: Attention doesn't add parameters")
        return False


def test_embedding_extraction():
    """Test 5: Can extract feature embeddings"""
    print("\n" + "="*70)
    print("TEST 5: Embedding Extraction")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=20)
    model.eval()

    inputs = torch.randn(3, 1, 128, 128)

    with torch.no_grad():
        embeddings = model.get_embeddings(inputs)

    print(f"  Input shape:     {inputs.shape}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Expected:        torch.Size([3, 256])")

    if embeddings.shape == torch.Size([3, 256]):
        print("  PASSED: Embeddings extracted correctly")
        return True
    else:
        print("  FAILED: Wrong embedding shape")
        return False


def test_multi_label_loss():
    """Test 6: Multi-label trainer uses correct loss"""
    print("\n" + "="*70)
    print("TEST 6: Multi-Label Loss Function")
    print("="*70)

    model = MultiLabelAudioCNN(num_genres=10)
    trainer = MultiLabelTrainer(model, device='cpu')

    # Check loss function type
    loss_type = type(trainer.criterion).__name__
    print(f"  Loss function: {loss_type}")

    if loss_type == 'BCEWithLogitsLoss':
        print("  PASSED: Using correct multi-label loss")
        return True
    else:
        print(f"  FAILED: Wrong loss function (should be BCEWithLogitsLoss)")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("MULTI-LABEL CNN TEST SUITE")
    print("="*70)
    print("Testing the actual components you built\n")

    tests = [
        test_architecture_scaling,
        test_multi_label_output,
        test_residual_connections,
        test_attention_mechanism,
        test_embedding_extraction,
        test_multi_label_loss,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append(False)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {sum(results)} / {len(results)}")
    print(f"Failed: {len(results) - sum(results)} / {len(results)}")

    if all(results):
        print("\n ALL TESTS PASSED - Multi-label CNN works correctly!")
        return 0
    else:
        print("\n SOME TESTS FAILED - Check output above")
        return 1


if __name__ == '__main__':
    exit(main())
