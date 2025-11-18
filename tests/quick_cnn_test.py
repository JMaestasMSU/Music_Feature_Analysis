"""
Quick CNN Architecture Test
Tests CNN model can be instantiated, forward pass works, and basic training loop runs.
Supports both CPU and GPU modes.
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Quick CNN Test')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only mode')
    parser.add_argument('--ci', action='store_true', help='CI mode (faster, less verbose)')
    return parser.parse_args()

class SimpleCNN(nn.Module):
    """Simple CNN for genre classification from spectrograms"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Input: 128 mel-bins × 216 time frames → after 3 poolings: 16 × 27
        self.fc1 = nn.Linear(128 * 16 * 27, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.conv1(x)))  # → 32 × 64 × 108
        
        # Conv block 2
        x = self.pool(self.relu(self.conv2(x)))  # → 64 × 32 × 54
        
        # Conv block 3
        x = self.pool(self.relu(self.conv3(x)))  # → 128 × 16 × 27
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def test_model_instantiation(device, verbose=True):
    """Test 1: Model can be instantiated"""
    if verbose:
        print("Test 1: Model instantiation...", end=" ")
    
    try:
        model = SimpleCNN(num_classes=10)
        model = model.to(device)
        
        if verbose:
            print("[OK] PASSED")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_forward_pass(device, verbose=True):
    """Test 2: Forward pass works"""
    if verbose:
        print("Test 2: Forward pass...", end=" ")
    
    try:
        model = SimpleCNN(num_classes=10).to(device)
        
        # Create dummy spectrogram input (batch_size=2, channels=1, height=128, width=216)
        dummy_input = torch.randn(2, 1, 128, 216).to(device)
        
        # Forward pass
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
        
        if verbose:
            print("[OK] PASSED")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_training_step(device, verbose=True):
    """Test 3: Training step works (backward pass)"""
    if verbose:
        print("Test 3: Training step...", end=" ")
    
    try:
        model = SimpleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Dummy data
        inputs = torch.randn(4, 1, 128, 216).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Check loss is finite
        assert torch.isfinite(loss), "Loss is not finite"
        
        if verbose:
            print(f"[OK] PASSED (loss: {loss.item():.4f})")
        return True
    except Exception as e:
        if verbose:
            print(f"[FAIL] FAILED: {e}")
        return False

def test_inference_mode(device, verbose=True):
    """Test 4: Inference mode works"""
    if verbose:
        print("Test 4: Inference mode...", end=" ")
    
    try:
        model = SimpleCNN(num_classes=10).to(device)
        model.eval()
        
        with torch.no_grad():
            inputs = torch.randn(1, 1, 128, 216).to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Check outputs
        assert probabilities.shape == (1, 10)
        assert torch.allclose(probabilities.sum(dim=1), torch.tensor([1.0]).to(device))
        assert 0 <= predicted_class.item() < 10
        
        if verbose:
            print(f"[OK] PASSED (predicted class: {predicted_class.item()})")
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
        print("Quick CNN Architecture Test")
        print("=" * 60)
        print()
    
    # Determine device
    if args.cpu_only:
        device = torch.device('cpu')
        if verbose:
            print("Mode: CPU-only (forced)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"Mode: {device.type.upper()}")
    
    if verbose:
        print()
    
    # Run tests
    tests = [
        test_model_instantiation,
        test_forward_pass,
        test_training_step,
        test_inference_mode
    ]
    
    results = []
    for test in tests:
        result = test(device, verbose=verbose)
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
            print("\n[OK] All CNN tests passed!")
        sys.exit(0)
    else:
        if verbose:
            print(f"\n[FAIL] {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
