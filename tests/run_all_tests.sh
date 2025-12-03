#!/bin/bash

echo "========================================================================"
echo "RUNNING ALL TESTS - Multi-Label CNN System"
echo "========================================================================"
echo ""

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TESTS_PASSED=0
TESTS_FAILED=0

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Test 1: Multi-Label CNN Architecture
echo "[1/3] Testing Multi-Label CNN Architecture..."
if $PYTHON_CMD test_multilabel_cnn.py; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test 2: Data Augmentation
echo "[2/3] Testing Data Augmentation..."
if $PYTHON_CMD test_augmentation.py; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test 3: Training Pipeline
echo "[3/3] Testing Training Pipeline..."
if $PYTHON_CMD test_training.py; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Summary
echo "========================================================================"
echo "FINAL TEST SUMMARY"
echo "========================================================================"
echo "Passed: $TESTS_PASSED / 3"
echo "Failed: $TESTS_FAILED / 3"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo " ALL TESTS PASSED!"
    echo ""
    echo "Your multi-label CNN system is working correctly:"
    echo "  Architecture scales with genre count"
    echo "  Multi-label predictions work"
    echo "  Data augmentation preserves shapes"
    echo "  Training pipeline completes"
    echo "========================================================================"
    exit 0
else
    echo " SOME TESTS FAILED"
    echo ""
    echo "Troubleshooting:"
    echo "  - Missing PyTorch? Activate venv: source ../venv/bin/activate"
    echo "  - Missing dependencies? Install: pip install -r ../requirements.txt"
    echo "  - Check error messages above for details"
    echo "========================================================================"
    exit 1
fi
