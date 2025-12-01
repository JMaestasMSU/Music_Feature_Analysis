#!/bin/bash

echo "========================================================================"
echo "RUNNING ALL QUICK TESTS - Music Feature Analysis"
echo "========================================================================"
echo ""

# Determine script directory (works on macOS, Linux, Windows Git Bash)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TESTS_PASSED=0
TESTS_FAILED=0

# Determine Python command (python3 on macOS/Linux, python on Windows)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Test 1: FFT Validation
echo "[1/4] Running FFT validation test..."
if $PYTHON_CMD quick_fft_test.py; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test 2: Audio Processing
echo "[2/4] Running audio processing test..."
if $PYTHON_CMD quick_audio_processing_test.py; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test 3: CNN Architecture
echo "[3/4] Running CNN architecture test..."
if $PYTHON_CMD quick_cnn_test.py --cpu-only; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test 4: Bayesian Optimization
echo "[4/4] Running Bayesian optimization test..."
if $PYTHON_CMD quick_bayesian_test.py; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Summary
echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo "Passed: $TESTS_PASSED / 4"
echo "Failed: $TESTS_FAILED / 4"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo "[OK] All tests passed! ✓"
    echo "========================================================================"
    exit 0
else
    echo ""
    echo "[FAIL] Some tests failed ✗"
    echo "========================================================================"
    exit 1
fi
