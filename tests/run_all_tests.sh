#!/bin/bash

# Music Feature Analysis - Quick Test Suite
# Runs all component tests in < 2 minutes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
CI_MODE=false
CPU_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ci-mode)
            CI_MODE=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Music Feature Analysis - Test Suite"
echo "=========================================="
echo ""

# Track results
TESTS_PASSED=0
TESTS_FAILED=0
START_TIME=$(date +%s)

# Function to run test and track result
run_test() {
    local test_name=$1
    local test_file=$2
    local test_args=$3
    
    echo -n "Running $test_name... "
    
    if [ "$CI_MODE" = true ]; then
        test_args="$test_args --ci"
    fi
    
    if [ "$CPU_ONLY" = true ]; then
        test_args="$test_args --cpu-only"
    fi
    
    if python "$test_file" $test_args > /dev/null 2>&1; then
        echo -e "${GREEN}[OK] PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}[FAIL] FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test 1: FFT Validation
run_test "FFT Validation" "quick_fft_test.py" ""

# Test 2: Audio Processing
run_test "Audio Processing" "quick_audio_processing_test.py" ""

# Test 3: CNN Architecture (CPU-only in CI)
if [ "$CI_MODE" = true ] || [ "$CPU_ONLY" = true ]; then
    run_test "CNN Architecture (CPU)" "quick_cnn_test.py" "--cpu-only"
else
    run_test "CNN Architecture" "quick_cnn_test.py" ""
fi

# Test 4: Bayesian Optimization (optional in CI)
if [ "$CI_MODE" = false ]; then
    run_test "Bayesian Optimization" "quick_bayesian_test.py" ""
fi

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Print summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo "Duration: ${DURATION}s"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else:
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
