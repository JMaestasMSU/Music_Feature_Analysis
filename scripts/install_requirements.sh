#!/usr/bin/env bash
set -euo pipefail

# Default to CPU environment
ENV_TYPE="cpu"
ENV_NAME="mfa-cpu"
ENV_FILE="../environments/environment-cpu.yml"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu|--cuda)
      ENV_TYPE="gpu"
      ENV_NAME="mfa-gpu-cuda11-8"
      ENV_FILE="../environments/environment-gpu-cuda11.8.yml"
      shift
      ;;
    --base)
      ENV_TYPE="base"
      ENV_NAME="music-feature-analysis"
      ENV_FILE="../environments/environment.yml"
      shift
      ;;
    --name)
      ENV_NAME="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: ./install_requirements.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --gpu, --cuda    Install GPU environment with CUDA 11.8 support (for training)"
      echo "  --base           Install base environment (general use)"
      echo "  --name NAME      Custom environment name (default: mfa-cpu or mfa-gpu-cuda11-8)"
      echo "  --help, -h       Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./install_requirements.sh              # CPU-only (default)"
      echo "  ./install_requirements.sh --gpu        # GPU with CUDA 11.8"
      echo "  ./install_requirements.sh --base       # Base environment"
      echo "  ./install_requirements.sh --gpu --name my-gpu-env"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

if command -v conda >/dev/null 2>&1; then
  echo "=================================================="
  echo "Conda Environment Setup"
  echo "=================================================="
  echo "Environment type: $ENV_TYPE"
  echo "Environment name: $ENV_NAME"
  echo "Environment file: $ENV_FILE"
  echo ""

  if [ "$ENV_TYPE" = "gpu" ]; then
    echo "  GPU environment requires:"
    echo "   - NVIDIA GPU with CUDA support"
    echo "   - NVIDIA drivers installed"
    echo "   - CUDA 11.8 compatible GPU"
    echo ""
  fi

  echo "Creating conda environment..."
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"

  echo ""
  echo " Environment created successfully!"
  echo ""
  echo "Activate with:"
  echo "  conda activate $ENV_NAME"
  echo ""
  if [ "$ENV_TYPE" = "gpu" ]; then
    echo "Verify GPU support:"
    echo "  conda activate $ENV_NAME"
    echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
  fi
else
  echo "Conda not found. Installing via pip into current environment..."
  echo "  Note: This will install CPU-only PyTorch"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo ""
  echo " Installed requirements into current Python environment"
fi
