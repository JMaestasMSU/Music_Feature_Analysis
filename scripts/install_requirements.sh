#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  echo "Conda detected. Creating environment from environment-cpu.yml..."
  conda env create -f environment-cpu.yml -n mfa-cpu
  echo "Activate with: conda activate mfa-cpu"
else
  echo "Conda not found. Installing via pip into current environment..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi
