# GPU setup notes

This project can benefit from NVIDIA GPUs for training CNN models on spectrograms.

Checklist to enable GPU support:

- NVIDIA GPU with recent drivers installed
- Install CUDA toolkit version compatible with the PyTorch build you plan to use
- Install PyTorch matching the CUDA version (see https://pytorch.org/get-started/locally/)

Example conda install for PyTorch with CUDA 11.8 (change version as needed):

```powershell
conda install -n mfa -c pytorch pytorch=2.9.0 cudatoolkit=13.0 torchvision torchaudio -y
```

If you don't have a GPU, install CPU-only PyTorch via pip or conda as documented in `environment.yml`.
