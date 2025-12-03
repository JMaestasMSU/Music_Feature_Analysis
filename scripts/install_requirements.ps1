<#
.SYNOPSIS
    Bootstrap installer for Music Feature Analysis project.

.DESCRIPTION
    Creates conda environment with CPU or GPU support, or installs via pip if conda is not available.

.PARAMETER GPU
    Install GPU environment with CUDA 11.8 support (for training with NVIDIA GPU).

.PARAMETER Base
    Install base environment (general use).

.PARAMETER EnvName
    Custom environment name. If not specified, uses default based on environment type:
    - CPU: mfa-cpu
    - GPU: mfa-gpu-cuda11-8
    - Base: music-feature-analysis

.PARAMETER Dev
    Install development dependencies (pytest, black, flake8, etc.).

.EXAMPLE
    .\install_requirements.ps1
    # Install CPU-only environment (default)

.EXAMPLE
    .\install_requirements.ps1 -GPU
    # Install GPU environment with CUDA 11.8 support

.EXAMPLE
    .\install_requirements.ps1 -GPU -Dev
    # Install GPU environment with dev dependencies

.EXAMPLE
    .\install_requirements.ps1 -Base -EnvName my-custom-env
    # Install base environment with custom name

.NOTES
    Requires conda for best experience. Falls back to pip if conda is not available.
#>

param(
    [switch]$GPU,
    [switch]$Base,
    [string]$EnvName = "",
    [switch]$Dev,
    [switch]$Help
)

function Show-Help {
    Write-Host ""
    Write-Host "Music Feature Analysis - Environment Setup" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\install_requirements.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -GPU           Install GPU environment with CUDA 11.8 (for training)"
    Write-Host "  -Base          Install base environment (general use)"
    Write-Host "  -EnvName NAME  Custom environment name"
    Write-Host "  -Dev           Install development dependencies"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\install_requirements.ps1                    # CPU-only (default)"
    Write-Host "  .\install_requirements.ps1 -GPU               # GPU with CUDA 11.8"
    Write-Host "  .\install_requirements.ps1 -GPU -Dev          # GPU with dev tools"
    Write-Host "  .\install_requirements.ps1 -Base              # Base environment"
    Write-Host "  .\install_requirements.ps1 -GPU -EnvName my-env"
    Write-Host ""
    Write-Host "Environment Types:" -ForegroundColor Yellow
    Write-Host "  CPU:  " -NoNewline
    Write-Host "CPU-only PyTorch (inference, feature extraction)" -ForegroundColor Gray
    Write-Host "  GPU:  " -NoNewline
    Write-Host "GPU PyTorch with CUDA 11.8 (fast training)" -ForegroundColor Gray
    Write-Host "  Base: " -NoNewline
    Write-Host "General environment (less specific)" -ForegroundColor Gray
    Write-Host ""
}

function Has-Conda {
    try {
        $null = conda --version 2>&1
        return $true
    } catch {
        return $false
    }
}

# Show help if requested
if ($Help) {
    Show-Help
    exit 0
}

# Determine environment type and file
$EnvType = "cpu"
$EnvFile = "..\environments\environment-cpu.yml"
$DefaultEnvName = "mfa-cpu"

if ($GPU) {
    $EnvType = "gpu"
    $EnvFile = "..\environments\environment-gpu-cuda11.8.yml"
    $DefaultEnvName = "mfa-gpu-cuda11-8"
} elseif ($Base) {
    $EnvType = "base"
    $EnvFile = "..\environments\environment.yml"
    $DefaultEnvName = "music-feature-analysis"
}

# Use custom name or default
if ([string]::IsNullOrEmpty($EnvName)) {
    $EnvName = $DefaultEnvName
}

if (Has-Conda) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "Conda Environment Setup" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "Environment type: " -NoNewline
    Write-Host $EnvType -ForegroundColor Green
    Write-Host "Environment name: " -NoNewline
    Write-Host $EnvName -ForegroundColor Green
    Write-Host "Environment file: " -NoNewline
    Write-Host $EnvFile -ForegroundColor Green
    Write-Host ""

    if ($GPU) {
        Write-Host "  GPU environment requires:" -ForegroundColor Yellow
        Write-Host "   - NVIDIA GPU with CUDA support"
        Write-Host "   - NVIDIA drivers installed"
        Write-Host "   - CUDA 11.8 compatible GPU"
        Write-Host ""
    }

    Write-Host "Creating conda environment..." -ForegroundColor Cyan
    conda env create -f $EnvFile -n $EnvName

    Write-Host ""
    Write-Host " Environment created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Activate with:" -ForegroundColor Yellow
    Write-Host "  conda activate $EnvName" -ForegroundColor White
    Write-Host ""

    if ($GPU) {
        Write-Host "Verify GPU support:" -ForegroundColor Yellow
        Write-Host "  conda activate $EnvName" -ForegroundColor White
        Write-Host "  python -c `"import torch; print(f'CUDA available: {torch.cuda.is_available()}')`"" -ForegroundColor White
        Write-Host ""
    }

    if ($Dev) {
        Write-Host "Installing dev dependencies into conda env: $EnvName" -ForegroundColor Cyan
        conda run -n $EnvName python -m pip install --upgrade pip

        if (Test-Path "..\scripts\dev-requirements.txt") {
            conda run -n $EnvName python -m pip install -r ..\scripts\dev-requirements.txt
        } else {
            Write-Host "  dev-requirements.txt not found, skipping dev dependencies" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "Conda not found. Installing via pip into current environment..." -ForegroundColor Yellow
    Write-Host "  Note: This will install CPU-only PyTorch" -ForegroundColor Yellow
    Write-Host ""

    python -m pip install --upgrade pip

    if (Test-Path "..\requirements.txt") {
        python -m pip install -r ..\requirements.txt
    } else {
        Write-Host " requirements.txt not found" -ForegroundColor Red
        exit 1
    }

    if ($Dev -and (Test-Path "..\scripts\dev-requirements.txt")) {
        python -m pip install -r ..\scripts\dev-requirements.txt
    }

    Write-Host ""
    Write-Host " Installed requirements into current Python environment" -ForegroundColor Green
}
