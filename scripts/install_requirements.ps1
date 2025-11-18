<#
Bootstrap installer for the project.
Try to create the conda environment if conda is available, otherwise install requirements.txt into the current Python environment.

Usage (PowerShell):
    .\scripts\install_requirements.ps1 [-EnvName mfa-cpu]
#>

param(
    [string]$EnvName = "mfa-cpu",
    [switch]$Dev
)

function Has-Conda {
    try {
        conda --version > $null 2>&1
        return $true
    } catch {
        return $false
    }
}

if (Has-Conda) {
    Write-Host "Conda detected. Creating environment from environment-cpu.yml..."
    conda env create -f environment-cpu.yml -n $EnvName
    Write-Host "Activate the environment with: conda activate $EnvName"
    if ($Dev) {
        Write-Host "Installing dev dependencies into conda env: $EnvName"
        conda run -n $EnvName python -m pip install --upgrade pip
        conda run -n $EnvName python -m pip install -r scripts/dev-requirements.txt
    }
} else {
    Write-Host "Conda not found. Installing via pip into current environment..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if ($Dev) {
        python -m pip install -r scripts/dev-requirements.txt
    }
    Write-Host "Installed requirements into current Python environment."
}
