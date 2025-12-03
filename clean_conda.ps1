try {
    $json = conda env list --json 2>$null | Out-String
    if (-not $json) {
        Write-Error "Failed to run 'conda env list --json'. Ensure conda is on PATH or run this from an Anaconda/Miniconda shell."
        exit 1
    }

    $envPaths = (ConvertFrom-Json $json).envs
    $envNames = $envPaths | ForEach-Object { Split-Path $_ -Leaf } | Where-Object { $_ -and $_ -ne 'base' } | Sort-Object -Unique

    if (-not $envNames) {
        Write-Output "No non-base conda environments found."
        exit 0
    }

    Write-Output "The following conda environments will be removed:`n$($envNames -join "`n")"
    $confirm = Read-Host "Type 'yes' to proceed"
    if ($confirm -ne 'yes') {
        Write-Output "Aborted by user."
        exit 0
    }

    foreach ($env in $envNames) {
        Write-Output "Removing environment: $env"
        conda env remove --name $env -y
    }

    Write-Output "All requested environments removed."
}
catch {
    Write-Error "Error: $($_.Exception.Message)"
    exit 1
}