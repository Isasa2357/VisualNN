param(
    [string]$TorchBackend = "auto"
)

$ErrorActionPreference = "Stop"

# Run this script from any working directory.
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv was not found. Please install uv and add it to PATH."
    }

    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host "[1/3] Creating a Python 3.12 virtual environment..."
        & uv venv --python 3.12 .venv
        if ($LASTEXITCODE -ne 0) { throw "uv venv failed." }
    }
    else {
        Write-Host "[1/3] Using the existing .venv."
    }

    # An installed CPU build may satisfy an unpinned torch requirement.
    # Remove it before resolving the correct build for this computer.
    Write-Host "[2/3] Removing existing PyTorch builds..."
    & uv pip uninstall --python $python torch torchvision torchaudio
    if ($LASTEXITCODE -ne 0) { throw "PyTorch uninstall failed." }

    Write-Host "[3/3] Installing project and development dependencies ($TorchBackend)..."
    & uv pip install --python $python --torch-backend $TorchBackend -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

    Write-Host ""
    Write-Host "PyTorch diagnostics:"
    & $python -c "import torch; print('version:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch verification failed." }

    Write-Host ""
    Write-Host "Setup complete. You can now run:"
    Write-Host "  cd StudyClassification"
    Write-Host "  uv run -m ResNet.main"
}
finally {
    Pop-Location
}
