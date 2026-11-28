param(
    [string]$TorchBackend
)

$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    $projectPath = Join-Path $PSScriptRoot "pyproject.toml"
    $original = [IO.File]::ReadAllText($projectPath)
    # Edit only the URL of VisualNN's named PyTorch index.
    $indexPattern = '(?m)^\[\[tool\.uv\.index\]\]\r?\nname = "pytorch"\r?\nurl = "https://download\.pytorch\.org/whl/(?<backend>[a-z0-9.]+)"'
    $index = [regex]::Match($original, $indexPattern)
    if (-not $index.Success) {
        throw "Could not locate the named PyTorch index in pyproject.toml."
    }

    $current = $index.Groups["backend"]
    if (-not $PSBoundParameters.ContainsKey("TorchBackend")) {
        $TorchBackend = Read-Host "PyTorch backend: cpu, auto (detect hardware), or a CUDA build such as cu130 [Enter = $($current.Value)]"
        if ([string]::IsNullOrWhiteSpace($TorchBackend)) {
            $TorchBackend = $current.Value
        }
    }
    $TorchBackend = $TorchBackend.Trim().ToLowerInvariant()
    if ($TorchBackend -eq "auto") {
        # Ask uv for its hardware-based choice without installing torch.
        $resolution = "torch" | uv pip compile - --python 3.12 --no-config --no-deps --no-header --emit-index-annotation --torch-backend auto
        if ($LASTEXITCODE -ne 0) { throw "uv could not detect and resolve a PyTorch backend." }
        $source = [regex]::Match(($resolution -join "`n"), '(?m)^\s*# from https://download\.pytorch\.org/whl/([a-z0-9.]+)/?\s*$')
        if (-not $source.Success) { throw "uv did not report a PyTorch index. Specify -TorchBackend explicitly." }
        $TorchBackend = $source.Groups[1].Value
    }
    if ($TorchBackend -notmatch '^(cpu|cu[0-9]+|xpu|rocm[0-9]+(?:\.[0-9]+)*)$') {
        throw "Unknown PyTorch backend '$TorchBackend'. Choose cpu, auto, or a PyTorch build such as cu130."
    }

    Write-Host "Using PyTorch index https://download.pytorch.org/whl/$TorchBackend"
    $updated = $original.Remove($current.Index, $current.Length).Insert($current.Index, $TorchBackend)
    $utf8 = [Text.UTF8Encoding]::new($false)
    try {
        if ($updated -cne $original) { [IO.File]::WriteAllText($projectPath, $updated, $utf8) }
        uv lock --python 3.12
        if ($LASTEXITCODE -ne 0) { throw "uv could not lock dependencies for $TorchBackend." }
    }
    catch {
        if ($updated -cne $original) { [IO.File]::WriteAllText($projectPath, $original, $utf8) }
        throw
    }

    uv sync --locked --python 3.12 --extra dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed. Rerun setup to retry with the selected backend." }

    uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch verification failed." }
}
finally {
    Pop-Location
}
