"""Configure VisualNN's PyTorch backend and synchronize its uv environment."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "pyproject.toml"
INDEX_URL = "https://download.pytorch.org/whl/"
BACKEND_PATTERN = r"(?:cpu|cu[0-9]+|xpu|rocm[0-9]+(?:\.[0-9]+)*)"


def detect_backend():
    result = subprocess.run(
        ["uv", "pip", "compile", "-", "--python", "3.12", "--no-config",
         "--no-deps", "--no-header", "--emit-index-annotation", "--torch-backend", "auto"],
        cwd=ROOT, input="torch\n", text=True, stdout=subprocess.PIPE, check=True,
    )
    match = re.search(
        rf"(?m)^[ \t]*# from {re.escape(INDEX_URL)}({BACKEND_PATTERN})/?[ \t]*$",
        result.stdout,
    )
    if not match:
        raise ValueError("uv did not report a PyTorch index. Specify --torch-backend explicitly.")
    return match[1]


def write_project(text):
    # Use the same directory for an atomic replacement, including on Windows.
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        temporary = Path(directory) / PROJECT.name
        temporary.write_bytes(text.encode("utf-8"))
        temporary.chmod(PROJECT.stat().st_mode)
        os.replace(temporary, PROJECT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-backend", "-TorchBackend", help="cpu, auto, or a build such as cu130")
    args = parser.parse_args()

    # Read bytes to preserve the checkout's original line endings.
    original = PROJECT.read_bytes().decode("utf-8")
    match = re.search(
        r'(?m)^\[\[tool\.uv\.index\]\]\r?\nname = "pytorch"\r?\n'
        rf'url = "{re.escape(INDEX_URL)}(?P<backend>{BACKEND_PATTERN})"',
        original,
    )
    if not match:
        raise ValueError("Could not locate the named PyTorch index in pyproject.toml.")
    current = match["backend"]
    backend = args.torch_backend
    if backend is None:
        answer = input(f"PyTorch backend: cpu, auto (detect hardware), or e.g. cu130 [Enter = {current}]: ")
        backend = answer.strip() or current
    backend = backend.strip().lower()
    if backend == "auto":
        backend = detect_backend()
    if not re.fullmatch(BACKEND_PATTERN, backend):
        raise ValueError(f"Unknown PyTorch backend {backend!r}. Choose cpu, auto, or e.g. cu130.")

    print(f"Using PyTorch index {INDEX_URL}{backend}", flush=True)
    if backend != current:
        start, end = match.span("backend")
        write_project(original[:start] + backend + original[end:])
    try:
        subprocess.run(["uv", "lock", "--python", "3.12"], cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError, KeyboardInterrupt):
        if backend != current:
            write_project(original)
        raise

    subprocess.run(["uv", "sync", "--locked", "--python", "3.12", "--extra", "dev"], cwd=ROOT, check=True)
    diagnostics = (
        "import torch\n"
        "print('PyTorch:', torch.__version__)\n"
        "print('CUDA build:', torch.version.cuda)\n"
        "print('CUDA available:', torch.cuda.is_available())"
    )
    subprocess.run(["uv", "run", "--no-sync", "python", "-c", diagnostics], cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, EOFError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Setup failed: {exc}")
