#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Force CPU-only PyTorch wheels. This avoids downloading CUDA dependencies.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Headless OpenCV variant is suitable for Linux servers without a desktop GUI.
pip install ultralytics-opencv-headless==8.4.104

echo "Environment ready. Activate it with: source .venv/bin/activate"
