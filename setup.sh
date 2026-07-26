#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip uninstall -y opencv-python-headless >/dev/null 2>&1 || true
python -m pip install -e .

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "For plate OCR, run: ./install_ocr.sh"
