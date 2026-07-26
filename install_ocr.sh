#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ "paddlepaddle>=3.2,<4"
python -m pip install -e ".[ocr]"

python - <<'PY'
import paddle
from paddleocr import TextRecognition
print("Paddle:", paddle.__version__)
print("PaddleOCR TextRecognition import: OK")
PY

echo "OCR setup complete."
