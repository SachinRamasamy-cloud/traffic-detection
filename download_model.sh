#!/usr/bin/env bash
set -euo pipefail
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
curl -fL "$MODEL_URL" -o yolo26m.pt
echo "Downloaded: $(pwd)/yolo26m.pt"
