#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
OUTPUT="${1:-yolo26n.pt}"

curl -L --fail --retry 3 --output "$OUTPUT" "$MODEL_URL"
echo "Downloaded YOLO26n to: $OUTPUT"
