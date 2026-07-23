#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:?Usage: scripts/run_linux_cpu.sh /path/to/video.mp4 [output.json]}"
OUTPUT="${2:-outputs/results.json}"

python -m traffic_plate_study run \
  --video "$VIDEO" \
  --output "$OUTPUT" \
  --config config/default.yaml \
  --annotated-video outputs/annotated.mp4
