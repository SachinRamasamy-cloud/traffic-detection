#!/usr/bin/env python3
"""Static checks that do not require downloading the YOLO model."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ast.parse((ROOT / "tracker.py").read_text(encoding="utf-8"))
    config = (ROOT / "bytetrack_cpu.yaml").read_text(encoding="utf-8")
    required = {
        "tracker_type:",
        "track_high_thresh:",
        "track_low_thresh:",
        "new_track_thresh:",
        "track_buffer:",
        "match_thresh:",
    }
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise AssertionError(f"Missing tracker keys: {missing}")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
