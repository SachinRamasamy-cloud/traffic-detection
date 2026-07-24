from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .runtime import configure_cpu_runtime


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLO26m ROI-aware tiled ByteTrack pipeline for Linux CPU")
    parser.add_argument("--source", required=True)
    parser.add_argument("--roi", default="", help="ROI JSON path. Empty uses the complete frame.")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--output-dir", default="runs/yolo26m_roi")
    parser.add_argument("--tracker", default=str(project_root() / "configs" / "bytetrack_traffic.yaml"))
    parser.add_argument("--classes", default="", help="Comma-separated class names or IDs. Empty means all classes.")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.06)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--one-to-many", action="store_true", help="Use YOLO26 one-to-many head for higher recall at extra cost.")

    parser.add_argument("--tile-size", type=int, default=960, help="Source-pixel tile size before YOLO resize.")
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--tile-batch-size", type=int, default=1, help="Keep 1 on low-memory CPUs; test 2 on stronger machines.")
    parser.add_argument("--min-tile-roi-ratio", type=float, default=0.005)
    parser.add_argument("--max-tiles", type=int, default=0, help="0 keeps all intersecting ROI tiles.")
    parser.add_argument("--merge-iou", type=float, default=0.55)
    parser.add_argument("--class-aware-merge", action="store_true", help="Default merge is class-agnostic to remove cross-class tile duplicates.")
    parser.add_argument("--max-detections", type=int, default=1000)
    parser.add_argument("--mask-outside-roi", action="store_true")
    parser.add_argument("--mask-dilation", type=int, default=8)

    parser.add_argument("--roi-rule", choices=["bottom_center", "center", "overlap"], default="")
    parser.add_argument("--roi-min-overlap", type=float, default=None)
    parser.add_argument("--curve-samples", type=int, default=24)

    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--prediction-frames", type=int, default=0, help="Export/draw short Kalman-only lost-track predictions. 0 disables.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    parser.add_argument("--class-history", type=int, default=20)
    parser.add_argument("--class-min-observations", type=int, default=4)
    parser.add_argument("--class-switch-ratio", type=float, default=1.75)
    parser.add_argument("--class-stale-frames", type=int, default=300)

    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--draw-roi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-tiles", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)



def validate(args) -> None:
    if args.imgsz <= 0 or args.tile_size <= 0 or args.tile_batch_size <= 0:
        raise ValueError("imgsz, tile-size, and tile-batch-size must be positive")
    for name in ("conf", "iou", "tile_overlap", "min_tile_roi_ratio", "merge_iou"):
        value = getattr(args, name)
        upper = 0.9 if name == "tile_overlap" else 1.0
        if not 0.0 <= value <= upper:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and {upper}")
    if args.vid_stride <= 0 or args.threads <= 0 or args.curve_samples < 4:
        raise ValueError("vid-stride and threads must be positive; curve-samples must be >= 4")
    if args.prediction_frames < 0:
        raise ValueError("prediction-frames cannot be negative")
    tracker_low = None
    import yaml
    tracker_cfg = yaml.safe_load(Path(args.tracker).read_text(encoding="utf-8"))
    tracker_low = float(tracker_cfg["track_low_thresh"])
    if args.conf > tracker_low:
        raise ValueError(
            f"Detector --conf ({args.conf}) must be <= ByteTrack track_low_thresh ({tracker_low}) "
            "or ByteTrack cannot use its low-confidence recovery stage."
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        validate(args)
        configure_cpu_runtime(args.threads)
        from .pipeline import run
        summary = run(args)
        print(json.dumps(summary, indent=2))
        return 0
    except KeyboardInterrupt:
        logging.getLogger("traffic_tracker").warning("Interrupted by user")
        return 130
    except Exception as exc:
        logging.getLogger("traffic_tracker").exception("Tracking failed: %s", exc)
        return 1
