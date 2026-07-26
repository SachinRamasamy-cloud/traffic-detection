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
    parser = argparse.ArgumentParser(
        description="YOLO26m static ROI-tile ByteTrack pipeline with plate detection and optional OCR"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--roi", default="", help="ROI JSON path. Empty uses the complete frame.")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--output-dir", default="runs/yolo26m_roi")
    parser.add_argument("--tracker", default=str(project_root() / "configs" / "bytetrack_traffic.yaml"))
    parser.add_argument("--classes", default="", help="Comma-separated class names or IDs. Empty means all classes.")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.06)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--one-to-many",
        action="store_true",
        help="Use YOLO26 one-to-many head for higher recall at extra cost.",
    )

    # Static ROI tile-plan settings. The coordinates are generated once and then
    # reused for every video frame.
    parser.add_argument("--tile-size", type=int, default=960, help="Source-pixel tile size before YOLO resize.")
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--tile-batch-size", type=int, default=1, help="Keep 1 on low-memory CPUs; test 2 on stronger machines.")
    parser.add_argument("--min-tile-roi-ratio", type=float, default=0.005)
    parser.add_argument("--max-tiles", type=int, default=0, help="0 keeps all selected ROI tiles.")
    parser.add_argument("--roi-tile-padding", type=int, default=96, help="Source-pixel context outside the ROI used when building the static tile plan.")
    parser.add_argument("--roi-detection-padding", type=int, default=64, help="Allow detections this many pixels outside the exact ROI so entry vehicles can start tracking early.")
    parser.add_argument(
        "--force-boundary-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep ROI entry/exit boundary tiles even when their ROI coverage ratio is small.",
    )
    parser.add_argument("--tile-plan", default="", help="Persistent tile-plan JSON. Empty writes <output-dir>/tile_plan.json.")
    parser.add_argument("--rebuild-tile-plan", action="store_true")
    parser.add_argument("--tile-plan-frame-index", type=int, default=0)
    parser.add_argument("--save-tile-preview", action=argparse.BooleanOptionalAction, default=True)
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

    # ANPR plate localization.
    parser.add_argument("--plate-model", default="", help="One-class license-plate detector weights. Empty disables plate detection.")
    parser.add_argument("--plate-imgsz", type=int, default=640)
    parser.add_argument("--plate-conf", type=float, default=0.20)
    parser.add_argument("--plate-iou", type=float, default=0.50)
    parser.add_argument("--plate-class-id", type=int, default=0, help="Plate class ID in the plate model. Use -1 to accept all classes.")
    parser.add_argument("--plate-interval", type=int, default=2, help="Run plate detection every N processed frames per vehicle, staggered by track ID.")
    parser.add_argument("--plate-batch-size", type=int, default=2)
    parser.add_argument("--plate-vehicle-classes", default="car,motorcycle,bus,truck", help="Vehicle classes eligible for plate search.")
    parser.add_argument("--plate-min-vehicle-width", type=int, default=96)
    parser.add_argument("--plate-min-vehicle-height", type=int, default=64)
    parser.add_argument("--plate-vehicle-padding", type=float, default=0.08)
    parser.add_argument("--plate-search-start", type=float, default=0.25, help="Search from this vertical fraction of each vehicle crop. 0.25 means lower 75%%.")
    parser.add_argument("--plate-search-full-vehicle", action="store_true")
    parser.add_argument("--plate-min-width", type=int, default=8)
    parser.add_argument("--plate-min-height", type=int, default=4)
    parser.add_argument("--plate-min-aspect", type=float, default=0.8)
    parser.add_argument("--plate-max-aspect", type=float, default=10.0)
    parser.add_argument("--plate-cache-frames", type=int, default=10, help="Keep the last plate box visible between scheduled detector frames.")
    parser.add_argument("--save-plate-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-only-best-plate-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-plates", action=argparse.BooleanOptionalAction, default=True)

    # Recognition-only OCR runs on localized plate crops. Results are aggregated
    # per ByteTrack vehicle ID using confidence-weighted temporal consensus.
    parser.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ocr-model", default="en_PP-OCRv5_mobile_rec")
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--ocr-engine", default="", help="Optional PaddleOCR engine, e.g. paddle_static or onnxruntime.")
    parser.add_argument("--ocr-cpu-threads", type=int, default=4)
    parser.add_argument("--ocr-enable-hpi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ocr-batch-size", type=int, default=4)
    parser.add_argument("--ocr-interval", type=int, default=1, help="Minimum frames between OCR attempts for the same vehicle track.")
    parser.add_argument("--ocr-max-reads-per-track", type=int, default=12, help="0 allows unlimited OCR attempts per track.")
    parser.add_argument("--ocr-target-height", type=int, default=96)
    parser.add_argument("--ocr-max-width", type=int, default=640)
    parser.add_argument("--ocr-min-score", type=float, default=0.20)
    parser.add_argument("--ocr-min-text-length", type=int, default=4)
    parser.add_argument("--ocr-max-text-length", type=int, default=14)
    parser.add_argument("--ocr-min-plate-width", type=int, default=12)
    parser.add_argument("--ocr-min-plate-height", type=int, default=4)
    parser.add_argument("--ocr-min-sharpness", type=float, default=0.0)
    parser.add_argument("--ocr-pattern", default="", help="Optional full-match regular expression for accepted plate text.")
    parser.add_argument("--ocr-variants", default="colour,gray,clahe,sharpened")
    parser.add_argument("--ocr-confirm-observations", type=int, default=3)
    parser.add_argument("--ocr-confirm-score", type=float, default=0.50)
    parser.add_argument("--ocr-confirm-dominance", type=float, default=0.60)
    parser.add_argument("--ocr-history-size", type=int, default=30)

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
    if args.roi_tile_padding < 0 or args.roi_detection_padding < 0:
        raise ValueError("ROI padding values cannot be negative")
    if args.tile_plan_frame_index < 0:
        raise ValueError("tile-plan-frame-index cannot be negative")

    import yaml

    tracker_cfg = yaml.safe_load(Path(args.tracker).read_text(encoding="utf-8"))
    tracker_low = float(tracker_cfg["track_low_thresh"])
    if args.conf > tracker_low:
        raise ValueError(
            f"Detector --conf ({args.conf}) must be <= ByteTrack track_low_thresh ({tracker_low}) "
            "or ByteTrack cannot use its low-confidence recovery stage."
        )

    if args.plate_model:
        if args.plate_imgsz <= 0 or args.plate_interval <= 0 or args.plate_batch_size <= 0:
            raise ValueError("plate-imgsz, plate-interval, and plate-batch-size must be positive")
        if args.plate_min_vehicle_width <= 0 or args.plate_min_vehicle_height <= 0:
            raise ValueError("minimum vehicle dimensions for plate detection must be positive")
        if not 0.0 <= args.plate_conf <= 1.0 or not 0.0 <= args.plate_iou <= 1.0:
            raise ValueError("plate-conf and plate-iou must be between 0 and 1")
        if not 0.0 <= args.plate_vehicle_padding <= 0.5:
            raise ValueError("plate-vehicle-padding must be between 0 and 0.5")
        if not 0.0 <= args.plate_search_start < 1.0:
            raise ValueError("plate-search-start must be in [0, 1)")
        if args.plate_min_aspect <= 0 or args.plate_max_aspect < args.plate_min_aspect:
            raise ValueError("invalid plate aspect-ratio range")
        if args.plate_cache_frames < 0:
            raise ValueError("plate-cache-frames cannot be negative")

    if args.ocr:
        if not args.plate_model:
            raise ValueError("--ocr requires --plate-model")
        positive_names = (
            "ocr_cpu_threads",
            "ocr_batch_size",
            "ocr_interval",
            "ocr_target_height",
            "ocr_max_width",
            "ocr_min_text_length",
            "ocr_max_text_length",
            "ocr_min_plate_width",
            "ocr_min_plate_height",
            "ocr_confirm_observations",
            "ocr_history_size",
        )
        if any(getattr(args, name) <= 0 for name in positive_names):
            raise ValueError("OCR dimensions, intervals, batch sizes, and history values must be positive")
        if args.ocr_max_reads_per_track < 0:
            raise ValueError("ocr-max-reads-per-track cannot be negative")
        if args.ocr_max_text_length < args.ocr_min_text_length:
            raise ValueError("ocr-max-text-length must be >= ocr-min-text-length")
        for name in ("ocr_min_score", "ocr_confirm_score", "ocr_confirm_dominance"):
            value = getattr(args, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
        if args.ocr_min_sharpness < 0:
            raise ValueError("ocr-min-sharpness cannot be negative")


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
