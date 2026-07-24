#!/usr/bin/env python3
"""CPU-oriented YOLO26 + ByteTrack video tracking.

Outputs:
- annotated MP4 video
- frame-by-frame JSONL tracking records
- CSV tracking table
- run summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Limit CPU thread oversubscription before importing torch/ultralytics.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 1)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

import cv2
import torch

# Disable NNPACK because this CPU does not support it.
# PyTorch will use another available CPU implementation.
if hasattr(torch.backends, "nnpack"):
    torch.backends.nnpack.set_flags(False)

from ultralytics import YOLO

LOGGER = logging.getLogger("yolo26_cpu_tracker")


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track objects in a video with YOLO26 and ByteTrack on CPU."
    )
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO26 weights path or model name. Default: yolo26n.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/yolo26_track",
        help="Directory for video, JSONL, CSV, and summary outputs.",
    )
    parser.add_argument(
        "--tracker",
        default=str(Path(__file__).with_name("bytetrack_cpu.yaml")),
        help="Tracker YAML path. Default: bundled ByteTrack CPU config.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Detector input size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="Detection IoU threshold.")
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated COCO class names or IDs, e.g. car,motorcycle,bus,truck. Empty means all classes.",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Process every Nth frame. Keep 1 for best ID continuity.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many processed frames. 0 means full video.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding-box line width in the annotated video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not write the annotated MP4; still write JSONL/CSV/summary.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="PyTorch CPU thread count.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1")
    if args.vid_stride <= 0:
        raise ValueError("--vid-stride must be at least 1")
    if args.threads <= 0:
        raise ValueError("--threads must be at least 1")
    return source


def read_video_info(source: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps <= 0 or not (fps < 1000):
        LOGGER.warning("Invalid source FPS %.3f; using 25 FPS for output timestamps.", fps)
        fps = 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine video width/height")
    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)


def resolve_classes(raw: str, names: dict[int, str] | list[str]) -> list[int] | None:
    if not raw.strip():
        return None

    name_map = names if isinstance(names, dict) else dict(enumerate(names))
    reverse = {str(name).lower(): int(idx) for idx, name in name_map.items()}
    resolved: list[int] = []

    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        if token.lstrip("-").isdigit():
            class_id = int(token)
            if class_id not in name_map:
                raise ValueError(f"Unknown class ID {class_id}. Valid IDs: 0..{max(name_map)}")
        else:
            class_id = reverse.get(token.lower(), -1)
            if class_id < 0:
                available = ", ".join(sorted(reverse))
                raise ValueError(f"Unknown class name '{token}'. Available names: {available}")
        if class_id not in resolved:
            resolved.append(class_id)
    return resolved or None


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def iter_records(result: Any, frame_index: int, source_fps: float, names: Any) -> Iterable[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().to(torch.int64).tolist()
    track_ids = (
        boxes.id.detach().cpu().to(torch.int64).tolist()
        if boxes.id is not None
        else [None] * len(xyxy)
    )

    timestamp_seconds = frame_index / source_fps
    records: list[dict[str, Any]] = []
    for coords, confidence, class_id, track_id in zip(xyxy, confs, classes, track_ids):
        x1, y1, x2, y2 = (float(value) for value in coords)
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "track_id": int(track_id) if track_id is not None else None,
                "class_id": int(class_id),
                "class_name": class_name(names, int(class_id)),
                "confidence": round(float(confidence), 6),
                "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                "center_xy": [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)],
                "width": round(x2 - x1, 3),
                "height": round(y2 - y1, 3),
            }
        )
    return records


def create_video_writer(path: Path, info: VideoInfo, vid_stride: int) -> cv2.VideoWriter:
    output_fps = info.fps / vid_stride
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        source = validate_args(args)
        info = read_video_info(source)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, args.threads)))
        except RuntimeError:
            # Can only be set before parallel work starts; non-fatal if already initialized.
            pass

        LOGGER.info("Loading model %s on CPU", args.model)
        model = YOLO(args.model)
        selected_classes = resolve_classes(args.classes, model.names)

        video_path = output_dir / "tracked.mp4"
        jsonl_path = output_dir / "tracks.jsonl"
        csv_path = output_dir / "tracks.csv"
        summary_path = output_dir / "summary.json"

        writer = None if args.no_video else create_video_writer(video_path, info, args.vid_stride)
        csv_fields = [
            "frame_index",
            "timestamp_seconds",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "width",
            "height",
        ]

        processed_frames = 0
        total_records = 0
        unique_track_ids: set[int] = set()
        started = time.perf_counter()

        try:
            with jsonl_path.open("w", encoding="utf-8") as jsonl_file, csv_path.open(
                "w", newline="", encoding="utf-8"
            ) as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
                csv_writer.writeheader()

                results = model.track(
                    source=str(source),
                    stream=True,
                    persist=False,
                    tracker=args.tracker,
                    device="cpu",
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    classes=selected_classes,
                    vid_stride=args.vid_stride,
                    verbose=args.verbose,
                    # half=False,
                )

                for result in results:
                    source_frame_index = processed_frames * args.vid_stride
                    records = iter_records(result, source_frame_index, info.fps, model.names)

                    frame_payload = {
                        "frame_index": source_frame_index,
                        "timestamp_seconds": round(source_frame_index / info.fps, 6),
                        "objects": records,
                    }
                    jsonl_file.write(json.dumps(frame_payload, ensure_ascii=False) + "\n")

                    for record in records:
                        if record["track_id"] is not None:
                            unique_track_ids.add(int(record["track_id"]))
                        x1, y1, x2, y2 = record["bbox_xyxy"]
                        center_x, center_y = record["center_xy"]
                        csv_writer.writerow(
                            {
                                "frame_index": record["frame_index"],
                                "timestamp_seconds": record["timestamp_seconds"],
                                "track_id": record["track_id"],
                                "class_id": record["class_id"],
                                "class_name": record["class_name"],
                                "confidence": record["confidence"],
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "center_x": center_x,
                                "center_y": center_y,
                                "width": record["width"],
                                "height": record["height"],
                            }
                        )

                    if writer is not None:
                        annotated = result.plot(line_width=args.line_width)
                        if annotated.shape[1] != info.width or annotated.shape[0] != info.height:
                            annotated = cv2.resize(annotated, (info.width, info.height))
                        writer.write(annotated)

                    processed_frames += 1
                    total_records += len(records)

                    if processed_frames % 100 == 0:
                        elapsed = time.perf_counter() - started
                        LOGGER.info(
                            "Processed %d frames | %.2f FPS | %d active/observed IDs",
                            processed_frames,
                            processed_frames / max(elapsed, 1e-9),
                            len(unique_track_ids),
                        )

                    if args.max_frames and processed_frames >= args.max_frames:
                        break
        finally:
            if writer is not None:
                writer.release()

        elapsed = time.perf_counter() - started
        summary = {
            "source": str(source),
            "model": args.model,
            "tracker": args.tracker,
            "device": "cpu",
            "source_fps": info.fps,
            "source_width": info.width,
            "source_height": info.height,
            "source_frame_count": info.frame_count,
            "processed_frames": processed_frames,
            "vid_stride": args.vid_stride,
            "elapsed_seconds": round(elapsed, 3),
            "processing_fps": round(processed_frames / max(elapsed, 1e-9), 3),
            "object_records": total_records,
            "unique_track_ids": len(unique_track_ids),
            "class_filter": selected_classes,
            "outputs": {
                "annotated_video": None if args.no_video else str(video_path),
                "jsonl": str(jsonl_path),
                "csv": str(csv_path),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        LOGGER.info("Completed: %s", json.dumps(summary, indent=2))
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.exception("Tracking failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
