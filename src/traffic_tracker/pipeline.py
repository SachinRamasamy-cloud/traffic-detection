from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2

from .detector import YoloTileDetector
from .drawing import draw_frame
from .exporter import ResultExporter
from .roi import load_roi
from .stabilization import TrackClassStabilizer
from .tiling import build_sparse_tiles
from .tracking import ByteTrackEngine
from .video import create_video_writer, read_video_info

LOGGER = logging.getLogger("traffic_tracker")


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def run(args) -> dict:
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = read_video_info(source)
    roi = load_roi(Path(args.roi).expanduser().resolve() if args.roi else None, info.width, info.height, args.curve_samples)
    if args.roi_rule:
        roi.rule = args.roi_rule
    if args.roi_min_overlap is not None:
        roi.minimum_intersection_ratio = args.roi_min_overlap

    tiles = build_sparse_tiles(
        info.width,
        info.height,
        roi,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        minimum_roi_ratio=args.min_tile_roi_ratio,
        max_tiles=args.max_tiles,
    )
    LOGGER.info("ROI selected %d sparse tiles", len(tiles))

    detector = YoloTileDetector(
        model_name=args.model,
        device="cpu",
        imgsz=args.imgsz,
        confidence=args.conf,
        detector_iou=args.iou,
        merge_iou=args.merge_iou,
        tile_batch_size=args.tile_batch_size,
        classes_raw=args.classes,
        class_agnostic_merge=not args.class_aware_merge,
        max_detections=args.max_detections,
        one_to_many=args.one_to_many,
        mask_outside_roi=args.mask_outside_roi,
        mask_dilation=args.mask_dilation,
        verbose=args.verbose,
    )
    tracker = ByteTrackEngine(Path(args.tracker), frame_rate=info.fps / args.vid_stride)
    stabilizer = TrackClassStabilizer(
        history_size=args.class_history,
        minimum_observations=args.class_min_observations,
        switch_ratio=args.class_switch_ratio,
        stale_after_frames=args.class_stale_frames,
    )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    video_path = output_dir / "tracked.mp4"
    writer = None if args.no_video else create_video_writer(video_path, info, args.vid_stride)
    processed = 0
    total_active = 0
    total_predicted = 0
    unique_ids: set[int] = set()
    started = time.perf_counter()
    source_frame_index = -1

    try:
        with ResultExporter(output_dir) as exporter:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                source_frame_index += 1
                if source_frame_index % args.vid_stride != 0:
                    continue

                detections = detector.detect(frame, tiles, roi)
                active = tracker.update(detections, frame)
                active_ids = {item.track_id for item in active}
                predicted = [
                    item
                    for item in tracker.predicted_lost(args.prediction_frames)
                    if item.track_id not in active_ids and roi.accepts_box(item.xyxy)
                ]
                records: list[dict] = []

                for observation in [*active, *predicted]:
                    raw_class_id = observation.class_id
                    if observation.state == "tracked":
                        stable_class_id = stabilizer.update(
                            observation.track_id,
                            raw_class_id,
                            observation.confidence,
                            source_frame_index,
                        )
                    else:
                        stable_class_id = stabilizer.get(observation.track_id, raw_class_id)
                    x1, y1, x2, y2 = [float(v) for v in observation.xyxy]
                    records.append(
                        {
                            "frame_index": source_frame_index,
                            "timestamp_seconds": round(source_frame_index / info.fps, 6),
                            "track_id": observation.track_id,
                            "state": observation.state,
                            "prediction_gap": observation.prediction_gap,
                            "raw_class_id": raw_class_id,
                            "raw_class_name": class_name(detector.names, raw_class_id),
                            "class_id": stable_class_id,
                            "class_name": class_name(detector.names, stable_class_id),
                            "class_stabilized": stable_class_id != raw_class_id,
                            "confidence": round(observation.confidence, 6),
                            "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                            "center_xy": [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)],
                            "width": round(x2 - x1, 3),
                            "height": round(y2 - y1, 3),
                        }
                    )
                    unique_ids.add(observation.track_id)

                timestamp = round(source_frame_index / info.fps, 6)
                exporter.write_frame(source_frame_index, timestamp, records)
                if writer is not None:
                    writer.write(
                        draw_frame(
                            frame,
                            records,
                            roi,
                            tiles,
                            draw_roi=args.draw_roi,
                            draw_tiles=args.draw_tiles,
                            line_width=args.line_width,
                        )
                    )

                processed += 1
                total_active += len(active)
                total_predicted += len(predicted)
                if processed % 25 == 0:
                    elapsed = time.perf_counter() - started
                    LOGGER.info(
                        "Processed %d frames | %.3f FPS | detections=%d | active_tracks=%d",
                        processed,
                        processed / max(elapsed, 1e-9),
                        len(detections),
                        len(active),
                    )
                stabilizer.prune(source_frame_index)
                if args.max_frames and processed >= args.max_frames:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "source": str(source),
        "model": args.model,
        "device": "cpu",
        "source_fps": info.fps,
        "source_dimensions": [info.width, info.height],
        "processed_frames": processed,
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(processed / max(elapsed, 1e-9), 4),
        "active_track_records": total_active,
        "predicted_track_records": total_predicted,
        "unique_track_ids": len(unique_ids),
        "roi": args.roi,
        "tiling": {
            "tile_size": args.tile_size,
            "overlap": args.tile_overlap,
            "tile_count": len(tiles),
            "batch_size": args.tile_batch_size,
            "merge_iou": args.merge_iou,
        },
        "kalman": {
            "provided_by": "Ultralytics BYTETracker",
            "prediction_export_frames": args.prediction_frames,
        },
        "outputs": {
            "annotated_video": None if args.no_video else str(video_path),
            "jsonl": str(output_dir / "tracks.jsonl"),
            "csv": str(output_dir / "tracks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
