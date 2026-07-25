from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .anpr import PlateDetector, PlateMemory
from .detector import YoloTileDetector
from .drawing import draw_frame
from .exporter import ResultExporter
from .roi import load_roi
from .stabilization import TrackClassStabilizer
from .tile_plan import load_or_build_tile_plan, save_tile_preview
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

    roi = load_roi(
        Path(args.roi).expanduser().resolve() if args.roi else None,
        info.width,
        info.height,
        args.curve_samples,
    )
    if args.roi_rule:
        roi.rule = args.roi_rule
    if args.roi_min_overlap is not None:
        roi.minimum_intersection_ratio = args.roi_min_overlap

    tile_plan_path = (
        Path(args.tile_plan).expanduser().resolve()
        if args.tile_plan
        else output_dir / "tile_plan.json"
    )
    tile_plan = load_or_build_tile_plan(
        path=tile_plan_path,
        frame_width=info.width,
        frame_height=info.height,
        roi=roi,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        minimum_roi_ratio=args.min_tile_roi_ratio,
        max_tiles=args.max_tiles,
        roi_padding=args.roi_tile_padding,
        force_boundary_tiles=args.force_boundary_tiles,
        rebuild=args.rebuild_tile_plan,
    )
    tiles = list(tile_plan.tiles)

    # Both masks are generated once. The static coordinates and masks are reused
    # for all frames; there is no per-frame tile generation.
    tile_mask = roi.dilated_mask(args.roi_tile_padding)
    acceptance_mask = roi.dilated_mask(args.roi_detection_padding)

    if args.save_tile_preview:
        preview_path = output_dir / "tile_plan_preview.jpg"
        save_tile_preview(
            source=source,
            frame_index=args.tile_plan_frame_index,
            output_path=preview_path,
            roi=roi,
            tiles=tiles,
        )
        LOGGER.info("Saved tile preview to %s", preview_path)

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

    plate_detector = None
    plate_memory = PlateMemory(cache_frames=args.plate_cache_frames, stale_frames=args.class_stale_frames)
    if args.plate_model:
        LOGGER.info("Loading number-plate model %s on CPU", args.plate_model)
        plate_detector = PlateDetector(
            model_name=args.plate_model,
            vehicle_names=detector.names,
            vehicle_classes_raw=args.plate_vehicle_classes,
            device="cpu",
            imgsz=args.plate_imgsz,
            confidence=args.plate_conf,
            iou=args.plate_iou,
            class_id=args.plate_class_id,
            interval=args.plate_interval,
            batch_size=args.plate_batch_size,
            minimum_vehicle_width=args.plate_min_vehicle_width,
            minimum_vehicle_height=args.plate_min_vehicle_height,
            vehicle_padding=args.plate_vehicle_padding,
            search_start_fraction=args.plate_search_start,
            search_full_vehicle=args.plate_search_full_vehicle,
            minimum_plate_width=args.plate_min_width,
            minimum_plate_height=args.plate_min_height,
            minimum_aspect_ratio=args.plate_min_aspect,
            maximum_aspect_ratio=args.plate_max_aspect,
            verbose=args.verbose,
        )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    video_path = output_dir / "tracked.mp4"
    writer = None if args.no_video else create_video_writer(video_path, info, args.vid_stride)
    plate_crop_root = output_dir / "plate_crops"
    if args.save_plate_crops and plate_detector is not None:
        plate_crop_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    total_active = 0
    total_predicted = 0
    total_plate_detections = 0
    saved_plate_crops = 0
    unique_ids: set[int] = set()
    plate_track_ids: set[int] = set()
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

                detections = detector.detect(
                    frame,
                    tiles,
                    roi,
                    tile_mask=tile_mask,
                    acceptance_mask=acceptance_mask,
                )
                active = tracker.update(detections, frame)
                active_ids = {item.track_id for item in active}
                predicted = [
                    item
                    for item in tracker.predicted_lost(args.prediction_frames)
                    if item.track_id not in active_ids
                    and roi.accepts_box(item.xyxy, mask=acceptance_mask)
                ]

                current_plates = (
                    plate_detector.detect(frame, active, source_frame_index)
                    if plate_detector is not None
                    else []
                )
                plate_memory.update(current_plates)
                total_plate_detections += len(current_plates)
                plate_track_ids.update(item.vehicle_track_id for item in current_plates)

                plate_records: list[dict] = []
                for plate in current_plates:
                    crop_path = None
                    should_save = args.save_plate_crops and (
                        not args.save_only_best_plate_crop or plate_memory.is_new_best(plate)
                    )
                    if should_save:
                        crop_path = _save_plate_crop(
                            frame=frame,
                            plate=plate,
                            root=plate_crop_root,
                        )
                        if crop_path is not None:
                            saved_plate_crops += 1
                    plate_records.append(plate.to_record(crop_path=crop_path))

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

                    x1, y1, x2, y2 = [float(value) for value in observation.xyxy]
                    plate_item = plate_memory.get(
                        observation.track_id,
                        source_frame_index,
                        current_vehicle_xyxy=observation.xyxy,
                    )
                    plate_payload = None
                    if plate_item is not None:
                        plate = plate_item.detection
                        px1, py1, px2, py2 = [float(value) for value in plate.xyxy]
                        plate_payload = {
                            "state": "current" if plate_item.is_current else "cached",
                            "age_frames": plate_item.age_frames,
                            "confidence": round(plate.confidence, 6),
                            "class_id": plate.class_id,
                            "bbox_xyxy": [round(px1, 3), round(py1, 3), round(px2, 3), round(py2, 3)],
                            "width": round(px2 - px1, 3),
                            "height": round(py2 - py1, 3),
                        }

                    records.append(
                        {
                            "frame_index": source_frame_index,
                            "timestamp_seconds": round(source_frame_index / info.fps, 6),
                            "track_id": observation.track_id,
                            "state": observation.state,
                            "prediction_gap": observation.prediction_gap,
                            "inside_exact_roi": roi.accepts_box(observation.xyxy),
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
                            "plate": plate_payload,
                        }
                    )
                    unique_ids.add(observation.track_id)

                timestamp = round(source_frame_index / info.fps, 6)
                exporter.write_frame(
                    source_frame_index,
                    timestamp,
                    records,
                    plate_records=plate_records,
                )

                if writer is not None:
                    writer.write(
                        draw_frame(
                            frame,
                            records,
                            roi,
                            tiles,
                            draw_roi=args.draw_roi,
                            draw_tiles=args.draw_tiles,
                            draw_plates=args.draw_plates,
                            line_width=args.line_width,
                        )
                    )

                processed += 1
                total_active += len(active)
                total_predicted += len(predicted)
                if processed % 25 == 0:
                    elapsed = time.perf_counter() - started
                    LOGGER.info(
                        "Processed %d frames | %.3f FPS | vehicle_detections=%d | "
                        "active_tracks=%d | plates=%d",
                        processed,
                        processed / max(elapsed, 1e-9),
                        len(detections),
                        len(active),
                        len(current_plates),
                    )

                stabilizer.prune(source_frame_index)
                plate_memory.prune(source_frame_index)
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
            "mode": "static_coordinate_plan_reused_for_all_frames",
            "tile_plan": str(tile_plan_path),
            "tile_size": args.tile_size,
            "overlap": args.tile_overlap,
            "tile_count": len(tiles),
            "boundary_tile_count": sum(1 for tile in tiles if tile.boundary_tile),
            "batch_size": args.tile_batch_size,
            "merge_iou": args.merge_iou,
            "roi_tile_padding": args.roi_tile_padding,
            "roi_detection_padding": args.roi_detection_padding,
        },
        "kalman": {
            "provided_by": "Ultralytics BYTETracker",
            "prediction_export_frames": args.prediction_frames,
        },
        "number_plate_detection": {
            "enabled": plate_detector is not None,
            "model": args.plate_model or None,
            "detections": total_plate_detections,
            "vehicle_tracks_with_plate": len(plate_track_ids),
            "saved_plate_crops": saved_plate_crops,
            "interval": args.plate_interval if plate_detector is not None else None,
            "ocr_enabled": False,
        },
        "outputs": {
            "annotated_video": None if args.no_video else str(video_path),
            "jsonl": str(output_dir / "tracks.jsonl"),
            "csv": str(output_dir / "tracks.csv"),
            "plates_jsonl": str(output_dir / "plates.jsonl"),
            "plates_csv": str(output_dir / "plates.csv"),
            "plate_crops": str(plate_crop_root) if plate_detector is not None and args.save_plate_crops else None,
            "tile_plan": str(tile_plan_path),
            "tile_preview": str(output_dir / "tile_plan_preview.jpg") if args.save_tile_preview else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _save_plate_crop(frame: np.ndarray, plate, root: Path) -> str | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in plate.xyxy]
    width = x2 - x1
    height = y2 - y1
    pad_x = max(2.0, width * 0.08)
    pad_y = max(2.0, height * 0.15)
    ix1 = max(0, int(np.floor(x1 - pad_x)))
    iy1 = max(0, int(np.floor(y1 - pad_y)))
    ix2 = min(frame_width, int(np.ceil(x2 + pad_x)))
    iy2 = min(frame_height, int(np.ceil(y2 + pad_y)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None

    track_dir = root / f"track_{plate.vehicle_track_id:06d}"
    track_dir.mkdir(parents=True, exist_ok=True)
    path = track_dir / f"frame_{plate.frame_index:08d}_conf_{plate.confidence:.3f}.jpg"
    if not cv2.imwrite(str(path), crop):
        LOGGER.warning("Could not save plate crop: %s", path)
        return None
    return str(path)
