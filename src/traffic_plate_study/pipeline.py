from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from traffic_plate_study import __version__
from traffic_plate_study.alpr import FastAlprEngine, candidate_to_observation
from traffic_plate_study.config import AppConfig
from traffic_plate_study.consensus import build_consensus
from traffic_plate_study.drawing import draw_plate_bbox, draw_vehicle
from traffic_plate_study.geometry import (
    apply_roi_mask,
    bottom_center,
    crop_image,
    expand_bbox,
    laplacian_blur_score,
    point_in_polygon,
)
from traffic_plate_study.schemas import TrackAccumulator, VehicleDetection
from traffic_plate_study.vehicle_tracker import YoloByteTracker
from traffic_plate_study.video import VideoReader


class TrafficPlatePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.vehicle_tracker = YoloByteTracker(config.vehicle)
        self.alpr = FastAlprEngine(config.plate)
        self.tracks: dict[int, TrackAccumulator] = {}

    def run(
        self,
        video_path: str | Path,
        output_path: str | Path,
        annotated_video_path: str | Path | None = None,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        asset_dir = output_path.parent / f"{output_path.stem}_assets"
        if self.config.output.save_best_crops:
            asset_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc)
        wall_start = time.perf_counter()
        processed_frames = 0
        plate_detections_this_frame: dict[int, tuple[tuple[int, int, int, int], str | None]] = {}

        with VideoReader(video_path) as reader:
            writer = self._create_writer(annotated_video_path, reader.metadata)
            try:
                for packet in reader:
                    if max_frames is not None and processed_frames >= max_frames:
                        break
                    frame = packet.image
                    inference_frame = apply_roi_mask(frame, self.config.roi.polygon)
                    detections = self.vehicle_tracker.track(inference_frame)
                    detections = self._filter_roi(detections)
                    plate_detections_this_frame.clear()

                    for detection in detections:
                        track = self._get_or_create_track(detection, packet.frame_index, packet.timestamp_ms)
                        track.update_vehicle(detection, packet.frame_index, packet.timestamp_ms)
                        observation = self._maybe_read_plate(
                            frame=frame,
                            detection=detection,
                            track=track,
                            frame_index=packet.frame_index,
                            timestamp_ms=packet.timestamp_ms,
                            asset_dir=asset_dir,
                            json_parent=output_path.parent,
                        )
                        if observation is not None:
                            plate_detections_this_frame[detection.track_id] = (
                                observation.plate_bbox,
                                observation.normalized_text,
                            )

                    if writer is not None:
                        annotated = frame.copy()
                        for detection in detections:
                            track = self.tracks[detection.track_id]
                            consensus = build_consensus(track.observations, self.config.consensus)
                            draw_vehicle(annotated, detection, consensus)
                            plate_info = plate_detections_this_frame.get(detection.track_id)
                            if plate_info is not None:
                                draw_plate_bbox(annotated, plate_info[0], plate_info[1])
                        writer.write(annotated)

                    processed_frames += 1
                    if processed_frames % 100 == 0:
                        print(
                            f"Processed {processed_frames} frames | "
                            f"tracks={len(self.tracks)}",
                            flush=True,
                        )
            finally:
                if writer is not None:
                    writer.release()

            result = self._build_result(
                video_metadata=reader.metadata,
                processed_frames=processed_frames,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - wall_start,
            )

        self._atomic_write_json(output_path, result)
        return result

    def _get_or_create_track(
        self,
        detection: VehicleDetection,
        frame_index: int,
        timestamp_ms: float,
    ) -> TrackAccumulator:
        if detection.track_id not in self.tracks:
            self.tracks[detection.track_id] = TrackAccumulator(
                track_id=detection.track_id,
                first_frame=frame_index,
                first_timestamp_ms=timestamp_ms,
                last_frame=frame_index,
                last_timestamp_ms=timestamp_ms,
            )
        return self.tracks[detection.track_id]

    def _filter_roi(self, detections: list[VehicleDetection]) -> list[VehicleDetection]:
        if not self.config.roi.polygon:
            return detections
        return [
            detection
            for detection in detections
            if point_in_polygon(bottom_center(detection.bbox), self.config.roi.polygon)
        ]

    def _maybe_read_plate(
        self,
        frame,
        detection: VehicleDetection,
        track: TrackAccumulator,
        frame_index: int,
        timestamp_ms: float,
        asset_dir: Path,
        json_parent: Path,
    ):
        plate_config = self.config.plate
        if frame_index % plate_config.every_n_frames != 0:
            return None
        if track.plate_attempts >= plate_config.max_attempts_per_track:
            return None
        height, width = frame.shape[:2]
        vehicle_bbox = expand_bbox(
            detection.bbox,
            self.config.vehicle.crop_expansion,
            width,
            height,
        )
        vehicle_crop = crop_image(frame, vehicle_bbox)
        crop_height, crop_width = vehicle_crop.shape[:2]
        if (
            crop_width < self.config.vehicle.min_crop_width
            or crop_height < self.config.vehicle.min_crop_height
        ):
            return None

        vehicle_blur = laplacian_blur_score(vehicle_crop)
        if vehicle_blur < self.config.vehicle.min_crop_blur_score:
            return None

        track.plate_attempts += 1
        candidates = self.alpr.predict(vehicle_crop)
        if not candidates:
            return None
        candidate = candidates[0]

        vehicle_crop_path: str | None = None
        plate_crop_path: str | None = None
        if self.config.output.save_best_crops and candidate.quality_score > track.best_quality:
            vehicle_file = asset_dir / f"track_{track.track_id:06d}_vehicle.jpg"
            plate_file = asset_dir / f"track_{track.track_id:06d}_plate.jpg"
            cv2.imwrite(str(vehicle_file), vehicle_crop)
            cv2.imwrite(str(plate_file), candidate.crop)
            vehicle_crop_path = os.path.relpath(vehicle_file, json_parent).replace(os.sep, "/")
            plate_crop_path = os.path.relpath(plate_file, json_parent).replace(os.sep, "/")
            track.best_quality = candidate.quality_score
            track.best_vehicle_crop_path = vehicle_crop_path
            track.best_plate_crop_path = plate_crop_path
        else:
            vehicle_crop_path = track.best_vehicle_crop_path
            plate_crop_path = track.best_plate_crop_path

        observation = candidate_to_observation(
            candidate=candidate,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            vehicle_bbox=vehicle_bbox,
            vehicle_crop_path=vehicle_crop_path,
            plate_crop_path=plate_crop_path,
        )
        track.add_observation(observation, plate_config.max_observations_per_track)
        return observation

    def _create_writer(self, path: str | Path | None, metadata):
        if path is None:
            return None
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        codec = self.config.output.video_codec
        if len(codec) != 4:
            raise ValueError("output.video_codec must contain exactly four characters")
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*codec),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open annotated video writer: {output}")
        return writer

    def _build_result(
        self,
        video_metadata,
        processed_frames: int,
        started_at: datetime,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        vehicles: list[dict[str, Any]] = []
        confirmed_count = 0
        accepted_plate_count = 0

        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            confirmed = track.frame_count >= self.config.vehicle.minimum_track_frames
            consensus = build_consensus(track.observations, self.config.consensus)
            if confirmed:
                confirmed_count += 1
            if consensus.accepted:
                accepted_plate_count += 1

            vehicle_result: dict[str, Any] = {
                "track_id": track.track_id,
                "vehicle_class": track.vehicle_class,
                "confirmed": confirmed,
                "frame_count": track.frame_count,
                "max_vehicle_confidence": round(track.max_vehicle_confidence, 6),
                "first_seen": {
                    "frame_index": track.first_frame,
                    "timestamp_ms": round(track.first_timestamp_ms, 3),
                },
                "last_seen": {
                    "frame_index": track.last_frame,
                    "timestamp_ms": round(track.last_timestamp_ms, 3),
                },
                "duration_ms": round(track.last_timestamp_ms - track.first_timestamp_ms, 3),
                "latest_bbox": list(track.latest_bbox),
                "plate_attempts": track.plate_attempts,
                "plate": consensus.to_dict(),
                "best_vehicle_crop_path": track.best_vehicle_crop_path,
                "best_plate_crop_path": track.best_plate_crop_path,
            }
            if self.config.output.include_all_plate_observations:
                vehicle_result["plate_observations"] = [
                    observation.to_dict() for observation in track.observations
                ]
            vehicles.append(vehicle_result)

        return {
            "schema_version": "1.0",
            "pipeline_version": __version__,
            "study_only": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "video": {
                "path": video_metadata.path,
                "width": video_metadata.width,
                "height": video_metadata.height,
                "fps": round(video_metadata.fps, 6),
                "reported_frame_count": video_metadata.reported_frame_count,
                "reported_duration_ms": round(video_metadata.reported_duration_ms, 3),
                "processed_frames": processed_frames,
            },
            "models": {
                "vehicle_detector": self.config.vehicle.model,
                "vehicle_tracker": "bytetrack",
                "vehicle_tracker_config": self.config.vehicle.tracker_file,
                "plate_detector": self.config.plate.detector_model,
                "plate_ocr": self.config.plate.ocr_model,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "average_processing_fps": round(
                    processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                    3,
                ),
            },
            "configuration": asdict(self.config),
            "summary": {
                "vehicle_tracks": len(vehicles),
                "confirmed_vehicle_tracks": confirmed_count,
                "tracks_with_any_plate_detection": sum(
                    bool(track.observations) for track in self.tracks.values()
                ),
                "tracks_with_accepted_plate": accepted_plate_count,
            },
            "vehicles": vehicles,
        }

    def _atomic_write_json(self, output_path: Path, result: dict[str, Any]) -> None:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                result,
                handle,
                indent=self.config.output.json_indent,
                ensure_ascii=False,
            )
            handle.write("\n")
        temporary.replace(output_path)
