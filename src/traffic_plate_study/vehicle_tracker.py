from __future__ import annotations

from typing import Any

import numpy as np

from traffic_plate_study.config import VehicleConfig
from traffic_plate_study.schemas import VehicleDetection


class YoloByteTracker:
    """Ultralytics YOLO detector with persistent ByteTrack state."""

    def __init__(self, config: VehicleConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install requirements-cpu.txt or requirements-gpu.txt."
            ) from exc

        self.config = config
        self.model = YOLO(config.model)

    def track(self, frame: np.ndarray) -> list[VehicleDetection]:
        kwargs: dict[str, Any] = {
            "source": frame,
            "persist": True,
            "tracker": self.config.tracker_file,
            "classes": list(self.config.class_ids),
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.image_size,
            "verbose": False,
        }
        if self.config.device is not None:
            kwargs["device"] = self.config.device

        result = self.model.track(**kwargs)[0]
        boxes = result.boxes
        if boxes is None or not boxes.is_track or boxes.id is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        track_ids = boxes.id.detach().cpu().numpy().astype(int)
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()

        detections: list[VehicleDetection] = []
        for coords, track_id, class_id, confidence in zip(
            xyxy, track_ids, class_ids, confidences, strict=True
        ):
            x1, y1, x2, y2 = (int(round(value)) for value in coords.tolist())
            class_name = str(result.names.get(int(class_id), class_id))
            detections.append(
                VehicleDetection(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
