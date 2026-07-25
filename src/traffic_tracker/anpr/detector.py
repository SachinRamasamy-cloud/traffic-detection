from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import PlateDetection, TrackObservation


@dataclass(frozen=True)
class VehicleCrop:
    track: TrackObservation
    image: np.ndarray
    origin_x: int
    origin_y: int
    region_name: str


class PlateDetector:
    """Detect number plates inside real, currently tracked vehicle crops."""

    def __init__(
        self,
        model_name: str,
        vehicle_names,
        vehicle_classes_raw: str,
        device: str = "cpu",
        imgsz: int = 640,
        confidence: float = 0.20,
        iou: float = 0.50,
        class_id: int = 0,
        interval: int = 2,
        batch_size: int = 2,
        minimum_vehicle_width: int = 96,
        minimum_vehicle_height: int = 64,
        vehicle_padding: float = 0.08,
        search_start_fraction: float = 0.25,
        search_full_vehicle: bool = False,
        minimum_plate_width: int = 8,
        minimum_plate_height: int = 4,
        minimum_aspect_ratio: float = 0.8,
        maximum_aspect_ratio: float = 10.0,
        verbose: bool = False,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.class_id = class_id
        self.interval = max(1, interval)
        self.batch_size = max(1, batch_size)
        self.minimum_vehicle_width = minimum_vehicle_width
        self.minimum_vehicle_height = minimum_vehicle_height
        self.vehicle_padding = vehicle_padding
        self.search_start_fraction = search_start_fraction
        self.search_full_vehicle = search_full_vehicle
        self.minimum_plate_width = minimum_plate_width
        self.minimum_plate_height = minimum_plate_height
        self.minimum_aspect_ratio = minimum_aspect_ratio
        self.maximum_aspect_ratio = maximum_aspect_ratio
        self.verbose = verbose
        self.vehicle_class_ids = _resolve_vehicle_classes(vehicle_classes_raw, vehicle_names)

    @property
    def names(self):
        return self.model.names

    def detect(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackObservation],
        frame_index: int,
    ) -> list[PlateDetection]:
        crops = self._prepare_crops(frame, tracks, frame_index)
        if not crops:
            return []

        output: list[PlateDetection] = []
        for offset in range(0, len(crops), self.batch_size):
            batch = crops[offset : offset + self.batch_size]
            results = self.model.predict(
                source=[item.image for item in batch],
                device=self.device,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                classes=None if self.class_id < 0 else [self.class_id],
                verbose=self.verbose,
            )
            for crop, result in zip(batch, results):
                best = self._best_detection(crop, result, frame.shape[1], frame.shape[0], frame_index)
                if best is not None:
                    output.append(best)
        return output

    def _prepare_crops(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackObservation],
        frame_index: int,
    ) -> list[VehicleCrop]:
        frame_height, frame_width = frame.shape[:2]
        output: list[VehicleCrop] = []

        for track in tracks:
            if track.state != "tracked":
                continue
            if self.vehicle_class_ids is not None and track.class_id not in self.vehicle_class_ids:
                continue
            if (frame_index + track.track_id) % self.interval != 0:
                continue

            x1, y1, x2, y2 = [float(value) for value in track.xyxy]
            width = x2 - x1
            height = y2 - y1
            if width < self.minimum_vehicle_width or height < self.minimum_vehicle_height:
                continue

            pad_x = width * self.vehicle_padding
            pad_y = height * self.vehicle_padding
            crop_x1 = max(0, int(np.floor(x1 - pad_x)))
            crop_y1 = max(0, int(np.floor(y1 - pad_y)))
            crop_x2 = min(frame_width, int(np.ceil(x2 + pad_x)))
            crop_y2 = min(frame_height, int(np.ceil(y2 + pad_y)))
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            region_name = "full_vehicle"
            if not self.search_full_vehicle:
                vehicle_height = crop_y2 - crop_y1
                lower_offset = int(round(vehicle_height * self.search_start_fraction))
                crop_y1 = min(crop_y2 - 1, crop_y1 + lower_offset)
                region_name = f"lower_{1.0 - self.search_start_fraction:.2f}"

            image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if image.size == 0:
                continue
            output.append(
                VehicleCrop(
                    track=track,
                    image=image,
                    origin_x=crop_x1,
                    origin_y=crop_y1,
                    region_name=region_name,
                )
            )
        return output

    def _best_detection(
        self,
        crop: VehicleCrop,
        result,
        frame_width: int,
        frame_height: int,
        frame_index: int,
    ) -> PlateDetection | None:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
        order = np.argsort(confs)[::-1]

        for index in order:
            box = xyxy[int(index)].copy()
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            if width < self.minimum_plate_width or height < self.minimum_plate_height:
                continue
            aspect = width / max(height, 1e-6)
            if not self.minimum_aspect_ratio <= aspect <= self.maximum_aspect_ratio:
                continue

            box[[0, 2]] += crop.origin_x
            box[[1, 3]] += crop.origin_y
            box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_height - 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            return PlateDetection(
                xyxy=box,
                confidence=float(confs[int(index)]),
                class_id=int(classes[int(index)]),
                vehicle_track_id=crop.track.track_id,
                vehicle_class_id=crop.track.class_id,
                vehicle_xyxy=np.asarray(crop.track.xyxy, dtype=np.float32).copy(),
                frame_index=frame_index,
                search_region=crop.region_name,
            )
        return None


def _resolve_vehicle_classes(raw: str, names) -> set[int] | None:
    if not raw.strip():
        return None
    mapping = names if isinstance(names, dict) else dict(enumerate(names))
    reverse = {str(name).lower(): int(index) for index, name in mapping.items()}
    output: set[int] = set()
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token.lstrip("-").isdigit():
            class_id = int(token)
            if class_id not in mapping:
                raise ValueError(f"Unknown vehicle class ID for plate detection: {class_id}")
        else:
            class_id = reverse.get(token.lower(), -1)
            if class_id < 0:
                raise ValueError(f"Unknown vehicle class name for plate detection: {token}")
        output.add(class_id)
    return output
