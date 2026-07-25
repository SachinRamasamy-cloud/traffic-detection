from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import PlateDetection


@dataclass(frozen=True)
class PlateMemoryItem:
    detection: PlateDetection
    age_frames: int
    is_current: bool


class PlateMemory:
    """Cache recent plate boxes between scheduled plate-detector frames.

    Cached plate boxes are projected through the vehicle box motion, so the
    visualization does not remain frozen at the previous frame location.
    """

    def __init__(self, cache_frames: int = 10, stale_frames: int = 300) -> None:
        self.cache_frames = max(0, cache_frames)
        self.stale_frames = max(self.cache_frames, stale_frames)
        self.latest: dict[int, PlateDetection] = {}
        self.best: dict[int, PlateDetection] = {}

    def update(self, detections: list[PlateDetection]) -> None:
        for detection in detections:
            track_id = detection.vehicle_track_id
            self.latest[track_id] = detection
            current_best = self.best.get(track_id)
            if current_best is None or detection.confidence > current_best.confidence:
                self.best[track_id] = detection

    def get(
        self,
        track_id: int,
        frame_index: int,
        current_vehicle_xyxy: np.ndarray | None = None,
    ) -> PlateMemoryItem | None:
        detection = self.latest.get(track_id)
        if detection is None:
            return None
        age = frame_index - detection.frame_index
        if age < 0 or age > self.cache_frames:
            return None

        projected = detection
        if age > 0 and current_vehicle_xyxy is not None:
            projected = _project_to_current_vehicle(detection, current_vehicle_xyxy)
        return PlateMemoryItem(detection=projected, age_frames=age, is_current=age == 0)

    def is_new_best(self, detection: PlateDetection) -> bool:
        best = self.best.get(detection.vehicle_track_id)
        return best is detection

    def prune(self, frame_index: int) -> None:
        stale_ids = [
            track_id
            for track_id, detection in self.latest.items()
            if frame_index - detection.frame_index > self.stale_frames
        ]
        for track_id in stale_ids:
            self.latest.pop(track_id, None)
            self.best.pop(track_id, None)


def _project_to_current_vehicle(
    detection: PlateDetection,
    current_vehicle_xyxy: np.ndarray,
) -> PlateDetection:
    old_vehicle = np.asarray(detection.vehicle_xyxy, dtype=np.float32)
    new_vehicle = np.asarray(current_vehicle_xyxy, dtype=np.float32)
    plate = np.asarray(detection.xyxy, dtype=np.float32)

    old_width = max(float(old_vehicle[2] - old_vehicle[0]), 1e-6)
    old_height = max(float(old_vehicle[3] - old_vehicle[1]), 1e-6)
    new_width = max(float(new_vehicle[2] - new_vehicle[0]), 1e-6)
    new_height = max(float(new_vehicle[3] - new_vehicle[1]), 1e-6)

    relative = np.asarray(
        [
            (plate[0] - old_vehicle[0]) / old_width,
            (plate[1] - old_vehicle[1]) / old_height,
            (plate[2] - old_vehicle[0]) / old_width,
            (plate[3] - old_vehicle[1]) / old_height,
        ],
        dtype=np.float32,
    )
    projected = np.asarray(
        [
            new_vehicle[0] + relative[0] * new_width,
            new_vehicle[1] + relative[1] * new_height,
            new_vehicle[0] + relative[2] * new_width,
            new_vehicle[1] + relative[3] * new_height,
        ],
        dtype=np.float32,
    )
    return PlateDetection(
        xyxy=projected,
        confidence=detection.confidence,
        class_id=detection.class_id,
        vehicle_track_id=detection.vehicle_track_id,
        vehicle_class_id=detection.vehicle_class_id,
        vehicle_xyxy=new_vehicle.copy(),
        frame_index=detection.frame_index,
        search_region=detection.search_region,
    )
