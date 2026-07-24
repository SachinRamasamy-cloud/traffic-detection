from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    size: int
    valid_width: int
    valid_height: int
    roi_ratio: float

    @property
    def x2(self) -> int:
        return self.x + self.valid_width

    @property
    def y2(self) -> int:
        return self.y + self.valid_height


@dataclass(frozen=True)
class Detection:
    xyxy: np.ndarray
    confidence: float
    class_id: int
    tile_index: int


@dataclass(frozen=True)
class TrackObservation:
    xyxy: np.ndarray
    track_id: int
    confidence: float
    class_id: int
    detection_index: int
    state: str = "tracked"
    prediction_gap: int = 0
