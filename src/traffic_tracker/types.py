from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    boundary_tile: bool = False

    @property
    def x2(self) -> int:
        return self.x + self.valid_width

    @property
    def y2(self) -> int:
        return self.y + self.valid_height

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "valid_width": self.valid_width,
            "valid_height": self.valid_height,
            "roi_ratio": float(self.roi_ratio),
            "boundary_tile": self.boundary_tile,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Tile":
        return cls(
            index=int(value["index"]),
            x=int(value["x"]),
            y=int(value["y"]),
            size=int(value["size"]),
            valid_width=int(value["valid_width"]),
            valid_height=int(value["valid_height"]),
            roi_ratio=float(value.get("roi_ratio", 0.0)),
            boundary_tile=bool(value.get("boundary_tile", False)),
        )


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


@dataclass(frozen=True)
class PlateDetection:
    """One plate detection associated with one tracked vehicle."""

    xyxy: np.ndarray
    confidence: float
    class_id: int
    vehicle_track_id: int
    vehicle_class_id: int
    vehicle_xyxy: np.ndarray
    frame_index: int
    search_region: str

    def to_record(self, crop_path: str | None = None) -> dict[str, Any]:
        x1, y1, x2, y2 = [float(value) for value in self.xyxy]
        vx1, vy1, vx2, vy2 = [float(value) for value in self.vehicle_xyxy]
        return {
            "frame_index": self.frame_index,
            "vehicle_track_id": self.vehicle_track_id,
            "vehicle_class_id": self.vehicle_class_id,
            "confidence": round(float(self.confidence), 6),
            "class_id": self.class_id,
            "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            "vehicle_bbox_xyxy": [round(vx1, 3), round(vy1, 3), round(vx2, 3), round(vy2, 3)],
            "width": round(x2 - x1, 3),
            "height": round(y2 - y1, 3),
            "search_region": self.search_region,
            "crop_path": crop_path,
        }
