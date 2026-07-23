from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_ms: float
    image: Any


@dataclass(frozen=True)
class VehicleDetection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class PlateObservation:
    frame_index: int
    timestamp_ms: float
    raw_text: str | None
    normalized_text: str | None
    text_valid: bool
    ocr_confidence: float
    detector_confidence: float
    quality_score: float
    vehicle_bbox: BBox
    plate_bbox: BBox
    plate_width: int
    plate_height: int
    blur_score: float
    vehicle_crop_path: str | None = None
    plate_crop_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": round(self.timestamp_ms, 3),
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "text_valid": self.text_valid,
            "ocr_confidence": round(self.ocr_confidence, 6),
            "detector_confidence": round(self.detector_confidence, 6),
            "quality_score": round(self.quality_score, 6),
            "vehicle_bbox": list(self.vehicle_bbox),
            "plate_bbox": list(self.plate_bbox),
            "plate_width": self.plate_width,
            "plate_height": self.plate_height,
            "blur_score": round(self.blur_score, 3),
            "vehicle_crop_path": self.vehicle_crop_path,
            "plate_crop_path": self.plate_crop_path,
        }


@dataclass(frozen=True)
class PlateConsensus:
    text: str | None
    accepted: bool
    confidence: float
    support_count: int
    observation_count: int
    exact_support_count: int
    alternatives: tuple[tuple[str, float], ...] = ()
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "accepted": self.accepted,
            "confidence": round(self.confidence, 6),
            "support_count": self.support_count,
            "observation_count": self.observation_count,
            "exact_support_count": self.exact_support_count,
            "alternatives": [
                {"text": text, "score": round(score, 6)} for text, score in self.alternatives
            ],
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class TrackAccumulator:
    track_id: int
    first_frame: int
    first_timestamp_ms: float
    last_frame: int
    last_timestamp_ms: float
    frame_count: int = 0
    plate_attempts: int = 0
    max_vehicle_confidence: float = 0.0
    class_votes: dict[str, float] = field(default_factory=dict)
    observations: list[PlateObservation] = field(default_factory=list)
    latest_bbox: BBox = (0, 0, 0, 0)
    best_quality: float = -1.0
    best_vehicle_crop_path: str | None = None
    best_plate_crop_path: str | None = None

    def update_vehicle(self, detection: VehicleDetection, frame_index: int, timestamp_ms: float) -> None:
        self.last_frame = frame_index
        self.last_timestamp_ms = timestamp_ms
        self.frame_count += 1
        self.latest_bbox = detection.bbox
        self.max_vehicle_confidence = max(self.max_vehicle_confidence, detection.confidence)
        self.class_votes[detection.class_name] = (
            self.class_votes.get(detection.class_name, 0.0) + detection.confidence
        )

    @property
    def vehicle_class(self) -> str:
        if not self.class_votes:
            return "unknown"
        return max(self.class_votes.items(), key=lambda item: item[1])[0]

    def add_observation(self, observation: PlateObservation, maximum: int) -> None:
        self.observations.append(observation)
        self.observations.sort(key=lambda item: item.quality_score, reverse=True)
        del self.observations[maximum:]
