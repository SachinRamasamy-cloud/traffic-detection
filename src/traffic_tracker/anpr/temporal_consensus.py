from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .ocr_engine import OCRRead


@dataclass(frozen=True)
class PlateTextResult:
    vehicle_track_id: int
    text: str
    status: str
    confidence: float
    weighted_score: float
    observation_count: int
    total_accepted_observations: int
    dominance: float
    first_frame: int
    last_frame: int

    def to_record(self) -> dict[str, Any]:
        return {
            "vehicle_track_id": self.vehicle_track_id,
            "plate_text": self.text,
            "status": self.status,
            "confidence": round(float(self.confidence), 6),
            "weighted_score": round(float(self.weighted_score), 6),
            "observation_count": self.observation_count,
            "total_accepted_observations": self.total_accepted_observations,
            "dominance": round(float(self.dominance), 6),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
        }


class PlateTextConsensus:
    """Confidence-weighted exact-text voting for OCR reads per vehicle track."""

    def __init__(
        self,
        minimum_observations: int = 3,
        minimum_confirm_confidence: float = 0.50,
        minimum_dominance: float = 0.60,
        history_size: int = 30,
    ) -> None:
        self.minimum_observations = max(1, minimum_observations)
        self.minimum_confirm_confidence = float(minimum_confirm_confidence)
        self.minimum_dominance = float(minimum_dominance)
        self.history_size = max(self.minimum_observations, history_size)
        self.histories: dict[int, deque[OCRRead]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.results: dict[int, PlateTextResult] = {}

    def update(self, read: OCRRead | None) -> PlateTextResult | None:
        if read is None:
            return None
        track_id = read.vehicle_track_id
        if read.accepted and read.text:
            self.histories[track_id].append(read)
            self.results[track_id] = self._calculate(track_id)
        return self.results.get(track_id)

    def get(self, track_id: int) -> PlateTextResult | None:
        return self.results.get(int(track_id))

    def all_results(self) -> list[PlateTextResult]:
        return [self.results[key] for key in sorted(self.results)]

    def _calculate(self, track_id: int) -> PlateTextResult:
        observations = list(self.histories[track_id])
        grouped: dict[str, list[OCRRead]] = defaultdict(list)
        for item in observations:
            grouped[item.text].append(item)

        def group_score(items: list[OCRRead]) -> float:
            return sum(max(item.weighted_score, 1e-9) for item in items)

        text, winners = max(
            grouped.items(),
            key=lambda item: (group_score(item[1]), len(item[1])),
        )
        winner_score = group_score(winners)
        total_score = sum(group_score(items) for items in grouped.values())
        dominance = winner_score / max(total_score, 1e-9)
        confidence = sum(item.confidence for item in winners) / len(winners)
        status = (
            "confirmed"
            if len(winners) >= self.minimum_observations
            and confidence >= self.minimum_confirm_confidence
            and dominance >= self.minimum_dominance
            else "provisional"
        )
        return PlateTextResult(
            vehicle_track_id=track_id,
            text=text,
            status=status,
            confidence=confidence,
            weighted_score=winner_score,
            observation_count=len(winners),
            total_accepted_observations=len(observations),
            dominance=dominance,
            first_frame=min(item.frame_index for item in winners),
            last_frame=max(item.frame_index for item in winners),
        )
