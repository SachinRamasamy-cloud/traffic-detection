from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import yaml

from .types import Detection, TrackObservation


class DetectionBatch:
    """Minimal Results-like container required by Ultralytics BYTETracker."""

    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray, classes: np.ndarray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(classes, dtype=np.float32).reshape(-1)
        if not (len(self.xyxy) == len(self.conf) == len(self.cls)):
            raise ValueError("DetectionBatch arrays must have equal lengths")

    @property
    def xywh(self) -> np.ndarray:
        output = self.xyxy.copy()
        output[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        output[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        output[:, 0] = self.xyxy[:, 0] + output[:, 2] / 2.0
        output[:, 1] = self.xyxy[:, 1] + output[:, 3] / 2.0
        return output

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, item) -> "DetectionBatch":
        return DetectionBatch(self.xyxy[item], self.conf[item], self.cls[item])

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> "DetectionBatch":
        if not detections:
            return cls(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return cls(
            np.asarray([d.xyxy for d in detections], dtype=np.float32),
            np.asarray([d.confidence for d in detections], dtype=np.float32),
            np.asarray([d.class_id for d in detections], dtype=np.float32),
        )


class ByteTrackEngine:
    """Single full-frame ByteTrack instance. ByteTrack already includes Kalman prediction."""

    def __init__(self, config_path, frame_rate: float) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        required = {
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "fuse_score",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"ByteTrack config missing fields: {', '.join(missing)}")
        config["frame_rate"] = frame_rate
        self.tracker = BYTETracker(SimpleNamespace(**config))

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackObservation]:
        output = self.tracker.update(DetectionBatch.from_detections(detections), img=frame)
        if output is None or np.asarray(output).size == 0:
            return []
        rows = np.asarray(output, dtype=np.float32).reshape(-1, 8)
        observations = []
        for row in rows:
            observations.append(
                TrackObservation(
                    xyxy=row[:4].copy(),
                    track_id=int(row[4]),
                    confidence=float(row[5]),
                    class_id=int(row[6]),
                    detection_index=int(row[7]),
                    state="tracked",
                    prediction_gap=0,
                )
            )
        return observations

    def predicted_lost(self, maximum_gap: int) -> list[TrackObservation]:
        """Expose short Kalman-only lost-track predictions for visualization/export.

        These are predictions, not detections. Keep disabled for measurements unless the
        downstream consumer explicitly checks state == 'predicted'.
        """
        if maximum_gap <= 0:
            return []
        output: list[TrackObservation] = []
        current_frame = int(self.tracker.frame_id)
        for track in self.tracker.lost_stracks:
            gap = current_frame - int(track.end_frame)
            if gap <= 0 or gap > maximum_gap or track.mean is None:
                continue
            output.append(
                TrackObservation(
                    xyxy=np.asarray(track.xyxy, dtype=np.float32),
                    track_id=int(track.track_id),
                    confidence=float(track.score),
                    class_id=int(track.cls),
                    detection_index=-1,
                    state="predicted",
                    prediction_gap=gap,
                )
            )
        return output
