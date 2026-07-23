from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from traffic_plate_study.schemas import FramePacket


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    reported_frame_count: int
    reported_duration_ms: float


class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.path}")

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0
        duration_ms = frame_count / fps * 1000.0 if frame_count > 0 else 0.0
        self.metadata = VideoMetadata(
            path=str(self.path),
            width=width,
            height=height,
            fps=fps,
            reported_frame_count=frame_count,
            reported_duration_ms=duration_ms,
        )

    def __iter__(self) -> Iterator[FramePacket]:
        frame_index = 0
        last_timestamp_ms = -1.0
        while True:
            success, frame = self.capture.read()
            if not success:
                break

            timestamp_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            fallback_timestamp = frame_index / self.metadata.fps * 1000.0
            if timestamp_ms < 0 or (frame_index > 0 and timestamp_ms <= last_timestamp_ms):
                timestamp_ms = fallback_timestamp
            last_timestamp_ms = timestamp_ms

            yield FramePacket(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                image=frame,
            )
            frame_index += 1

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()
