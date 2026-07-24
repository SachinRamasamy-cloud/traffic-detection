from __future__ import annotations

from pathlib import Path

import cv2

from .types import VideoInfo


def read_video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps <= 0 or fps >= 1000:
        fps = 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine video dimensions")
    return VideoInfo(fps=fps, width=width, height=height, frame_count=count)


def create_video_writer(path: Path, info: VideoInfo, stride: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps / max(1, stride),
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer
