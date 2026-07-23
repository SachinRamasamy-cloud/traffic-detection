from __future__ import annotations

import cv2
import numpy as np

from traffic_plate_study.schemas import PlateConsensus, VehicleDetection


def draw_vehicle(
    frame: np.ndarray,
    detection: VehicleDetection,
    consensus: PlateConsensus | None,
) -> None:
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
    label = f"ID {detection.track_id} {detection.class_name} {detection.confidence:.2f}"
    if consensus and consensus.text:
        plate_status = "OK" if consensus.accepted else "?"
        label += f" | {consensus.text} {consensus.confidence:.2f} {plate_status}"
    _draw_label(frame, label, x1, y1)


def draw_plate_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str | None,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 180, 255), 2)
    if text:
        _draw_label(frame, text, x1, y1, color=(30, 180, 255))


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (60, 220, 60),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.8, frame.shape[1] / 1800.0))
    thickness = 1 if font_scale < 0.65 else 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top = max(0, y - text_height - baseline - 8)
    right = min(frame.shape[1] - 1, x + text_width + 8)
    cv2.rectangle(frame, (x, top), (right, y), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x + 4, max(text_height + 2, y - baseline - 4)),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
