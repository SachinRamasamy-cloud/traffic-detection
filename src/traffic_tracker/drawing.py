from __future__ import annotations

import cv2
import numpy as np

from .roi import ROI
from .types import Tile


def draw_frame(
    frame: np.ndarray,
    records: list[dict],
    roi: ROI,
    tiles: list[Tile],
    draw_roi: bool,
    draw_tiles: bool,
    draw_plates: bool,
    line_width: int,
) -> np.ndarray:
    output = frame.copy()
    if draw_roi:
        for region in roi.regions:
            cv2.polylines(
                output,
                [np.rint(region.outer).astype(np.int32)],
                True,
                (255, 255, 0),
                2,
            )
    if draw_tiles:
        for tile in tiles:
            color = (0, 165, 255) if tile.boundary_tile else (255, 0, 255)
            cv2.rectangle(output, (tile.x, tile.y), (tile.x2, tile.y2), color, 1)
            cv2.putText(
                output,
                f"T{tile.index}",
                (tile.x + 4, tile.y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    for record in records:
        x1, y1, x2, y2 = [int(round(value)) for value in record["bbox_xyxy"]]
        predicted = record["state"] == "predicted"
        inside_exact_roi = bool(record.get("inside_exact_roi", True))
        if predicted:
            color = (0, 200, 255)
        elif not inside_exact_roi:
            # Entry-margin track: accepted for continuity but not yet inside the
            # exact user ROI.
            color = (255, 180, 0)
        else:
            color = (0, 255, 0)

        label = f"ID {record['track_id']} {record['class_name']}"
        if predicted:
            label += f" P+{record['prediction_gap']}"
            _dashed_rectangle(output, (x1, y1), (x2, y2), color, line_width)
        else:
            label += f" {record['confidence']:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width)
        _draw_label(output, label, x1, y1, color)

        plate = record.get("plate")
        if draw_plates and plate:
            px1, py1, px2, py2 = [int(round(value)) for value in plate["bbox_xyxy"]]
            plate_color = (0, 255, 255) if plate["state"] == "current" else (0, 180, 220)
            cv2.rectangle(output, (px1, py1), (px2, py2), plate_color, max(1, line_width))
            plate_label = f"PLATE V{record['track_id']} {plate['confidence']:.2f}"
            if plate["state"] == "cached":
                plate_label += f" C+{plate['age_frames']}"
            _draw_label(output, plate_label, px1, py1, plate_color, scale=0.43)
    return output


def _draw_label(image, label: str, x: int, y: int, color, scale: float = 0.5) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        1,
    )
    top = max(0, y - text_height - baseline - 6)
    cv2.rectangle(image, (x, top), (x + text_width + 6, y), color, -1)
    cv2.putText(
        image,
        label,
        (x + 3, y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _dashed_rectangle(image, p1, p2, color, thickness, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)
