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
    line_width: int,
) -> np.ndarray:
    output = frame.copy()
    if draw_roi:
        for region in roi.regions:
            cv2.polylines(output, [np.rint(region.outer).astype(np.int32)], True, (255, 255, 0), 2)
    if draw_tiles:
        for tile in tiles:
            cv2.rectangle(output, (tile.x, tile.y), (tile.x2, tile.y2), (255, 0, 255), 1)
            cv2.putText(output, f"T{tile.index}", (tile.x + 4, tile.y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    for record in records:
        x1, y1, x2, y2 = [int(round(v)) for v in record["bbox_xyxy"]]
        predicted = record["state"] == "predicted"
        color = (0, 200, 255) if predicted else (0, 255, 0)
        label = f"ID {record['track_id']} {record['class_name']}"
        if predicted:
            label += f" P+{record['prediction_gap']}"
            _dashed_rectangle(output, (x1, y1), (x2, y2), color, line_width)
        else:
            label += f" {record['confidence']:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - baseline - 6)
        cv2.rectangle(output, (x1, top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def _dashed_rectangle(image, p1, p2, color, thickness, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)
