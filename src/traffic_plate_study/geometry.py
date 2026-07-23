from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from traffic_plate_study.schemas import BBox


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, expansion: float, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    dx = box_width * expansion
    dy = box_height * expansion
    return clamp_bbox(
        (round(x1 - dx), round(y1 - dy), round(x2 + dx), round(y2 + dy)),
        width,
        height,
    )


def crop_image(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def bottom_center(bbox: BBox) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[int, int]]) -> bool:
    if not polygon:
        return True
    contour = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def apply_roi_mask(image: np.ndarray, polygon: Sequence[tuple[int, int]]) -> np.ndarray:
    if not polygon:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return cv2.bitwise_and(image, image, mask=mask)


def laplacian_blur_score(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
