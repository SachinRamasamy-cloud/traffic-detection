from __future__ import annotations

import numpy as np

from .types import Detection


def merge_detections(
    detections: list[Detection],
    iou_threshold: float,
    class_agnostic: bool = True,
    max_detections: int = 1000,
) -> list[Detection]:
    """Merge cross-tile duplicates after boxes are remapped to full-frame coordinates."""
    if not detections:
        return []
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("merge IoU must be between 0 and 1")

    boxes = np.asarray([d.xyxy for d in detections], dtype=np.float32)
    scores = np.asarray([d.confidence for d in detections], dtype=np.float32)
    classes = np.asarray([d.class_id for d in detections], dtype=np.int32)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size and len(keep) < max_detections:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[index], boxes[remaining])
        suppress = ious > iou_threshold
        if not class_agnostic:
            suppress &= classes[remaining] == classes[index]
        order = remaining[~suppress]

    return [detections[index] for index in keep]


def box_iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if len(others) == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(0.0, others[:, 3] - others[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)
