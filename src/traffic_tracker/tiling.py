from __future__ import annotations

import math

import cv2
import numpy as np

from .roi import ROI
from .types import Tile


def build_sparse_tiles(
    frame_width: int,
    frame_height: int,
    roi: ROI,
    tile_size: int,
    overlap: float,
    minimum_roi_ratio: float,
    max_tiles: int = 0,
) -> list[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("tile overlap must be in [0, 0.9)")
    if not 0.0 <= minimum_roi_ratio <= 1.0:
        raise ValueError("minimum ROI ratio must be between 0 and 1")

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = _positions(frame_width, tile_size, stride)
    ys = _positions(frame_height, tile_size, stride)
    integral = cv2.integral((roi.mask > 0).astype(np.uint8), sdepth=cv2.CV_64F)
    rx, ry, rw, rh = roi.bounding_rect
    rx2, ry2 = rx + rw, ry + rh

    candidates: list[Tile] = []
    for y in ys:
        for x in xs:
            valid_width = min(tile_size, frame_width - x)
            valid_height = min(tile_size, frame_height - y)
            if valid_width <= 0 or valid_height <= 0:
                continue
            if x >= rx2 or y >= ry2 or x + valid_width <= rx or y + valid_height <= ry:
                continue
            roi_pixels = _integral_sum(integral, x, y, x + valid_width, y + valid_height)
            if roi_pixels <= 0:
                continue
            ratio = roi_pixels / float(valid_width * valid_height)
            if ratio < minimum_roi_ratio:
                continue
            candidates.append(
                Tile(
                    index=len(candidates),
                    x=x,
                    y=y,
                    size=tile_size,
                    valid_width=valid_width,
                    valid_height=valid_height,
                    roi_ratio=ratio,
                )
            )

    if not candidates:
        raise ValueError("No tiles intersect the ROI; reduce --min-tile-roi-ratio or check ROI coordinates")

    if max_tiles > 0 and len(candidates) > max_tiles:
        selected = sorted(candidates, key=lambda tile: tile.roi_ratio, reverse=True)[:max_tiles]
        selected.sort(key=lambda tile: (tile.y, tile.x))
        candidates = selected

    return [
        Tile(i, tile.x, tile.y, tile.size, tile.valid_width, tile.valid_height, tile.roi_ratio)
        for i, tile in enumerate(candidates)
    ]


def prepare_tile_image(
    frame: np.ndarray,
    tile: Tile,
    roi_mask: np.ndarray | None = None,
    mask_outside_roi: bool = False,
    mask_dilation: int = 0,
) -> np.ndarray:
    crop = frame[tile.y : tile.y2, tile.x : tile.x2]
    output = np.zeros((tile.size, tile.size, 3), dtype=frame.dtype)
    output[: tile.valid_height, : tile.valid_width] = crop

    if mask_outside_roi and roi_mask is not None:
        local_mask = roi_mask[tile.y : tile.y2, tile.x : tile.x2]
        if mask_dilation > 0:
            kernel_size = 2 * mask_dilation + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            local_mask = cv2.dilate(local_mask, kernel, iterations=1)
        output[: tile.valid_height, : tile.valid_width] = cv2.bitwise_and(
            crop,
            crop,
            mask=local_mask,
        )
    return output


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return sorted(set(positions))


def _integral_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])
