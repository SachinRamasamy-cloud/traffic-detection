from __future__ import annotations

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
    roi_padding: int = 96,
    force_boundary_tiles: bool = True,
) -> list[Tile]:
    """Build a static ROI-only tile grid that is reused for every video frame.

    The grid is anchored to the padded ROI bounding rectangle rather than the
    complete image. Boundary-intersecting tiles are retained even when their
    ROI ratio is small, which protects vehicles entering at the ROI edge.
    """

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("tile overlap must be in [0, 0.9)")
    if not 0.0 <= minimum_roi_ratio <= 1.0:
        raise ValueError("minimum ROI ratio must be between 0 and 1")
    if roi_padding < 0:
        raise ValueError("roi_padding cannot be negative")

    selection_mask = roi.dilated_mask(roi_padding)
    bounding_rect = _mask_bounding_rect(selection_mask)
    if bounding_rect is None:
        raise ValueError("ROI has no pixels available for tiling")

    rx, ry, rw, rh = bounding_rect
    rx2, ry2 = rx + rw, ry + rh
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = _interval_positions(rx, rx2, frame_width, tile_size, stride)
    ys = _interval_positions(ry, ry2, frame_height, tile_size, stride)

    roi_binary = (selection_mask > 0).astype(np.uint8)
    integral = cv2.integral(roi_binary, sdepth=cv2.CV_64F)

    # A thick boundary band catches entry/exit tiles even when only a narrow
    # part of the ROI is present inside the tile.
    boundary_width = max(3, min(31, int(round(tile_size * 0.03))))
    kernel = np.ones((boundary_width, boundary_width), dtype=np.uint8)
    eroded = cv2.erode(roi_binary, kernel, iterations=1)
    boundary = cv2.subtract(roi_binary, eroded)
    boundary_integral = cv2.integral(boundary, sdepth=cv2.CV_64F)

    candidates: list[Tile] = []
    for y in ys:
        for x in xs:
            valid_width = min(tile_size, frame_width - x)
            valid_height = min(tile_size, frame_height - y)
            if valid_width <= 0 or valid_height <= 0:
                continue

            x2 = x + valid_width
            y2 = y + valid_height
            roi_pixels = _integral_sum(integral, x, y, x2, y2)
            if roi_pixels <= 0:
                continue

            area = float(valid_width * valid_height)
            ratio = roi_pixels / area
            boundary_pixels = _integral_sum(boundary_integral, x, y, x2, y2)
            is_boundary = boundary_pixels > 0

            if ratio < minimum_roi_ratio and not (force_boundary_tiles and is_boundary):
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
                    boundary_tile=is_boundary,
                )
            )

    if not candidates:
        raise ValueError(
            "No tiles intersect the ROI; reduce --min-tile-roi-ratio, increase "
            "--roi-tile-padding, or verify the ROI coordinates"
        )

    if max_tiles > 0 and len(candidates) > max_tiles:
        candidates = _coverage_preserving_selection(
            candidates,
            selection_mask,
            maximum=max_tiles,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    candidates.sort(key=lambda tile: (tile.y, tile.x))
    return [
        Tile(
            index=index,
            x=tile.x,
            y=tile.y,
            size=tile.size,
            valid_width=tile.valid_width,
            valid_height=tile.valid_height,
            roi_ratio=tile.roi_ratio,
            boundary_tile=tile.boundary_tile,
        )
        for index, tile in enumerate(candidates)
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


def _mask_bounding_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    return cv2.boundingRect(points)


def _interval_positions(
    interval_start: int,
    interval_end: int,
    frame_length: int,
    tile_size: int,
    stride: int,
) -> list[int]:
    """Anchor the grid to both ends of the ROI interval, not the full frame."""

    if frame_length <= tile_size:
        return [0]

    interval_start = max(0, min(interval_start, frame_length - 1))
    interval_end = max(interval_start + 1, min(interval_end, frame_length))
    maximum_start = frame_length - tile_size

    first = max(0, min(interval_start, maximum_start))
    last = max(0, min(interval_end - tile_size, maximum_start))

    positions = [first]
    current = first
    while current + stride < last:
        current += stride
        positions.append(current)
    positions.append(last)

    return sorted(set(positions))


def _coverage_preserving_selection(
    candidates: list[Tile],
    mask: np.ndarray,
    maximum: int,
    frame_width: int,
    frame_height: int,
) -> list[Tile]:
    """Greedy set-cover selection that keeps ROI boundary coverage."""

    if maximum <= 0 or len(candidates) <= maximum:
        return candidates

    uncovered = (mask > 0).astype(np.uint8)
    selected: list[Tile] = []
    remaining = list(candidates)

    # Prefer boundary tiles first, then choose tiles that cover the most still
    # uncovered ROI pixels. This avoids dropping entry/exit tiles solely because
    # their ROI ratio is lower than central tiles.
    boundary_tiles = sorted(
        [tile for tile in remaining if tile.boundary_tile],
        key=lambda tile: tile.roi_ratio,
        reverse=True,
    )
    for tile in boundary_tiles:
        if len(selected) >= maximum:
            break
        selected.append(tile)
        uncovered[tile.y : tile.y2, tile.x : tile.x2] = 0
        remaining.remove(tile)

    while remaining and len(selected) < maximum:
        best_tile = None
        best_gain = -1
        for tile in remaining:
            gain = int(np.count_nonzero(uncovered[tile.y : tile.y2, tile.x : tile.x2]))
            if gain > best_gain:
                best_gain = gain
                best_tile = tile
        if best_tile is None:
            break
        selected.append(best_tile)
        uncovered[best_tile.y : best_tile.y2, best_tile.x : best_tile.x2] = 0
        remaining.remove(best_tile)

    return selected


def _integral_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])
