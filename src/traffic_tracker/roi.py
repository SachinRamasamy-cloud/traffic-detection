from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PolygonRegion:
    outer: np.ndarray
    holes: tuple[np.ndarray, ...] = ()


@dataclass
class ROI:
    regions: list[PolygonRegion]
    mask: np.ndarray
    rule: str = "bottom_center"
    minimum_intersection_ratio: float = 0.25

    @property
    def enabled(self) -> bool:
        return bool(self.regions)

    @property
    def bounding_rect(self) -> tuple[int, int, int, int]:
        points = np.concatenate([region.outer for region in self.regions], axis=0)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        return x, y, w, h

    def contains_point(self, x: float, y: float) -> bool:
        xi = int(round(x))
        yi = int(round(y))
        if yi < 0 or xi < 0 or yi >= self.mask.shape[0] or xi >= self.mask.shape[1]:
            return False
        return bool(self.mask[yi, xi])

    def accepts_box(self, box: np.ndarray) -> bool:
        x1, y1, x2, y2 = [float(v) for v in box]
        if self.rule == "center":
            return self.contains_point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if self.rule == "overlap":
            ix1 = max(0, int(np.floor(x1)))
            iy1 = max(0, int(np.floor(y1)))
            ix2 = min(self.mask.shape[1], int(np.ceil(x2)))
            iy2 = min(self.mask.shape[0], int(np.ceil(y2)))
            if ix2 <= ix1 or iy2 <= iy1:
                return False
            area = (ix2 - ix1) * (iy2 - iy1)
            overlap = int(np.count_nonzero(self.mask[iy1:iy2, ix1:ix2]))
            return overlap / max(area, 1) >= self.minimum_intersection_ratio
        # Traffic default: approximate road contact point.
        return self.contains_point((x1 + x2) / 2.0, y2)


def full_frame_roi(width: int, height: int) -> ROI:
    mask = np.full((height, width), 255, dtype=np.uint8)
    outer = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return ROI([PolygonRegion(outer)], mask, rule="bottom_center", minimum_intersection_ratio=0.0)


def load_roi(path: Path | None, width: int, height: int, curve_samples: int = 24) -> ROI:
    if path is None:
        return full_frame_roi(width, height)

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_regions = _extract_regions(payload, curve_samples=max(4, curve_samples))
    if not raw_regions:
        raise ValueError("ROI JSON contains no usable polygon geometry")

    coordinate_space = str(payload.get("coordinate_space", "source_pixel")).lower()
    reference = payload.get("reference_frame") or {}
    ref_width = float(reference.get("width", width))
    ref_height = float(reference.get("height", height))
    if ref_width <= 0 or ref_height <= 0:
        raise ValueError("ROI reference frame dimensions must be positive")

    def transform(points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).copy()
        if coordinate_space in {"normalized", "normalised"}:
            result[:, 0] *= width
            result[:, 1] *= height
        elif coordinate_space in {"source_pixel", "pixel", "pixels"}:
            result[:, 0] *= width / ref_width
            result[:, 1] *= height / ref_height
        else:
            raise ValueError(f"Unsupported ROI coordinate_space: {coordinate_space}")
        result[:, 0] = np.clip(result[:, 0], 0, width - 1)
        result[:, 1] = np.clip(result[:, 1], 0, height - 1)
        return result

    regions = [
        PolygonRegion(transform(outer), tuple(transform(hole) for hole in holes))
        for outer, holes in raw_regions
    ]

    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        cv2.fillPoly(mask, [np.rint(region.outer).astype(np.int32)], 255)
        for hole in region.holes:
            cv2.fillPoly(mask, [np.rint(hole).astype(np.int32)], 0)

    if not np.any(mask):
        raise ValueError("ROI has zero area after conversion to video coordinates")

    filtering = payload.get("filtering") or {}
    rule = str(filtering.get("rule", "bottom_center")).lower()
    if rule not in {"bottom_center", "center", "overlap"}:
        raise ValueError("ROI filtering rule must be bottom_center, center, or overlap")
    min_overlap = float(filtering.get("minimum_intersection_ratio", 0.25))
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("minimum_intersection_ratio must be between 0 and 1")
    return ROI(regions, mask, rule=rule, minimum_intersection_ratio=min_overlap)


def _extract_regions(payload: dict[str, Any], curve_samples: int) -> list[tuple[np.ndarray, tuple[np.ndarray, ...]]]:
    candidate = payload.get("detection_geometry") or payload.get("geometry")

    if candidate:
        geometry_type = str(candidate.get("type", "Polygon")).lower()
        if "points" in candidate:
            return [(_points(candidate["points"]), ())]
        coordinates = candidate.get("coordinates")
        if coordinates is not None:
            if geometry_type == "polygon":
                rings = [_points(ring) for ring in coordinates]
                return [(rings[0], tuple(rings[1:]))]
            if geometry_type == "multipolygon":
                output = []
                for polygon in coordinates:
                    rings = [_points(ring) for ring in polygon]
                    output.append((rings[0], tuple(rings[1:])))
                return output

    if "points" in payload:
        return [(_points(payload["points"]), ())]

    original = payload.get("original_geometry")
    if original:
        sampled = _sample_path(original, curve_samples)
        return [(sampled, ())]

    return []


def _points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Each ROI ring must contain at least three [x, y] points")
    if not np.isfinite(points).all():
        raise ValueError("ROI points must contain finite numbers")
    return points


def _sample_path(original: dict[str, Any], samples: int) -> np.ndarray:
    commands = original.get("segments") or original.get("commands") or []
    output: list[np.ndarray] = []
    current: np.ndarray | None = None
    start: np.ndarray | None = None

    for item in commands:
        command = str(item.get("command") or item.get("type") or "").upper()
        command = {"MOVE": "M", "LINE": "L", "CUBIC_BEZIER": "C", "QUADRATIC_BEZIER": "Q", "CLOSE": "Z"}.get(command, command)
        points = item.get("points")

        if command == "M":
            p = np.asarray((points or [item.get("point")])[0], dtype=np.float32)
            output.append(p)
            current = p
            start = p.copy()
        elif command == "L":
            p = np.asarray((points or [item.get("end")])[0], dtype=np.float32)
            output.append(p)
            current = p
        elif command == "C":
            if current is None:
                raise ValueError("Bezier path must start with a move command")
            if points:
                c1, c2, end = [np.asarray(p, dtype=np.float32) for p in points]
            else:
                c1 = np.asarray(item["control1"], dtype=np.float32)
                c2 = np.asarray(item["control2"], dtype=np.float32)
                end = np.asarray(item["end"], dtype=np.float32)
            for t in np.linspace(0.0, 1.0, samples + 1, dtype=np.float32)[1:]:
                p = ((1 - t) ** 3) * current + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * (t**2) * c2 + (t**3) * end
                output.append(p)
            current = end
        elif command == "Q":
            if current is None:
                raise ValueError("Bezier path must start with a move command")
            if points:
                control, end = [np.asarray(p, dtype=np.float32) for p in points]
            else:
                control = np.asarray(item["control"], dtype=np.float32)
                end = np.asarray(item["end"], dtype=np.float32)
            for t in np.linspace(0.0, 1.0, samples + 1, dtype=np.float32)[1:]:
                p = ((1 - t) ** 2) * current + 2 * (1 - t) * t * control + (t**2) * end
                output.append(p)
            current = end
        elif command == "Z":
            if start is not None and (not output or not np.allclose(output[-1], start)):
                output.append(start.copy())
            current = start
        else:
            raise ValueError(f"Unsupported path command: {command}")

    return _points(output)
