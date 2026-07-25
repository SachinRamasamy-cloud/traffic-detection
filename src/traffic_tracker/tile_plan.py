from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .roi import ROI
from .tiling import build_sparse_tiles
from .types import Tile

LOGGER = logging.getLogger("traffic_tracker")


@dataclass(frozen=True)
class TilePlan:
    frame_width: int
    frame_height: int
    roi_digest: str
    tile_size: int
    overlap: float
    minimum_roi_ratio: float
    roi_padding: int
    force_boundary_tiles: bool
    max_tiles: int
    tiles: tuple[Tile, ...]

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "coordinate_space": "source_pixel",
            "reference_frame": {
                "width": self.frame_width,
                "height": self.frame_height,
            },
            "roi_digest": self.roi_digest,
            "settings": {
                "tile_size": self.tile_size,
                "overlap": self.overlap,
                "minimum_roi_ratio": self.minimum_roi_ratio,
                "roi_padding": self.roi_padding,
                "force_boundary_tiles": self.force_boundary_tiles,
                "max_tiles": self.max_tiles,
            },
            "tiles": [tile.to_dict() for tile in self.tiles],
        }


def roi_digest(roi: ROI) -> str:
    digest = hashlib.sha256()
    digest.update(str(roi.mask.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(roi.mask).tobytes())
    return digest.hexdigest()


def load_or_build_tile_plan(
    path: Path,
    frame_width: int,
    frame_height: int,
    roi: ROI,
    tile_size: int,
    overlap: float,
    minimum_roi_ratio: float,
    max_tiles: int,
    roi_padding: int,
    force_boundary_tiles: bool,
    rebuild: bool = False,
) -> TilePlan:
    current_digest = roi_digest(roi)

    if path.is_file() and not rebuild:
        plan = load_tile_plan(path)
        _validate_plan(
            plan,
            frame_width=frame_width,
            frame_height=frame_height,
            roi_digest_value=current_digest,
            tile_size=tile_size,
            overlap=overlap,
            minimum_roi_ratio=minimum_roi_ratio,
            roi_padding=roi_padding,
            force_boundary_tiles=force_boundary_tiles,
            max_tiles=max_tiles,
        )
        LOGGER.info("Loaded static tile plan with %d tiles from %s", len(plan.tiles), path)
        return plan

    tiles = build_sparse_tiles(
        frame_width=frame_width,
        frame_height=frame_height,
        roi=roi,
        tile_size=tile_size,
        overlap=overlap,
        minimum_roi_ratio=minimum_roi_ratio,
        max_tiles=max_tiles,
        roi_padding=roi_padding,
        force_boundary_tiles=force_boundary_tiles,
    )
    plan = TilePlan(
        frame_width=frame_width,
        frame_height=frame_height,
        roi_digest=current_digest,
        tile_size=tile_size,
        overlap=overlap,
        minimum_roi_ratio=minimum_roi_ratio,
        roi_padding=roi_padding,
        force_boundary_tiles=force_boundary_tiles,
        max_tiles=max_tiles,
        tiles=tuple(tiles),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Built and saved static tile plan with %d tiles to %s", len(tiles), path)
    return plan


def load_tile_plan(path: Path) -> TilePlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = payload.get("reference_frame") or {}
    settings = payload.get("settings") or {}
    tiles = tuple(Tile.from_dict(value) for value in payload.get("tiles", []))
    if not tiles:
        raise ValueError(f"Tile plan contains no tiles: {path}")
    return TilePlan(
        frame_width=int(reference["width"]),
        frame_height=int(reference["height"]),
        roi_digest=str(payload.get("roi_digest", "")),
        tile_size=int(settings["tile_size"]),
        overlap=float(settings["overlap"]),
        minimum_roi_ratio=float(settings["minimum_roi_ratio"]),
        roi_padding=int(settings.get("roi_padding", 0)),
        force_boundary_tiles=bool(settings.get("force_boundary_tiles", True)),
        max_tiles=int(settings.get("max_tiles", 0)),
        tiles=tiles,
    )


def save_tile_preview(
    source: Path,
    frame_index: int,
    output_path: Path,
    roi: ROI,
    tiles: list[Tile] | tuple[Tile, ...],
) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for tile preview: {source}")
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} for tile preview")

    preview = frame.copy()
    for region in roi.regions:
        cv2.polylines(preview, [np.rint(region.outer).astype(np.int32)], True, (255, 255, 0), 2)
    for tile in tiles:
        color = (0, 165, 255) if tile.boundary_tile else (255, 0, 255)
        cv2.rectangle(preview, (tile.x, tile.y), (tile.x2, tile.y2), color, 2)
        cv2.putText(
            preview,
            f"T{tile.index}",
            (tile.x + 5, tile.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Could not save tile preview: {output_path}")


def _validate_plan(
    plan: TilePlan,
    frame_width: int,
    frame_height: int,
    roi_digest_value: str,
    tile_size: int,
    overlap: float,
    minimum_roi_ratio: float,
    roi_padding: int,
    force_boundary_tiles: bool,
    max_tiles: int,
) -> None:
    mismatches: list[str] = []
    if (plan.frame_width, plan.frame_height) != (frame_width, frame_height):
        mismatches.append("video dimensions")
    if plan.roi_digest != roi_digest_value:
        mismatches.append("ROI geometry")
    if plan.tile_size != tile_size:
        mismatches.append("tile size")
    if abs(plan.overlap - overlap) > 1e-9:
        mismatches.append("tile overlap")
    if abs(plan.minimum_roi_ratio - minimum_roi_ratio) > 1e-9:
        mismatches.append("minimum ROI ratio")
    if plan.roi_padding != roi_padding:
        mismatches.append("ROI tile padding")
    if plan.force_boundary_tiles != force_boundary_tiles:
        mismatches.append("boundary-tile policy")
    if plan.max_tiles != max_tiles:
        mismatches.append("maximum tile count")
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            f"Existing tile plan does not match the current {joined}. "
            "Run again with --rebuild-tile-plan."
        )
