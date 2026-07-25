from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .roi import load_roi
from .tile_plan import load_or_build_tile_plan, save_tile_preview
from .video import read_video_info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate one static ROI tile-coordinate plan and preview for reuse across every video frame"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--output", default="configs/tile_plan.custom.json")
    parser.add_argument("--preview", default="configs/tile_plan.custom.jpg")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--min-tile-roi-ratio", type=float, default=0.005)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--roi-tile-padding", type=int, default=96)
    parser.add_argument(
        "--force-boundary-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--curve-samples", type=int, default=24)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        source = Path(args.source).expanduser().resolve()
        roi_path = Path(args.roi).expanduser().resolve()
        output = Path(args.output).expanduser().resolve()
        preview = Path(args.preview).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input video not found: {source}")
        if not roi_path.is_file():
            raise FileNotFoundError(f"ROI JSON not found: {roi_path}")
        if args.frame_index < 0:
            raise ValueError("frame-index cannot be negative")

        info = read_video_info(source)
        roi = load_roi(roi_path, info.width, info.height, args.curve_samples)
        plan = load_or_build_tile_plan(
            path=output,
            frame_width=info.width,
            frame_height=info.height,
            roi=roi,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            minimum_roi_ratio=args.min_tile_roi_ratio,
            max_tiles=args.max_tiles,
            roi_padding=args.roi_tile_padding,
            force_boundary_tiles=args.force_boundary_tiles,
            rebuild=args.rebuild,
        )
        save_tile_preview(
            source=source,
            frame_index=args.frame_index,
            output_path=preview,
            roi=roi,
            tiles=plan.tiles,
        )
        print(
            json.dumps(
                {
                    "tile_plan": str(output),
                    "preview": str(preview),
                    "tile_count": len(plan.tiles),
                    "boundary_tile_count": sum(1 for tile in plan.tiles if tile.boundary_tile),
                    "frame_dimensions": [info.width, info.height],
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        logging.getLogger("traffic_tracker").exception("Tile-plan generation failed: %s", exc)
        return 1
