from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from traffic_plate_study.config import load_config
from traffic_plate_study.evaluation import evaluate_predictions


def _default_config() -> Path:
    candidates = [
        Path.cwd() / "config" / "default.yaml",
        Path(__file__).resolve().parents[2] / "config" / "default.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic-plate-study",
        description="Offline vehicle tracking and number-plate recognition study pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Process a video")
    run_parser.add_argument("--video", required=True, help="Input video path")
    run_parser.add_argument("--output", required=True, help="Output JSON path")
    run_parser.add_argument(
        "--config",
        default=str(_default_config()),
        help="YAML configuration path",
    )
    run_parser.add_argument(
        "--annotated-video",
        default=None,
        help="Optional annotated MP4 path",
    )
    run_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for a quick test",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate plate results")
    evaluate_parser.add_argument("--predictions", required=True, help="Pipeline JSON result")
    evaluate_parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON")
    evaluate_parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.80,
        help="Exact plate accuracy target",
    )
    evaluate_parser.add_argument(
        "--output",
        default=None,
        help="Optional evaluation JSON output",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            from traffic_plate_study.pipeline import TrafficPlatePipeline

            config = load_config(args.config)
            pipeline = TrafficPlatePipeline(config)
            result = pipeline.run(
                video_path=args.video,
                output_path=args.output,
                annotated_video_path=args.annotated_video,
                max_frames=args.max_frames,
            )
            print(
                json.dumps(
                    {
                        "output": str(Path(args.output).expanduser().resolve()),
                        "summary": result["summary"],
                    },
                    indent=2,
                )
            )
            return

        if args.command == "evaluate":
            result = evaluate_predictions(
                predictions_path=args.predictions,
                ground_truth_path=args.ground_truth,
                target_accuracy=args.target_accuracy,
            )
            rendered = json.dumps(result, indent=2)
            print(rendered)
            if args.output:
                output = Path(args.output).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(rendered + "\n", encoding="utf-8")
            return
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
