from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Traffic ROI Drawer"


@dataclass
class ViewTransform:
    source_width: int
    source_height: int
    display_width: int
    display_height: int
    scale: float

    @classmethod
    def fit(
        cls,
        source_width: int,
        source_height: int,
        max_width: int,
        max_height: int,
    ) -> "ViewTransform":
        scale = min(
            max_width / max(source_width, 1),
            max_height / max(source_height, 1),
            1.0,
        )
        display_width = max(1, int(round(source_width * scale)))
        display_height = max(1, int(round(source_height * scale)))
        return cls(source_width, source_height, display_width, display_height, scale)

    def display_to_source(self, x: int, y: int) -> tuple[float, float]:
        source_x = float(np.clip(x / self.scale, 0, self.source_width - 1))
        source_y = float(np.clip(y / self.scale, 0, self.source_height - 1))
        return source_x, source_y

    def source_to_display(self, point: tuple[float, float]) -> tuple[int, int]:
        return int(round(point[0] * self.scale)), int(round(point[1] * self.scale))


@dataclass
class DrawerState:
    mode: str
    transform: ViewTransform
    points: list[tuple[float, float]] = field(default_factory=list)
    drawing: bool = False
    cursor: tuple[int, int] | None = None
    minimum_freehand_distance: float = 3.0

    def add_display_point(self, x: int, y: int, force: bool = False) -> None:
        point = self.transform.display_to_source(x, y)
        if self.points and not force:
            last = self.points[-1]
            distance = math.hypot(point[0] - last[0], point[1] - last[1])
            if distance < self.minimum_freehand_distance:
                return
        self.points.append(point)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a polygon/freehand ROI on a video frame and save tracker-compatible JSON."
    )
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        default="configs/roi.custom.json",
        help="Output ROI JSON path.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Video frame used as the drawing background.",
    )
    parser.add_argument(
        "--mode",
        choices=["polygon", "freehand"],
        default="polygon",
        help="Polygon: click vertices. Freehand: hold left mouse and draw.",
    )
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=850)
    parser.add_argument(
        "--freehand-step",
        type=float,
        default=3.0,
        help="Minimum source-pixel distance between recorded freehand points.",
    )
    parser.add_argument(
        "--simplify-epsilon",
        type=float,
        default=2.0,
        help="Freehand polygon simplification tolerance in source pixels. Set 0 to disable.",
    )
    parser.add_argument(
        "--filter-rule",
        choices=["bottom_center", "center", "overlap"],
        default="bottom_center",
    )
    parser.add_argument("--minimum-overlap", type=float, default=0.25)
    return parser.parse_args(argv)


def read_frame(source: Path, frame_index: int) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if frame_index < 0:
            raise ValueError("--frame-index cannot be negative")
        if total_frames > 0 and frame_index >= total_frames:
            raise ValueError(
                f"--frame-index {frame_index} is outside video frame count {total_frames}"
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {source}")
        return frame, fps, total_frames
    finally:
        cap.release()


def simplify_points(
    points: list[tuple[float, float]],
    epsilon: float,
) -> list[tuple[float, float]]:
    if len(points) < 3 or epsilon <= 0:
        return points
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in simplified]


def build_payload(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    frame_index: int,
    filter_rule: str,
    minimum_overlap: float,
    roi_id: str,
) -> dict:
    if len(points) < 3:
        raise ValueError("ROI must contain at least three points")

    clean = [(float(x), float(y)) for x, y in points]
    if not np.allclose(clean[0], clean[-1]):
        clean.append(clean[0])

    return {
        "roi_id": roi_id,
        "version": 1,
        "coordinate_space": "source_pixel",
        "reference_frame": {
            "width": int(width),
            "height": int(height),
            "frame_index": int(frame_index),
        },
        "detection_geometry": {
            "type": "Polygon",
            "points": [[round(x, 3), round(y, 3)] for x, y in clean],
        },
        "filtering": {
            "rule": filter_rule,
            "minimum_intersection_ratio": float(minimum_overlap),
        },
    }


def render(
    base: np.ndarray,
    state: DrawerState,
    frame_index: int,
) -> np.ndarray:
    canvas = base.copy()
    display_points = [state.transform.source_to_display(point) for point in state.points]

    if len(display_points) >= 2:
        cv2.polylines(
            canvas,
            [np.asarray(display_points, dtype=np.int32)],
            isClosed=False,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if state.mode == "polygon" and display_points and state.cursor is not None:
        cv2.line(canvas, display_points[-1], state.cursor, (160, 160, 160), 1, cv2.LINE_AA)

    for index, point in enumerate(display_points):
        cv2.circle(canvas, point, 4, (0, 255, 0), -1, cv2.LINE_AA)
        if state.mode == "polygon":
            cv2.putText(
                canvas,
                str(index + 1),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if len(display_points) >= 3:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.asarray(display_points, dtype=np.int32)], (0, 180, 0))
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(
            canvas,
            [np.asarray(display_points, dtype=np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    instructions = [
        f"Mode: {state.mode} | Frame: {frame_index} | Points: {len(state.points)}",
        "Polygon: left-click vertices | Freehand: hold left mouse and drag",
        "S/Enter: save | U/Backspace: undo | R: reset | Q/Esc: cancel",
    ]
    y = 24
    for text in instructions:
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return canvas


def make_mouse_callback(state: DrawerState):
    def callback(event: int, x: int, y: int, flags: int, _param) -> None:
        state.cursor = (x, y)

        if state.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                state.add_display_point(x, y, force=True)
            elif event == cv2.EVENT_RBUTTONDOWN and state.points:
                state.points.pop()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.add_display_point(x, y, force=True)
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            state.add_display_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.add_display_point(x, y, force=True)
            state.drawing = False

    return callback


def run(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.max_width <= 0 or args.max_height <= 0:
        raise ValueError("Display dimensions must be positive")
    if args.freehand_step < 0:
        raise ValueError("--freehand-step cannot be negative")
    if args.simplify_epsilon < 0:
        raise ValueError("--simplify-epsilon cannot be negative")
    if not 0.0 <= args.minimum_overlap <= 1.0:
        raise ValueError("--minimum-overlap must be between 0 and 1")

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    output = Path(args.output).expanduser().resolve()

    frame, _fps, _total_frames = read_frame(source, args.frame_index)
    height, width = frame.shape[:2]
    transform = ViewTransform.fit(width, height, args.max_width, args.max_height)
    display_frame = cv2.resize(
        frame,
        (transform.display_width, transform.display_height),
        interpolation=cv2.INTER_AREA,
    )
    state = DrawerState(
        mode=args.mode,
        transform=transform,
        minimum_freehand_distance=args.freehand_step,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, transform.display_width, transform.display_height)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(state))

    saved = False
    try:
        while True:
            cv2.imshow(WINDOW_NAME, render(display_frame, state, args.frame_index))
            key = cv2.waitKey(16) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord("r"),):
                state.points.clear()
                continue
            if key in (ord("u"), 8, 127):
                if state.points:
                    state.points.pop()
                continue
            if key in (ord("s"), 10, 13):
                points = state.points
                if args.mode == "freehand":
                    points = simplify_points(points, args.simplify_epsilon)
                if len(points) < 3:
                    print("ROI needs at least three points before saving.")
                    continue
                payload = build_payload(
                    points=points,
                    width=width,
                    height=height,
                    frame_index=args.frame_index,
                    filter_rule=args.filter_rule,
                    minimum_overlap=args.minimum_overlap,
                    roi_id=output.stem,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                saved = True
                break
    finally:
        cv2.destroyAllWindows()

    if not saved:
        raise RuntimeError("ROI drawing cancelled; no file was written")

    print(f"Saved ROI: {output}")
    print(f"Source dimensions: {width}x{height}")
    print(f"Stored points: {len(payload['detection_geometry']['points'])}")
    return output


def main(argv: list[str] | None = None) -> int:
    try:
        run(argv)
        return 0
    except KeyboardInterrupt:
        print("ROI drawing interrupted")
        return 130
    except Exception as exc:
        print(f"ROI drawing failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
