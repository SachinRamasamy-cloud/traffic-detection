# YOLO26m ROI-Aware Tiled ByteTrack

A study-oriented Linux CPU pipeline with:

- **Detector:** `yolo26m.pt`
- **Preprocessing:** polygon/curve ROI + sparse overlapping tiles
- **Duplicate merge:** global cross-tile NMS
- **Tracker:** one full-frame ByteTrack instance
- **Motion prediction:** ByteTrack's built-in linear Kalman filter
- **Class output:** confidence-weighted track-level class stabilization
- **Outputs:** annotated MP4, JSONL, CSV, and summary JSON

## Why this architecture

```text
video frame
  -> ROI-intersecting tiles only
  -> YOLO26m detection on every selected tile
  -> map tile boxes to full-frame coordinates
  -> merge cross-tile duplicates
  -> filter using ROI bottom-center/center/overlap rule
  -> one ByteTrack update for the complete frame
  -> stable class + export
```

Do not run a different tracker per tile. Tracking must receive one merged detection set in the original video coordinate system.

ByteTrack already performs Kalman prediction before association. Its track buffer allows a recently missed object to remain recoverable. This can preserve the same ID when the object reappears, but it cannot guarantee recovery after long occlusion, severe detector failure, or crossing with a similar object.

Optional `--prediction-frames N` draws/exports short Kalman-only lost-track estimates. They are marked `state="predicted"` and must not be treated as measured detections.

## Folder structure

```text
yolo26_midrange_roi_tracker/
├── configs/
│   ├── bytetrack_traffic.yaml
│   └── roi.example.json
├── src/traffic_tracker/
│   ├── cli.py
│   ├── detector.py
│   ├── drawing.py
│   ├── exporter.py
│   ├── nms.py
│   ├── pipeline.py
│   ├── roi.py
│   ├── runtime.py
│   ├── stabilization.py
│   ├── tiling.py
│   ├── tracking.py
│   ├── types.py
│   └── video.py
├── tests/test_core.py
├── pyproject.toml
├── requirements.txt
├── setup.sh
├── download_model.sh
└── run_tracker.py
```

## Installation

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg curl

chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

The model downloads automatically on first use. Manual download:

```bash
./download_model.sh
```

Official weight asset:

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt
```

## ROI input

The preferred transfer/storage format is JSON. The backend converts it to NumPy/OpenCV arrays at runtime.

Pixel polygon:

```json
{
  "coordinate_space": "source_pixel",
  "reference_frame": {"width": 1920, "height": 1080},
  "detection_geometry": {
    "type": "Polygon",
    "points": [[140,710],[235,570],[430,475],[770,430],[1130,455],[1510,570],[1830,910],[150,980]]
  },
  "filtering": {"rule": "bottom_center"}
}
```

The loader also supports normalized Polygon/MultiPolygon data, polygon holes, and sampled `M/L/C/Q/Z` curve commands under `original_geometry` when no `detection_geometry` is supplied.

## Recommended first run

```bash
traffic-track \
  --source /path/to/test.mp4 \
  --roi configs/roi.example.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --tile-size 960 \
  --tile-overlap 0.20 \
  --tile-batch-size 1 \
  --imgsz 640 \
  --conf 0.06 \
  --prediction-frames 8 \
  --output-dir runs/midrange_roi
```

For a short validation:

```bash
traffic-track \
  --source /path/to/test.mp4 \
  --roi configs/roi.example.json \
  --classes car,motorcycle,bus,truck,bicycle \
  --max-frames 100 \
  --draw-tiles \
  --output-dir runs/smoke
```

## Important tuning

- `--tile-size 960`: source-pixel crop size. Larger tiles mean fewer detector calls but smaller objects after resize.
- `--imgsz 640`: YOLO input size for each tile.
- `--tile-overlap 0.20`: protects objects cut by tile boundaries.
- `--merge-iou 0.55`: removes duplicate boxes created by overlapping tiles.
- `--conf 0.06`: must remain at or below ByteTrack `track_low_thresh`.
- `track_buffer: 90`: keeps lost state for 90 **processed** frames.
- `--prediction-frames 8`: visualization/export of brief Kalman-only predictions; use `0` for strict detection-only outputs.
- `--one-to-many`: optional higher-recall YOLO26 head, with additional CPU/post-processing cost.
- `--mask-outside-roi`: blackens non-ROI pixels inside selected tiles. Leave disabled initially because masking can truncate objects at the ROI boundary.

## CPU warning

`yolo26m.pt` plus multiple tiles is substantially heavier than `yolo26n.pt`. Begin with `--max-frames 100`, inspect `summary.json`, and then tune tile count, tile size, ROI size, and image size. Tiling improves small-object visibility but does not make a medium model inexpensive.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Interactive custom ROI drawing

The package includes a desktop OpenCV ROI editor. It reads a selected video frame, displays it at a screen-friendly size, maps every mouse point back into the **original source-video pixel coordinates**, and writes tracker-compatible JSON.

Install Linux GUI libraries once if needed:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

After activating the environment:

```bash
source .venv/bin/activate
```

### Polygon drawing

```bash
traffic-draw-roi \
  --source /path/to/test.mp4 \
  --frame-index 0 \
  --mode polygon \
  --output configs/roi.custom.json
```

Controls:

- Left-click: add polygon point
- Right-click, `U`, or Backspace: remove the last point
- `R`: reset all points
- `S` or Enter: save
- `Q` or Escape: cancel

### Freehand curved ROI

```bash
traffic-draw-roi \
  --source /path/to/test.mp4 \
  --frame-index 0 \
  --mode freehand \
  --freehand-step 3 \
  --simplify-epsilon 2 \
  --output configs/roi.curved.json
```

Hold the left mouse button and draw the ROI boundary. The saved shape remains curved because the utility stores many source-pixel points. `--simplify-epsilon` removes redundant points without converting the curve into a straight line. Use `0` to disable simplification.

Run tracking with the saved ROI:

```bash
traffic-track \
  --source /path/to/test.mp4 \
  --roi configs/roi.custom.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --draw-roi \
  --draw-tiles \
  --max-frames 100 \
  --output-dir runs/custom_roi_test
```

You can also run the editor without installing the command entry point:

```bash
python draw_roi.py --source /path/to/test.mp4 --output configs/roi.custom.json
```
