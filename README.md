# YOLO26m Static ROI-Tile Tracker + Number-Plate Detection

Linux CPU study pipeline with:

- `yolo26m.pt` vehicle detection
- polygon/freehand ROI support
- a **static ROI tile plan generated once and reused for every frame**
- boundary-aware overlapping tiling for vehicles entering at the ROI edge
- global cross-tile duplicate merging
- one full-frame ByteTrack instance with its built-in Kalman filter
- confidence-weighted class stabilization
- optional second-stage license-plate detection inside tracked vehicle crops
- annotated MP4, vehicle CSV/JSONL, plate CSV/JSONL, saved plate crops, and summary JSON

Plate OCR is not included in this version. First verify that the plate detector consistently finds clear plate crops; OCR should be added only after localization is reliable.

## Processing architecture

```text
One reference frame + ROI
    -> generate ROI-only tile coordinates once
    -> save tile_plan.json and preview image
    -> reuse the same tile coordinates for every video frame

Each video frame
    -> crop only the saved ROI tiles
    -> YOLO26m vehicle detections
    -> map detections to source-frame coordinates
    -> cross-tile duplicate merge
    -> single ByteTrack update
    -> tracked vehicle crops
    -> small plate detector
    -> associate plate box with vehicle track ID
    -> video + CSV + JSONL + plate crops
```

Do not run a separate tracker per tile. All tile detections are remapped and merged before one ByteTrack update.

## Folder architecture

```text
yolo26_midrange_roi_tracker/
├── configs/
│   ├── bytetrack_traffic.yaml
│   └── roi.example.json
├── models/
│   └── README.md
├── src/traffic_tracker/
│   ├── anpr/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── memory.py
│   ├── cli.py
│   ├── detector.py
│   ├── drawing.py
│   ├── exporter.py
│   ├── nms.py
│   ├── pipeline.py
│   ├── roi.py
│   ├── roi_drawer.py
│   ├── runtime.py
│   ├── stabilization.py
│   ├── tile_plan.py
│   ├── tile_plan_cli.py
│   ├── tiling.py
│   ├── tracking.py
│   ├── types.py
│   └── video.py
├── tests/test_core.py
├── draw_roi.py
├── plan_tiles.py
├── run_tracker.py
├── pyproject.toml
├── requirements.txt
└── setup.sh
```

## 1. Installation

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg curl libgl1 libglib2.0-0

chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

Verify commands:

```bash
traffic-draw-roi --help
traffic-plan-tiles --help
traffic-track --help
```

## 2. Draw the ROI

Polygon:

```bash
traffic-draw-roi \
  --source /home/sachin/projects/traffic-detection/test.mp4 \
  --frame-index 0 \
  --mode polygon \
  --output configs/roi.custom.json
```

Freehand curve:

```bash
traffic-draw-roi \
  --source /home/sachin/projects/traffic-detection/test.mp4 \
  --frame-index 0 \
  --mode freehand \
  --freehand-step 3 \
  --simplify-epsilon 2 \
  --output configs/roi.custom.json
```

## 3. Generate tile coordinates once

This command uses one selected video frame for the preview, generates tiles only around the ROI, and saves the coordinates:

```bash
traffic-plan-tiles \
  --source /home/sachin/projects/traffic-detection/test.mp4 \
  --roi configs/roi.custom.json \
  --output configs/tile_plan.custom.json \
  --preview configs/tile_plan.custom.jpg \
  --frame-index 0 \
  --tile-size 960 \
  --tile-overlap 0.25 \
  --roi-tile-padding 96
```

Outputs:

```text
configs/tile_plan.custom.json  # fixed source-pixel tile coordinates
configs/tile_plan.custom.jpg   # visual verification
```

The tracker loads these coordinates and reuses them for all frames. It does not regenerate the tile layout every frame.

### Why entry vehicles are handled better

- The tile grid is anchored to the ROI bounds, not the complete image grid.
- `--roi-tile-padding 96` adds context outside the ROI.
- Boundary tiles are retained even when only a narrow part intersects the ROI.
- `--roi-detection-padding` lets tracking begin slightly before the vehicle's road-contact point enters the exact ROI.

If an entry vehicle is still clipped, increase:

```bash
--roi-tile-padding 160 \
--roi-detection-padding 96 \
--tile-overlap 0.30
```

## 4. Add the license-plate model

Place a trained one-class plate detector at:

```text
models/license_plate.pt
```

The expected class is:

```text
class 0: license_plate
```

A normal COCO YOLO model cannot detect license plates because COCO has no license-plate class. Use a custom-trained plate detector. A small model such as a YOLO26n-derived plate detector is recommended because it runs once for multiple tracked vehicle crops on CPU.

## 5. Run vehicle tracking and plate detection

```bash
traffic-track \
  --source /home/sachin/projects/traffic-detection/test.mp4 \
  --roi configs/roi.custom.json \
  --tile-plan configs/tile_plan.custom.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --plate-model models/license_plate.pt \
  --plate-vehicle-classes car,motorcycle,bus,truck \
  --plate-interval 2 \
  --plate-imgsz 640 \
  --plate-conf 0.20 \
  --tile-size 960 \
  --tile-overlap 0.25 \
  --roi-tile-padding 96 \
  --roi-detection-padding 64 \
  --imgsz 640 \
  --conf 0.06 \
  --draw-tiles \
  --max-frames 200 \
  --output-dir runs/plate_test
```

The tile settings supplied to `traffic-track` must match the saved tile plan. If you change them, regenerate the plan or add:

```bash
--rebuild-tile-plan
```

## Plate-detection scheduling

The plate detector runs only on eligible, real tracked vehicle boxes. It does not run on Kalman-only predicted boxes.

```text
--plate-interval 2
```

runs the plate detector every second processed frame for each vehicle. Calls are staggered using the vehicle track ID so all vehicles are not processed on the same frame.

The previous plate box is cached and projected using the current vehicle motion between scheduled detector frames:

```text
--plate-cache-frames 10
```

## Plate search region

By default, the detector searches the lower 75% of each padded vehicle crop:

```text
--plate-search-start 0.25
```

For cameras where the plate can appear anywhere in the vehicle box:

```bash
--plate-search-full-vehicle
```

## Output files

```text
runs/plate_test/
├── tracked.mp4
├── tracks.csv
├── tracks.jsonl
├── plates.csv
├── plates.jsonl
├── plate_crops/
│   └── track_000027/
│       └── frame_00000140_conf_0.912.jpg
├── tile_plan.json                 # when --tile-plan was not supplied
├── tile_plan_preview.jpg
└── summary.json
```

`plates.csv` contains only current plate detector results. The plate columns in `tracks.csv` may show `current` or motion-projected `cached` plate boxes.

## Short vehicle-only test

Plate detection is disabled when `--plate-model` is omitted:

```bash
traffic-track \
  --source /home/sachin/projects/traffic-detection/test.mp4 \
  --roi configs/roi.custom.json \
  --tile-plan configs/tile_plan.custom.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --max-frames 100 \
  --output-dir runs/vehicle_only
```

## Important tuning

### Vehicles missed at the beginning or ROI boundary

```text
Increase --roi-tile-padding
Increase --roi-detection-padding
Increase --tile-overlap
Keep --force-boundary-tiles enabled
Do not use --mask-outside-roi initially
```

### Small or distant vehicles missed

```text
Reduce --tile-size, for example 960 -> 768
Keep --imgsz 640 or increase it if CPU speed allows
Use --one-to-many for higher recall at extra cost
```

A smaller source-pixel tile makes each object larger after resizing to the YOLO input, but increases the number of detector calls.

### Plate not detected

```text
Lower --plate-conf from 0.20 to 0.10
Use --plate-search-full-vehicle
Reduce --plate-interval to 1
Check plate crop pixel size
Fine-tune the plate detector on frames from the same camera
```

### CPU too slow

```text
Increase --plate-interval, for example 2 -> 3
Increase --tile-size to reduce tile count
Keep --tile-batch-size 1
Keep --plate-batch-size 1 or 2
Use a nano plate detector
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover ROI-anchored tiling, static tile-plan persistence, cross-tile duplicate merging, ByteTrack input conversion, class stabilization, and cached plate-box motion projection.

## Integrated plate-number OCR

Install the optional recognition dependencies:

```bash
./install_ocr.sh
source .venv/bin/activate
```

Enable OCR in the existing tracking command by adding:

```bash
--ocr \
--ocr-model en_PP-OCRv5_mobile_rec \
--ocr-min-score 0.20 \
--ocr-min-plate-width 12 \
--ocr-min-plate-height 4 \
--ocr-confirm-observations 3
```

OCR output is written to:

```text
plate_numbers.json
plate_numbers.csv
plates.jsonl
plates.csv
tracks.jsonl
tracks.csv
```

`plate_numbers.json` contains one temporally aggregated result per vehicle track. A provisional value becomes confirmed only after the configured number of matching observations, confidence threshold, and vote dominance are satisfied.
