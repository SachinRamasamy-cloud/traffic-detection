# YOLO26m Static ROI-Tile Traffic Tracker with License-Plate Detection and OCR

**Version:** 0.3.0  
**Primary target:** Linux CPU study and prototyping  
**Pipeline:** YOLO26m vehicle detection + static ROI tiling + ByteTrack + license-plate detection + PaddleOCR temporal consensus

> This repository is an experimental traffic-video processing pipeline. It is suitable for research, learning, algorithm evaluation, and controlled prototyping. It is not a certified enforcement, tolling, legal-evidence, or production ANPR system.

---

## 1. Purpose

This project processes traffic video and produces:

- vehicle detections;
- persistent vehicle track IDs;
- stabilized vehicle classes;
- license-plate bounding boxes associated with vehicle track IDs;
- saved plate crops;
- OCR observations for detected plates;
- provisional or confirmed plate numbers based on temporal voting;
- annotated video, CSV, JSONL, and summary files.

The pipeline is designed mainly for:

- static CCTV cameras;
- fixed or near-fixed traffic scenes;
- CPU-only development;
- testing ROI-aware tiling;
- studying vehicle tracking and ANPR;
- evaluating plate visibility before building a production system.

It is not designed to guarantee an accurate registration number from every frame. OCR reliability depends heavily on the original plate pixel size, blur, camera angle, compression, lighting, and model generalization.

---

## 2. Current capabilities

### Implemented

- Ultralytics `yolo26m.pt` vehicle detection.
- Polygon and freehand ROI drawing.
- Static tile coordinates generated once and reused for every frame.
- ROI-only tiling instead of blindly tiling the complete image.
- Tile overlap and ROI boundary protection.
- Global-coordinate restoration after tile inference.
- Cross-tile duplicate merging.
- One ByteTrack instance for the complete frame.
- ByteTrack Kalman motion prediction.
- Low-confidence ByteTrack recovery.
- Optional short Kalman-only prediction export.
- Confidence-weighted vehicle-class stabilization.
- Second-stage license-plate detection inside tracked vehicle crops.
- Plate-to-vehicle association using `vehicle_track_id`.
- Cached plate-box projection between scheduled plate detections.
- Plate crop saving.
- Recognition-only PaddleOCR integration.
- Multiple OCR preprocessing variants.
- OCR normalization to uppercase letters and digits.
- Confidence-weighted exact-text temporal consensus.
- Provisional and confirmed plate-number states.
- Annotated MP4 output.
- Vehicle CSV and JSONL output.
- Plate CSV and JSONL output.
- Final plate-number JSON and CSV output.

### Not implemented yet

- Four-corner plate detection.
- Perspective rectification using detected plate corners.
- Character-level temporal consensus.
- Indian-number-plate-specific OCR training.
- Multi-frame super-resolution.
- Appearance-based vehicle ReID.
- Automatic lane configuration.
- Production database integration.
- Privacy retention controls.
- Enforcement-grade validation.

---

## 3. High-level architecture

```text
Input video
    |
    v
Load static ROI geometry
    |
    v
Load or build a static tile plan once
    |
    v
For every processed frame:
    |
    +--> crop saved ROI tiles
    |
    +--> YOLO26m vehicle inference on each tile
    |
    +--> convert tile detections to source-frame coordinates
    |
    +--> merge overlapping duplicate detections
    |
    +--> apply ROI acceptance rule
    |
    +--> update one ByteTrack instance
    |       |
    |       +--> Kalman prediction
    |       +--> high-confidence association
    |       +--> low-confidence recovery
    |
    +--> stabilize class per vehicle track ID
    |
    +--> crop eligible tracked vehicles
    |
    +--> run license-plate detector
    |
    +--> associate plate with vehicle track ID
    |
    +--> save plate crop
    |
    +--> run PaddleOCR recognition
    |
    +--> normalize OCR text
    |
    +--> update temporal plate-text consensus
    |
    v
Export annotated video, CSV, JSONL, summary, plate crops,
and aggregated plate-number results
```

### Important tracking rule

Do not create one tracker per tile.

The correct sequence is:

```text
Tile detections
    -> source-frame coordinate restoration
    -> duplicate merge
    -> one ByteTrack update
```

Running a separate tracker on each tile creates duplicate IDs and broken trajectories.

---

## 4. Folder structure

```text
yolo26_midrange_roi_tracker/
├── .gitignore
├── README.md
├── configs/
│   ├── bytetrack_traffic.yaml
│   └── roi.example.json
├── models/
│   ├── .gitkeep
│   ├── README.md
│   └── license_plate.pt             # downloaded separately
├── src/traffic_tracker/
│   ├── __init__.py
│   ├── cli.py
│   ├── detector.py                  # tiled YOLO vehicle inference
│   ├── drawing.py                   # annotations
│   ├── exporter.py                  # CSV/JSONL/final plate numbers
│   ├── nms.py                       # duplicate merge
│   ├── pipeline.py                  # end-to-end orchestration
│   ├── roi.py                       # ROI loading and masks
│   ├── roi_drawer.py                # interactive ROI drawing
│   ├── runtime.py                   # CPU runtime configuration
│   ├── stabilization.py             # temporal class stabilization
│   ├── tile_plan.py                 # static tile plan persistence
│   ├── tile_plan_cli.py
│   ├── tiling.py
│   ├── tracking.py                  # Ultralytics ByteTrack wrapper
│   ├── types.py
│   ├── video.py
│   └── anpr/
│       ├── __init__.py
│       ├── detector.py              # license-plate localization
│       ├── memory.py                # cached plate projection
│       ├── ocr_engine.py            # PaddleOCR recognition
│       └── temporal_consensus.py    # per-track OCR voting
├── tests/
│   └── test_core.py
├── download_model.sh
├── draw_roi.py
├── install_ocr.sh
├── plan_tiles.py
├── pyproject.toml
├── requirements.txt
├── requirements-ocr.txt
├── run_tracker.py
└── setup.sh
```

---

# 5. Models, sources, credits, and licences

## 5.1 Vehicle detector: Ultralytics YOLO26m

| Item | Value |
|---|---|
| Model used | `yolo26m.pt` |
| Purpose | Vehicle and traffic-object detection |
| Source | Ultralytics YOLO26 |
| Pretraining | COCO detection classes |
| Loader | `ultralytics.YOLO` |
| Licence | Ultralytics AGPL-3.0 or an applicable Ultralytics enterprise licence |
| Official documentation | https://docs.ultralytics.com/models/yolo26/ |
| Official repository | https://github.com/ultralytics/ultralytics |

YOLO26 is credited to the Ultralytics team. The official documentation lists `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, and `yolo26x.pt` as the released detection variants.

The project uses the medium model because it offers stronger detection accuracy than nano or small variants, while remaining more practical than large or extra-large models for local testing.

### Model acquisition

When this is run for the first time:

```python
from ultralytics import YOLO
model = YOLO("yolo26m.pt")
```

Ultralytics can download the official weights automatically.

The file may also be kept in the repository root:

```text
yolo26m.pt
```

Model files are ignored by Git.

### YOLO26 research citation

```bibtex
@misc{jocher2026ultralyticsyolo26unifiedrealtime,
  title={Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author={Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
  year={2026},
  eprint={2606.03748},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2606.03748},
  url={https://arxiv.org/abs/2606.03748}
}
```

---

## 5.2 Vehicle tracker: ByteTrack

| Item | Value |
|---|---|
| Tracker | ByteTrack |
| Purpose | Multi-object tracking and persistent IDs |
| Implementation used | Ultralytics built-in ByteTrack |
| Motion model | Linear Kalman filter |
| Association | High-score association followed by low-score recovery |
| Original authors | Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang |
| Original repository | https://github.com/FoundationVision/ByteTrack |
| Original paper | https://arxiv.org/abs/2110.06864 |
| Original repository licence | MIT |

ByteTrack helps recover brief missed detections by associating lower-confidence detections with existing tracks instead of discarding all low-score boxes.

This project does not add a separate Kalman filter on top of ByteTrack. ByteTrack already uses Kalman state prediction internally.

### Current tracker profile

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.06
new_track_thresh: 0.25
track_buffer: 90
match_thresh: 0.82
fuse_score: true
```

The vehicle detector confidence must remain less than or equal to `track_low_thresh`; otherwise, ByteTrack cannot receive the low-confidence detections required for its second association stage.

### ByteTrack citation

```bibtex
@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

---

## 5.3 License-plate detector: YOLOv11 nano fine-tuned by morsetechlab

| Item | Value |
|---|---|
| Local filename | `models/license_plate.pt` |
| Upstream filename | `license-plate-finetune-v1n.pt` |
| Purpose | License-plate bounding-box detection |
| Architecture | Fine-tuned YOLOv11n |
| Model author/publisher | `morsetechlab` |
| Source | Hugging Face model repository |
| Upstream dataset reference | Roboflow `license-plate-recognition-rxg4e` |
| Upstream dataset size reported by model card | 10,125 images |
| Model licence | AGPL-3.0 |
| Model card | https://huggingface.co/morsetechlab/yolov11-license-plate-detection |

### Download command

```bash
mkdir -p models

wget -O models/license_plate.pt \
  "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt?download=true"
```

### Verify the plate model

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("models/license_plate.pt")
print(model.names)
PY
```

Expected output:

```text
{0: 'License_Plate'}
```

### Upstream limitation notice

The model card explicitly warns that the source dataset contains train/test contamination. Published evaluation metrics may therefore be inflated and should not be treated as proof of production accuracy.

The model card also warns that:

- small or distant plates may be missed;
- real-world generalization may be lower than reported metrics;
- motorcycles and unusual plate formats are not guaranteed;
- country-specific plate styles may require fine-tuning.

This project therefore treats the model as a research baseline, not a production-grade Indian ANPR detector.

### Credit statement

When publishing results produced with this model, credit:

- `morsetechlab` for the fine-tuned YOLOv11 license-plate model;
- Ultralytics for YOLOv11 and the inference framework;
- the Roboflow dataset authors referenced by the model card.

---

## 5.4 OCR model: PaddleOCR `en_PP-OCRv5_mobile_rec`

| Item | Value |
|---|---|
| Model | `en_PP-OCRv5_mobile_rec` |
| Purpose | Recognition-only OCR on already-localized plate crops |
| Framework | PaddleOCR / PaddlePaddle |
| Language focus | English letters and numeric text |
| Model type | Lightweight mobile recognizer |
| Official documentation | https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html |
| Official repository | https://github.com/PaddlePaddle/PaddleOCR |
| PaddleOCR licence | Apache-2.0 |

The project does not ask PaddleOCR to find the plate. The YOLO plate detector localizes the plate first, and PaddleOCR only attempts to recognize the cropped text.

### Local model cache

The OCR model is downloaded automatically during the first successful OCR run and is usually cached at:

```text
~/.paddlex/official_models/en_PP-OCRv5_mobile_rec/
```

Example full path:

```text
/home/sachin/.paddlex/official_models/en_PP-OCRv5_mobile_rec/
```

### Credit statement

Credit PaddlePaddle and PaddleOCR for the OCR inference framework and the `en_PP-OCRv5_mobile_rec` recognition model.

---

## 5.5 Licence summary

| Component | Upstream licence |
|---|---|
| Ultralytics package and YOLO26 models | AGPL-3.0 or Ultralytics enterprise licence |
| Morsetechlab plate detector | AGPL-3.0 |
| Original ByteTrack repository | MIT |
| PaddleOCR | Apache-2.0 |
| PaddlePaddle | Apache-2.0 |

> The supplied project archive does not currently contain a top-level licence for the project’s own custom source code. Add an explicit `LICENSE` file before publishing or distributing the repository. Third-party models and libraries remain governed by their own licences. This section is informational, not legal advice.

---

# 6. System requirements

## Tested development profile

- Ubuntu Linux.
- x86-64 CPU.
- Python 3.13 virtual environment.
- CPU-only PyTorch.
- Ultralytics 8.4.x.
- OpenCV 4.x.
- PaddlePaddle 3.3.1.
- PaddleOCR 3.x.
- FFmpeg 7.x.

## Minimum recommended hardware

- 4 CPU cores minimum.
- 8 GB RAM minimum.
- 16 GB RAM preferred for larger videos and multiple models.
- SSD storage for model caches, crops, and outputs.

YOLO26m plus plate detection and OCR is computationally expensive on CPU. Processing may be much slower than real time.

---

# 7. Installation

## 7.1 System packages

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  ffmpeg \
  wget \
  curl \
  libgl1 \
  libglib2.0-0 \
  libxkbcommon-x11-0
```

## 7.2 Create the Python environment

From the project root:

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

The setup script installs CPU PyTorch and installs the project in editable mode.

For an existing environment after code changes:

```bash
source .venv/bin/activate
python -m pip install -e .
```

## 7.3 Install OCR dependencies

### Recommended PyPI installation

The PaddlePaddle regional mirror can be slow. The following installation uses PyPI:

```bash
source .venv/bin/activate

python -m pip install \
  paddlepaddle==3.3.1 \
  --index-url https://pypi.org/simple \
  --timeout 300 \
  --retries 10

python -m pip install \
  "paddleocr>=3.3,<4" \
  --index-url https://pypi.org/simple \
  --timeout 300 \
  --retries 10
```

### Verify OCR installation

```bash
python - <<'PY'
import paddle
import paddleocr
from paddleocr import TextRecognition

print("PaddlePaddle:", paddle.__version__)
print("PaddleOCR:", paddleocr.__version__)
print("TextRecognition import: OK")
PY
```

## 7.4 Verify CLI commands

```bash
traffic-draw-roi --help
traffic-plan-tiles --help
traffic-track --help
```

---

# 8. Model setup

## 8.1 Vehicle model

The default command uses:

```text
yolo26m.pt
```

It can be downloaded automatically by Ultralytics on first use.

## 8.2 Plate model

```bash
mkdir -p models

wget -O models/license_plate.pt \
  "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt?download=true"
```

Check the files:

```bash
ls -lh yolo26m.pt models/license_plate.pt
```

Model weights should remain ignored by Git.

---

# 9. Standard workflow for a new video

Each camera geometry should have its own ROI and tile-plan files.

Example input:

```text
/home/sachin/projects/traffic-detection/test5.mp4
```

## Step 1: draw the ROI

```bash
traffic-draw-roi \
  --source /home/sachin/projects/traffic-detection/test5.mp4 \
  --frame-index 0 \
  --mode freehand \
  --freehand-step 3 \
  --simplify-epsilon 2 \
  --output configs/roi.test5.json
```

Controls:

| Input | Action |
|---|---|
| Left mouse drag | Draw freehand ROI |
| `S` or Enter | Save |
| `R` | Reset |
| `U` or Backspace | Undo |
| `Q` or Escape | Cancel |

The ROI is stored in source-video pixel coordinates.

## Step 2: create the static tile plan

```bash
traffic-plan-tiles \
  --source /home/sachin/projects/traffic-detection/test5.mp4 \
  --roi configs/roi.test5.json \
  --output configs/tile_plan.test5.768.json \
  --preview configs/tile_plan.test5.768.jpg \
  --frame-index 0 \
  --tile-size 768 \
  --tile-overlap 0.20 \
  --roi-tile-padding 48
```

Open the preview:

```bash
xdg-open configs/tile_plan.test5.768.jpg
```

The tile plan stores source-frame coordinates. It is generated once and reused for every frame.

## Step 3: run the complete pipeline

```bash
traffic-track \
  --source /home/sachin/projects/traffic-detection/test5.mp4 \
  --roi configs/roi.test5.json \
  --tile-plan configs/tile_plan.test5.768.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --plate-model models/license_plate.pt \
  --plate-class-id 0 \
  --plate-vehicle-classes car,motorcycle,bus,truck \
  --plate-interval 2 \
  --plate-imgsz 960 \
  --plate-conf 0.10 \
  --plate-iou 0.50 \
  --plate-search-full-vehicle \
  --plate-min-vehicle-width 50 \
  --plate-min-vehicle-height 40 \
  --plate-min-width 20 \
  --plate-min-height 6 \
  --plate-min-aspect 1.5 \
  --plate-max-aspect 8.0 \
  --plate-batch-size 2 \
  --ocr \
  --ocr-model en_PP-OCRv5_mobile_rec \
  --ocr-device cpu \
  --ocr-interval 2 \
  --ocr-batch-size 2 \
  --ocr-min-score 0.50 \
  --ocr-min-text-length 5 \
  --ocr-min-plate-width 60 \
  --ocr-min-plate-height 15 \
  --ocr-confirm-observations 3 \
  --ocr-confirm-score 0.50 \
  --ocr-confirm-dominance 0.65 \
  --tile-size 768 \
  --tile-overlap 0.20 \
  --roi-tile-padding 48 \
  --roi-detection-padding 32 \
  --imgsz 640 \
  --conf 0.06 \
  --max-frames 120 \
  --output-dir runs/test5_plate_ocr_improved
```

This is a validation configuration. Remove `--max-frames 120` or set `--max-frames 0` to process the complete video.

---

# 10. Single-frame diagnostic command

A single-frame run is useful only for testing installation and output generation. It is not a reliable way to confirm a plate number.

```bash
traffic-track \
  --source /home/sachin/projects/traffic-detection/test5.mp4 \
  --roi configs/roi.test5.json \
  --tile-plan configs/tile_plan.test5.768.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --plate-model models/license_plate.pt \
  --plate-class-id 0 \
  --plate-vehicle-classes car,motorcycle,bus,truck \
  --plate-interval 1 \
  --plate-imgsz 960 \
  --plate-conf 0.03 \
  --plate-iou 0.50 \
  --plate-search-full-vehicle \
  --plate-min-vehicle-width 40 \
  --plate-min-vehicle-height 30 \
  --plate-min-width 3 \
  --plate-min-height 2 \
  --plate-min-aspect 0.5 \
  --plate-max-aspect 12 \
  --plate-batch-size 1 \
  --ocr \
  --ocr-model en_PP-OCRv5_mobile_rec \
  --ocr-device cpu \
  --ocr-interval 1 \
  --ocr-batch-size 1 \
  --ocr-min-score 0.10 \
  --ocr-min-text-length 3 \
  --ocr-min-plate-width 3 \
  --ocr-min-plate-height 2 \
  --ocr-confirm-observations 1 \
  --ocr-confirm-score 0.10 \
  --ocr-confirm-dominance 0.50 \
  --tile-size 768 \
  --tile-overlap 0.20 \
  --roi-tile-padding 48 \
  --roi-detection-padding 32 \
  --imgsz 640 \
  --conf 0.06 \
  --max-frames 1 \
  --output-dir runs/test5_single_frame_ocr
```

> A result marked `confirmed` in this diagnostic mode only means it passed the deliberately weak one-frame thresholds. It does not mean the text is correct.

---

# 11. Vehicle-only command

Omit `--plate-model` and `--ocr`:

```bash
traffic-track \
  --source /home/sachin/projects/traffic-detection/test5.mp4 \
  --roi configs/roi.test5.json \
  --tile-plan configs/tile_plan.test5.768.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --tile-size 768 \
  --tile-overlap 0.20 \
  --roi-tile-padding 48 \
  --roi-detection-padding 32 \
  --imgsz 640 \
  --conf 0.06 \
  --max-frames 100 \
  --output-dir runs/test5_vehicle_only
```

---

# 12. How static ROI tiling works

## Why tiling is used

Small and distant vehicles can occupy too few pixels when the complete frame is resized to the detector input size.

Example:

```text
1280x720 frame -> resize to 640
```

A small object becomes approximately half its original width and height.

With two 768-pixel source tiles, each tile is independently resized to 640, so objects inside a tile occupy a larger fraction of the model input.

## Why the tile plan is static

For a fixed camera, the ROI and useful image regions do not change every frame. The project therefore:

1. reads one reference frame;
2. creates ROI-aware tile coordinates;
3. stores the coordinates in JSON;
4. reuses those coordinates for every frame.

The model still runs on the tile image content for every frame. Only tile geometry generation is avoided.

## Recommended CPU tile count

Aim for approximately:

```text
2 to 4 tiles
```

Eight or more tiles with YOLO26m on CPU can become extremely slow.

## Main tiling parameters

| Parameter | Effect |
|---|---|
| `--tile-size` | Smaller values increase small-object zoom but create more tiles |
| `--tile-overlap` | Protects objects near tile boundaries |
| `--roi-tile-padding` | Adds visual context outside the ROI when selecting tiles |
| `--roi-detection-padding` | Allows tracks to begin slightly before exact ROI entry |
| `--force-boundary-tiles` | Retains low-coverage ROI entry/exit tiles |
| `--max-tiles` | Limits tile count; `0` means no explicit limit |

---

# 13. How vehicle tracking works

ByteTrack receives merged full-frame detections once per processed frame.

The configured stages are:

1. predict active track positions with a Kalman filter;
2. match high-confidence detections to tracks;
3. retry unmatched tracks using low-confidence detections;
4. retain unmatched tracks for `track_buffer` frames;
5. create new tracks only above `new_track_thresh`.

This improves short-term continuity when a vehicle is briefly occluded or receives a weak detector score.

ByteTrack does not use an appearance embedding in this project. A vehicle that disappears for a long time may return with a new ID.

---

# 14. How license-plate detection works

The plate detector is not run on the complete frame by default.

For each eligible current vehicle track:

1. validate vehicle class;
2. validate vehicle dimensions;
3. schedule plate inference using `--plate-interval`;
4. pad the vehicle bounding box;
5. crop the full vehicle or lower vehicle region;
6. run the plate detector on the crop;
7. select the highest-confidence valid plate candidate;
8. restore the plate box to source-frame coordinates;
9. associate it with `vehicle_track_id`;
10. save the crop when configured.

## Plate scheduling

```text
--plate-interval 2
```

runs plate detection approximately every second processed frame per track. Scheduling is staggered using track IDs.

## Plate-cache behavior

The last detected plate box can be projected with vehicle motion for a limited number of frames:

```text
--plate-cache-frames 10
```

A cached box is useful for visualization. It is not a new detector observation.

---

# 15. How OCR works

## Recognition sequence

```text
Plate crop
    -> quality calculation
    -> resize and preprocessing variants
    -> PaddleOCR recognition
    -> normalize A-Z and 0-9
    -> acceptance checks
    -> weighted temporal consensus per vehicle track ID
```

## OCR preprocessing variants

The default list is:

```text
colour,gray,clahe,sharpened
```

The recognizer runs on enabled variants and chooses a candidate using normalized text presence and OCR confidence.

For poor-quality plates, aggressive preprocessing can produce unrelated strings. A conservative configuration can be used:

```bash
--ocr-variants colour,gray,clahe
```

## OCR normalization

The current normalizer:

- converts text to uppercase;
- removes spaces and punctuation;
- retains only `A-Z` and `0-9`.

Example:

```text
"KL 07 AB-1234" -> "KL07AB1234"
```

## OCR acceptance

An OCR read is accepted only when it satisfies configured checks such as:

- minimum OCR confidence;
- minimum and maximum normalized length;
- minimum plate width and height;
- optional regular-expression pattern;
- optional minimum sharpness.

## Temporal consensus

Accepted reads are grouped by exact normalized text for each vehicle track.

The weighted score is based on:

```text
OCR confidence
x plate-detector confidence
x crop-quality score
```

The winning exact string is marked:

- `provisional` when there is insufficient support;
- `confirmed` when observation count, average confidence, and dominance pass the configured thresholds.

### Recommended confirmation values

```bash
--ocr-confirm-observations 3 \
--ocr-confirm-score 0.50 \
--ocr-confirm-dominance 0.65
```

A single observation should not be considered reliable.

---

# 16. Plate readability requirements

Plate detection can succeed even when OCR cannot.

A detector may identify a plate-shaped region at only a few pixels high. That does not mean individual characters exist in the image.

## Practical OCR thresholds

Use OCR only when the original plate crop is approximately at least:

```text
60 pixels wide
15 pixels high
```

A better target is:

```text
100+ pixels wide
25-30+ pixels high
```

Upscaling a `30x8` plate to `300x80` does not recreate missing character strokes.

## Capture-quality factors

Reliable ANPR normally requires:

- sufficient optical zoom;
- fast shutter speed;
- low motion blur;
- controlled camera angle;
- correct focus;
- adequate plate illumination;
- multiple usable frames;
- country-specific recognition training.

---

# 17. Output structure

Example:

```text
runs/test5_plate_ocr_improved/
├── tracked.mp4
├── tracks.csv
├── tracks.jsonl
├── plates.csv
├── plates.jsonl
├── plate_numbers.csv
├── plate_numbers.json
├── summary.json
├── tile_plan_preview.jpg
└── plate_crops/
    ├── track_000004/
    │   └── frame_00000086_conf_0.755.jpg
    └── track_000007/
        └── frame_00000065_conf_0.778.jpg
```

## `tracked.mp4`

Annotated output showing:

- ROI boundary;
- optional tile rectangles;
- vehicle boxes;
- vehicle track IDs;
- stabilized classes;
- plate boxes;
- provisional plate text with `?`;
- confirmed plate text without `?`.

## `tracks.jsonl`

One record per processed frame containing:

- vehicle objects;
- current track states;
- optional predicted states;
- raw and stabilized classes;
- current or cached plate association;
- OCR consensus fields.

## `tracks.csv`

Flattened vehicle-track table with fields such as:

- frame and timestamp;
- track ID;
- tracked/predicted state;
- raw and stabilized class;
- bounding box;
- ROI state;
- plate box state;
- plate text and consensus status.

## `plates.jsonl`

One line per processed frame containing current plate-detector observations and OCR information.

## `plates.csv`

Flattened current plate observations including:

- vehicle track ID;
- plate detector confidence;
- plate box;
- vehicle box;
- crop path;
- OCR raw and normalized text;
- OCR confidence;
- preprocessing variant;
- accepted flag;
- crop quality and sharpness;
- current consensus result.

## `plate_numbers.json`

One final temporal result per vehicle track with at least one accepted OCR read.

Example:

```json
{
  "plate_numbers": [
    {
      "vehicle_track_id": 7,
      "plate_text": "KL07AB1234",
      "status": "confirmed",
      "confidence": 0.84,
      "weighted_score": 1.92,
      "observation_count": 4,
      "total_accepted_observations": 5,
      "dominance": 0.79,
      "first_frame": 23,
      "last_frame": 61
    }
  ]
}
```

## `summary.json`

Contains:

- input properties;
- frame count;
- elapsed time;
- processing FPS;
- tile-plan information;
- vehicle and plate counts;
- OCR attempts and accepted reads;
- output file paths.

---

# 18. Viewing results

## Final plate numbers

```bash
python -m json.tool \
  runs/test5_plate_ocr_improved/plate_numbers.json
```

## Plate observations

```bash
column -s, -t \
  < runs/test5_plate_ocr_improved/plates.csv \
  | less -S
```

## Plate crops

```bash
xdg-open runs/test5_plate_ocr_improved/plate_crops
```

## Annotated video

```bash
xdg-open runs/test5_plate_ocr_improved/tracked.mp4
```

## Run summary

```bash
python -m json.tool \
  runs/test5_plate_ocr_improved/summary.json
```

---

# 19. Important parameters

## Vehicle detector

| Argument | Default | Purpose |
|---|---:|---|
| `--model` | `yolo26m.pt` | Vehicle detector weights |
| `--imgsz` | `640` | YOLO input resolution |
| `--conf` | `0.06` | Minimum vehicle detector score |
| `--iou` | `0.70` | Detector IoU setting |
| `--classes` | all | Class filter |
| `--one-to-many` | off | Higher-recall YOLO26 head at extra cost |

## Static tiling

| Argument | Default | Purpose |
|---|---:|---|
| `--tile-size` | `960` | Source-pixel tile size |
| `--tile-overlap` | `0.25` | Tile overlap |
| `--tile-batch-size` | `1` | Vehicle tile inference batch |
| `--roi-tile-padding` | `96` | Tile-selection context outside ROI |
| `--roi-detection-padding` | `64` | Detection acceptance beyond exact ROI |
| `--merge-iou` | `0.55` | Cross-tile duplicate merge threshold |
| `--max-tiles` | `0` | Maximum selected tiles; `0` is unlimited |

## Tracking

| Argument | Default | Purpose |
|---|---:|---|
| `--tracker` | bundled YAML | ByteTrack configuration |
| `--vid-stride` | `1` | Process every Nth frame |
| `--prediction-frames` | `0` | Draw/export short lost-track predictions |
| `--class-history` | `20` | Class-voting history |
| `--class-min-observations` | `4` | Evidence needed to stabilize class |
| `--class-switch-ratio` | `1.75` | Evidence required to change class |

## Plate detection

| Argument | Default | Purpose |
|---|---:|---|
| `--plate-model` | empty | Enables plate detection when provided |
| `--plate-imgsz` | `640` | Plate detector input size |
| `--plate-conf` | `0.20` | Plate confidence threshold |
| `--plate-class-id` | `0` | Plate class ID |
| `--plate-interval` | `2` | Plate detector schedule |
| `--plate-batch-size` | `2` | Vehicle-crop batch size |
| `--plate-search-full-vehicle` | off | Search complete vehicle crop |
| `--plate-min-width` | `8` | Minimum detected plate width |
| `--plate-min-height` | `4` | Minimum detected plate height |
| `--plate-min-aspect` | `0.8` | Minimum plate aspect ratio |
| `--plate-max-aspect` | `10.0` | Maximum plate aspect ratio |

## OCR

| Argument | Default | Purpose |
|---|---:|---|
| `--ocr` | off | Enable OCR |
| `--ocr-model` | `en_PP-OCRv5_mobile_rec` | PaddleOCR recognition model |
| `--ocr-batch-size` | `4` | OCR variant batch size |
| `--ocr-interval` | `1` | Minimum frames between OCR attempts per track |
| `--ocr-max-reads-per-track` | `12` | OCR attempt cap per track |
| `--ocr-min-score` | `0.20` | Minimum accepted OCR confidence |
| `--ocr-min-text-length` | `4` | Minimum normalized length |
| `--ocr-min-plate-width` | `12` | Minimum plate width for OCR |
| `--ocr-min-plate-height` | `4` | Minimum plate height for OCR |
| `--ocr-variants` | four variants | OCR preprocessing list |
| `--ocr-confirm-observations` | `3` | Matching reads needed for confirmation |
| `--ocr-confirm-score` | `0.50` | Minimum winner average confidence |
| `--ocr-confirm-dominance` | `0.60` | Minimum weighted vote dominance |

Run this for the authoritative option list:

```bash
traffic-track --help
```

---

# 20. Tuning guidance

## Vehicles missed at the ROI entrance

Try:

```bash
--roi-tile-padding 96 \
--roi-detection-padding 64 \
--tile-overlap 0.30
```

Keep boundary tiles enabled and avoid masking outside the ROI during initial tuning.

## Small vehicles missed

- Reduce `--tile-size`.
- Keep or increase `--imgsz`.
- Use `--one-to-many` for higher recall.
- Verify the entry region is included in the tile preview.

Trade-off: smaller tiles mean more model calls.

## CPU is too slow

- Increase tile size to reduce tile count.
- Target two to four tiles.
- Increase `--plate-interval` to 3 or 5.
- Increase `--ocr-interval`.
- Reduce `--plate-imgsz` from 960 to 640.
- Use `--vid-stride 2` only after evaluating tracking impact.
- Disable `--draw-tiles` after debugging.
- Use a smaller vehicle model for speed experiments.

## Plate detector finds no plates

- Check the model class with `model.names`.
- Use `--plate-class-id 0` for this model.
- Lower `--plate-conf` temporarily.
- Use `--plate-search-full-vehicle`.
- Reduce vehicle minimum dimensions during diagnostics.
- Inspect the actual plate pixel size.

## OCR produces random words

- Increase minimum plate size.
- Require multi-frame confirmation.
- Increase `--ocr-min-score`.
- Increase `--ocr-min-text-length`.
- Remove `sharpened` from OCR variants.
- Use an optional plate-format regex.
- Fine-tune a plate-specific OCR model.
- Improve camera capture quality.

## Better OCR validation configuration

```bash
--ocr-min-score 0.50 \
--ocr-min-text-length 5 \
--ocr-min-plate-width 60 \
--ocr-min-plate-height 15 \
--ocr-confirm-observations 3 \
--ocr-confirm-score 0.50 \
--ocr-confirm-dominance 0.65 \
--ocr-variants colour,gray,clahe
```

---

# 21. Troubleshooting

## Tile-plan mismatch

Error:

```text
Existing tile plan does not match the current video dimensions, ROI geometry.
```

Cause:

- the tile plan belongs to another video size;
- the ROI changed;
- tile size or overlap changed.

Fix:

Create a unique ROI and tile plan for the new video:

```text
configs/roi.<video>.json
configs/tile_plan.<video>.768.json
```

Or rebuild intentionally:

```bash
--rebuild-tile-plan
```

## Plate model missing

Error:

```text
No such file or directory: models/license_plate.pt
```

Fix:

```bash
mkdir -p models
wget -O models/license_plate.pt \
  "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt?download=true"
```

## PaddleOCR missing

Error:

```text
ModuleNotFoundError: No module named 'paddleocr'
```

Fix:

```bash
source .venv/bin/activate
python -m pip install paddlepaddle==3.3.1 --index-url https://pypi.org/simple
python -m pip install "paddleocr>=3.3,<4" --index-url https://pypi.org/simple
```

## No progress log for several minutes

The default progress logger reports every 25 processed frames. A large number of tiles plus plate detection and OCR can make the first log take several minutes.

Check CPU activity:

```bash
htop
```

Check output growth:

```bash
watch -n 5 'wc -l runs/<run-name>/tracks.jsonl 2>/dev/null'
```

## OpenCV Wayland or font warnings

Warnings such as:

```text
Ignoring XDG_SESSION_TYPE=wayland
QFontDatabase: Cannot find font directory
```

can appear while the ROI window still works. They are normally non-fatal.

## `ccache` warning from Paddle

The `No ccache found` warning is not an OCR failure. It only indicates that compiled-extension rebuilds may be slower.

---

# 22. Tests

Run the unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The current tests cover core geometry and integration helpers such as:

- ROI coordinate handling;
- ROI-anchored tiling;
- tile-plan persistence;
- duplicate merging;
- tracker input conversion;
- class stabilization;
- plate-memory projection.

A complete accuracy benchmark requires a labelled traffic and ANPR dataset.

---

# 23. Known technical limitations

## Vehicle detection

`yolo26m.pt` is COCO-pretrained. Its vehicle taxonomy is limited to COCO classes such as car, motorcycle, bus, truck, and bicycle. It does not natively provide a detailed Indian traffic taxonomy.

## Tracking

ByteTrack has no appearance ReID in this project. Long occlusions and similar vehicles may produce ID changes.

## Plate detection

The current detector returns an axis-aligned rectangle and chooses the highest-confidence valid candidate within each vehicle crop.

It does not currently select a crop based on the strongest combined readability score. A smaller high-confidence crop can therefore be saved instead of a larger and more readable crop.

## OCR

The OCR recognizer is a generic English recognizer, not an Indian-registration-specific model.

The current temporal method votes on exact complete strings. It does not perform character-position consensus, edit-distance clustering, or state/district pattern reasoning.

## Plate geometry

No four-corner plate model or perspective correction is implemented. Strongly skewed plates may remain difficult to read.

## Enhancement

The pipeline resizes and preprocesses crops but does not reconstruct true missing detail. Super-resolution output should not be treated as forensic truth.

## Production readiness

The pipeline does not include:

- authentication;
- encryption policies;
- audit logging;
- data-retention enforcement;
- access controls;
- legal compliance workflows;
- formal accuracy validation;
- health monitoring or distributed execution.

---

# 24. Recommended next improvements

Priority order:

1. Change best-crop selection from detector confidence only to a readability score.
2. Keep the best 5-10 crops per vehicle track.
3. Add plate padding before OCR.
4. Train or use a four-corner plate detector.
5. Apply perspective rectification.
6. Add edit-distance and character-level temporal consensus.
7. Add Indian registration-format-aware validation without hard rejecting special formats.
8. Fine-tune a recognizer on Indian plates and the target camera.
9. Add optional plate-track super-resolution for research.
10. Export vehicle, plate, and OCR observations into the SaveTrax backend schema.

---

# 25. Privacy and responsible use

License-plate numbers can be sensitive personal or vehicle-linked data.

Before processing real-world footage:

- confirm that collection and processing are lawful;
- use the system only for authorized purposes;
- restrict access to videos, crops, and recognized text;
- define a retention period;
- avoid publishing identifiable plate data;
- encrypt stored and transmitted data where appropriate;
- document false-positive and false-negative risk;
- do not use one-frame OCR as evidence;
- obtain legal review before enforcement or commercial deployment.

---

# 26. Credits

This project builds on the work of:

- **Ultralytics** for the YOLO inference framework and YOLO26 vehicle model.
- **Glenn Jocher, Jing Qiu, Mengyu Liu, Shuai Lyu, Fatih Cagatay Akyon, and Muhammet Esat Kalfaoglu** for the YOLO26 paper.
- **Yifu Zhang and the ByteTrack authors** for the ByteTrack multi-object tracking method.
- **Ultralytics contributors** for the integrated ByteTrack implementation used by this pipeline.
- **morsetechlab** for the fine-tuned YOLOv11 license-plate detector.
- **Roboflow dataset contributors** referenced by the plate model card.
- **PaddlePaddle and PaddleOCR contributors** for PaddlePaddle, PaddleOCR, and `en_PP-OCRv5_mobile_rec`.
- **OpenCV**, **PyTorch**, **NumPy**, **PyYAML**, and **FFmpeg** contributors.

The custom project code coordinates these components for static ROI tiling, traffic tracking, plate association, OCR aggregation, and export.

---

# 27. Reference links

- Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/
- Ultralytics repository: https://github.com/ultralytics/ultralytics
- YOLO26 paper: https://arxiv.org/abs/2606.03748
- ByteTrack repository: https://github.com/FoundationVision/ByteTrack
- ByteTrack paper: https://arxiv.org/abs/2110.06864
- License-plate model card: https://huggingface.co/morsetechlab/yolov11-license-plate-detection
- Plate model files: https://huggingface.co/morsetechlab/yolov11-license-plate-detection/tree/main
- PaddleOCR repository: https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR text-recognition documentation: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html
- PaddlePaddle package: https://pypi.org/project/paddlepaddle/
- PaddleOCR package: https://pypi.org/project/paddleocr/

---

# 28. Quick command summary

```bash
# Activate
cd /home/sachin/projects/traffic-detection
source .venv/bin/activate

# Draw ROI
traffic-draw-roi \
  --source test5.mp4 \
  --mode freehand \
  --output configs/roi.test5.json

# Build tile plan
traffic-plan-tiles \
  --source test5.mp4 \
  --roi configs/roi.test5.json \
  --output configs/tile_plan.test5.768.json \
  --preview configs/tile_plan.test5.768.jpg \
  --tile-size 768 \
  --tile-overlap 0.20 \
  --roi-tile-padding 48

# Run
traffic-track \
  --source test5.mp4 \
  --roi configs/roi.test5.json \
  --tile-plan configs/tile_plan.test5.768.json \
  --model yolo26m.pt \
  --classes car,motorcycle,bus,truck,bicycle \
  --plate-model models/license_plate.pt \
  --plate-class-id 0 \
  --plate-search-full-vehicle \
  --ocr \
  --ocr-model en_PP-OCRv5_mobile_rec \
  --ocr-confirm-observations 3 \
  --max-frames 120 \
  --output-dir runs/test5_complete
```

---

## Current project status

```text
Vehicle detection: implemented
ROI drawing: implemented
Static ROI tile plan: implemented
Cross-tile duplicate merge: implemented
ByteTrack and Kalman tracking: implemented
Class stabilization: implemented
License-plate localization: implemented
Plate crop export: implemented
PaddleOCR integration: implemented
Temporal exact-string consensus: implemented
Reliable Indian ANPR: requires further model and capture improvements
```
