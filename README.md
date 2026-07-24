# YOLO26 CPU Video Tracker

A study-oriented Linux CPU baseline using:

- **Detector:** Ultralytics YOLO26n, pretrained on COCO
- **Tracker:** ByteTrack
- **Input:** video file
- **Outputs:** annotated MP4, JSONL, CSV, and run summary

YOLO26n is selected because it is the smallest YOLO26 detection model and is the appropriate starting point for CPU inference. ByteTrack is used because it does not require a ReID network or camera-motion computation.

## 1. What to learn

Study these topics in this order:

1. Python, NumPy, OpenCV, video frames, FPS, codecs, and pixel coordinates.
2. Object detection: bounding boxes, class IDs, confidence, IoU, precision, recall, mAP.
3. YOLO inference: image resizing, confidence filtering, model sizes, and COCO classes.
4. Multi-object tracking: track IDs, data association, Kalman filters, occlusion, ID switches, track birth/death.
5. ByteTrack: high-confidence association followed by low-confidence recovery.
6. Evaluation: detection mAP plus tracking metrics such as IDF1, HOTA, MOTA, and ID switches.
7. Deployment: CPU profiling, OpenVINO/ONNX export, frame stride, resolution, threading, logging, and output schemas.
8. Custom training: annotation quality, train/validation/test splits, class imbalance, augmentation, and domain shift.

## 2. Linux setup

Recommended: Ubuntu 22.04/24.04, Python 3.11 or 3.12, and FFmpeg installed.

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg curl

chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

The setup installs CPU-only PyTorch and Ultralytics 8.4.104 with headless OpenCV.

## 3. Model download

You normally do not need to download the model manually. The first run automatically downloads `yolo26n.pt`.

Manual download:

```bash
./download_model.sh
```

Official model URL:

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

## 4. Run tracking

All COCO classes:

```bash
python tracker.py \
  --source /path/to/input.mp4 \
  --output-dir runs/example
```

Traffic classes only:

```bash
python tracker.py \
  --source /path/to/traffic.mp4 \
  --classes car,motorcycle,bus,truck \
  --conf 0.25 \
  --imgsz 640 \
  --output-dir runs/traffic
```

Quick CPU test on the first 200 processed frames:

```bash
python tracker.py \
  --source /path/to/input.mp4 \
  --max-frames 200 \
  --output-dir runs/test
```

Faster but less reliable tracking:

```bash
python tracker.py \
  --source /path/to/input.mp4 \
  --vid-stride 2 \
  --imgsz 512 \
  --output-dir runs/faster
```

Frame skipping can increase ID switches, so `--vid-stride 1` is the tracking-quality default.

## 5. Outputs

Each run writes:

```text
runs/example/
├── tracked.mp4
├── tracks.jsonl
├── tracks.csv
└── summary.json
```

Each tracked object contains:

- frame index and timestamp
- persistent track ID
- class ID and class name
- detection confidence
- bounding box `[x1, y1, x2, y2]`
- box center, width, and height

## 6. CPU tuning

Start with:

- `yolo26n.pt`
- `imgsz=640`
- `vid_stride=1`
- ByteTrack
- traffic class filtering when applicable

If processing is too slow, change one variable at a time:

1. Reduce `--imgsz` from 640 to 512 or 416.
2. Use `--classes` to ignore irrelevant categories.
3. Use `--vid-stride 2`, accepting weaker temporal continuity.
4. Disable annotated-video encoding with `--no-video` while benchmarking inference.
5. Export the detector to OpenVINO after validating the PyTorch baseline.

Do not start with YOLO26m/l/x on CPU. Their accuracy is higher, but their latency is substantially greater.

## 7. When pretrained COCO is insufficient

COCO weights detect common classes, not every domain-specific category. Fine-tune a custom model when:

- the required object class is absent from COCO;
- the camera angle is unusual;
- objects are very small or dense;
- night/rain/blur conditions dominate;
- class definitions differ from COCO.

For a credible study, create separate train, validation, and held-out test sets. Evaluate detection and tracking independently before tuning the full pipeline.

## 8. License note

Ultralytics provides AGPL-3.0 and enterprise licensing options. For study and research, review the AGPL-3.0 terms before redistributing or deploying the software.
