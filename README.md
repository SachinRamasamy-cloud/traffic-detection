# Traffic Vehicle and Number-Plate Study

Offline research pipeline for:

1. Reading a road video.
2. Detecting and tracking vehicles.
3. Detecting and recognizing number plates inside vehicle crops.
4. Combining OCR observations across a vehicle track.
5. Writing structured JSON results.
6. Optionally writing an annotated MP4 and best vehicle/plate crops.

This repository is for study and experimentation. It does not issue challans, query owner records, or provide legal speed enforcement.

## Selected models

- **Vehicle detector:** `yolo26s.pt`
- **Vehicle tracker:** ByteTrack through Ultralytics
- **Plate detector:** `yolo-v9-s-608-license-plate-end2end`
- **Plate OCR:** `cct-s-v2-global-model`

The default plate model is the larger FastALPR detector because plate recall is more important than maximum speed for the first study milestone.

## Accuracy target

`80%` is an evaluation target, not a property that can be guaranteed before testing on the target videos. Accuracy depends strongly on plate pixel width, motion blur, viewing angle, illumination, camera compression, traffic occlusion, and whether the model has been fine-tuned for local plates.

The project includes `evaluate` so exact plate accuracy can be measured on manually labelled tracks.

## Project flow

```text
Video
  -> YOLO vehicle detection
  -> ByteTrack vehicle IDs
  -> crop each tracked vehicle from the original frame
  -> FastALPR plate detection + OCR
  -> quality filtering
  -> multi-frame plate consensus per vehicle ID
  -> JSON + optional annotated video + best crops
```

## Requirements

- Python 3.11 recommended
- FFmpeg installed for broad video codec support
- 16 GB RAM recommended
- NVIDIA GPU recommended but not required

The first run downloads model weights to the respective framework caches.

## Installation

### Windows PowerShell: CPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
pip install -e .
```

### Windows or Linux: NVIDIA GPU

Install the appropriate CUDA-enabled PyTorch build first, then:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-gpu.txt
pip install -e .
```

If ONNX Runtime GPU conflicts with your CUDA version, use `requirements-cpu.txt` first. Vehicle detection can still use CUDA while FastALPR uses CPU.

## Run

From the repository root:

```bash
traffic-plate-study run \
  --video "data/input.mp4" \
  --output "outputs/results.json" \
  --config "config/default.yaml" \
  --annotated-video "outputs/annotated.mp4"
```

Equivalent module command:

```bash
python -m traffic_plate_study run --video data/input.mp4 --output outputs/results.json
```

To process only the first 500 frames during a quick test:

```bash
traffic-plate-study run \
  --video data/input.mp4 \
  --output outputs/test-results.json \
  --max-frames 500
```

## Important configuration

Edit `config/default.yaml`:

```yaml
vehicle:
  model: yolo26s.pt
  image_size: 960
  confidence: 0.25

plate:
  detector_model: yolo-v9-s-608-license-plate-end2end
  ocr_model: cct-s-v2-global-model
  every_n_frames: 2

consensus:
  minimum_support: 3
  minimum_confidence: 0.80
```

For a road-only region, add a polygon:

```yaml
roi:
  polygon:
    - [220, 180]
    - [1650, 190]
    - [1900, 1050]
    - [50, 1050]
```

Coordinates are pixels in the original video.

## JSON structure

```json
{
  "video": {
    "path": "data/input.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 25.0,
    "processed_frames": 1000
  },
  "models": {
    "vehicle_detector": "yolo26s.pt",
    "vehicle_tracker": "bytetrack",
    "plate_detector": "yolo-v9-s-608-license-plate-end2end",
    "plate_ocr": "cct-s-v2-global-model"
  },
  "summary": {
    "vehicle_tracks": 28,
    "confirmed_vehicle_tracks": 25,
    "tracks_with_accepted_plate": 19
  },
  "vehicles": [
    {
      "track_id": 4,
      "vehicle_class": "car",
      "confirmed": true,
      "first_seen": {"frame_index": 106, "timestamp_ms": 4240.0},
      "last_seen": {"frame_index": 188, "timestamp_ms": 7520.0},
      "plate": {
        "text": "TN38AB1234",
        "accepted": true,
        "confidence": 0.91,
        "support_count": 6,
        "observation_count": 8
      },
      "plate_observations": []
    }
  ]
}
```

## Evaluate the 80% target

Create a ground-truth JSON:

```json
{
  "tracks": [
    {"track_id": 4, "plate_text": "TN38AB1234"},
    {"track_id": 9, "plate_text": "KL07CD4455"}
  ]
}
```

Run:

```bash
traffic-plate-study evaluate \
  --predictions outputs/results.json \
  --ground-truth data/ground-truth.json
```

The evaluator reports:

- exact plate accuracy
- accepted-track coverage
- character similarity
- false accepted results
- whether the exact-accuracy target was reached

## Improving accuracy toward 80%+

Apply these in order:

1. Use footage where plates are at least roughly 100 pixels wide in the best frames.
2. Restrict processing to the road ROI.
3. Use the original-resolution vehicle crop; do not OCR the globally resized detector input.
4. Increase `vehicle.image_size` to `1280` for distant vehicles if GPU memory permits.
5. Reduce `plate.every_n_frames` to `1` for short tracks.
6. Require at least three compatible OCR observations.
7. Fine-tune the plate detector and OCR model on manually labelled examples from the target camera.
8. Evaluate day and night footage separately.

## Tests

```bash
pytest -q
```

## Licensing note

This repository's original code is MIT licensed. The model frameworks, downloaded model weights, and datasets retain their own licences. Ultralytics uses AGPL-3.0 by default. FastALPR, Fast Plate OCR, Open Image Models, and ByteTrack repositories use permissive licences, but model-weight and dataset terms must still be reviewed independently.
