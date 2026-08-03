# YOLO26m Static ROI-Tile Traffic Tracker with License-Plate Detection and OCR

**Version:** `0.3.0`
**Primary target:** Linux CPU research and prototyping
**Pipeline:** YOLO26m vehicle detection → static ROI tiling → ByteTrack → license-plate detection → PaddleOCR temporal consensus

## Overview

This repository contains an experimental traffic-video analytics pipeline designed for static CCTV footage and CPU-based research environments.

The system detects and tracks vehicles, stabilizes vehicle classes across frames, detects license plates inside tracked vehicle regions, performs OCR on saved plate crops, and derives provisional or confirmed plate numbers using confidence-weighted temporal voting.

The project is intended for:

* traffic-video research;
* algorithm development;
* controlled prototyping;
* ROI-aware tiled inference evaluation;
* vehicle-tracking experiments;
* license-plate visibility assessment;
* early-stage ANPR feasibility studies.

> **Important:** This project is not a certified enforcement, tolling, legal-evidence, access-control, or production ANPR system.

---

## 1. Purpose

The pipeline processes traffic video and generates:

* vehicle detections;
* persistent vehicle track IDs;
* temporally stabilized vehicle classes;
* license-plate bounding boxes linked to vehicle track IDs;
* saved plate-image crops;
* frame-level OCR observations;
* provisional and confirmed plate numbers;
* annotated video output;
* CSV and JSONL datasets;
* processing summaries.

The implementation is optimized primarily for:

* static CCTV cameras;
* fixed or near-fixed traffic scenes;
* CPU-only development;
* manually configured road ROIs;
* repeated use of static tile coordinates;
* evaluation of tracking and ANPR quality before production deployment.

The system does not guarantee that a correct registration number will be extracted from every vehicle or frame.

OCR accuracy depends heavily on:

* plate pixel dimensions;
* motion blur;
* vehicle speed;
* camera angle;
* plate orientation;
* lighting conditions;
* shadows and reflections;
* video compression;
* occlusion;
* detector generalization;
* OCR model generalization.

---

## 2. Processing Pipeline

```text
Input Traffic Video
        │
        ▼
Manual Polygon or Freehand ROI
        │
        ▼
Static ROI Tile Generation
        │
        ▼
YOLO26m Vehicle Detection
        │
        ▼
Global Coordinate Restoration
        │
        ▼
Cross-Tile Duplicate Merging
        │
        ▼
Full-Frame ByteTrack Association
        │
        ▼
Vehicle-Class Stabilization
        │
        ▼
Tracked Vehicle Crop Extraction
        │
        ▼
License-Plate Detection
        │
        ▼
Plate-to-Vehicle Association
        │
        ▼
Plate Crop Preprocessing
        │
        ▼
PaddleOCR Recognition
        │
        ▼
OCR Normalization and Temporal Voting
        │
        ▼
Provisional or Confirmed Plate Number
        │
        ▼
Video, CSV, JSONL, Images, and Summary Outputs
```

---

## 3. Core Design

### 3.1 Static ROI-aware tiling

The road region is defined using a polygon or freehand ROI.

Tile coordinates are generated once during initialization and reused for every frame. This avoids recalculating tile geometry throughout the video and reduces unnecessary inference outside the configured traffic region.

The tiling implementation includes:

* ROI-only coverage;
* configurable tile overlap;
* protection around ROI boundaries;
* clipping against frame dimensions;
* global-coordinate restoration;
* cross-tile duplicate suppression.

Unlike full-frame blind tiling, this approach concentrates compute on the road area and reduces irrelevant detections from buildings, vegetation, sky, sidewalks, and other non-road regions.

### 3.2 Vehicle detection

Vehicle detection is performed using:

```text
yolo26m.pt
```

Each tile is processed independently. Tile-relative detections are projected back into the original frame coordinate system before tracking.

### 3.3 Cross-tile merging

Objects near overlapping tile boundaries may be detected more than once.

The pipeline merges duplicate detections after restoring them to full-frame coordinates. This prevents duplicate detections from being sent independently to the tracker.

### 3.4 Full-frame tracking

A single ByteTrack instance processes all merged detections for the complete frame.

This is important because separate trackers per tile would create inconsistent identities when vehicles move between tiles.

The tracking stage supports:

* Kalman motion prediction;
* high-confidence association;
* low-confidence detection recovery;
* configurable track buffering;
* short prediction-only continuation;
* optional Kalman-only row export;
* one persistent ID namespace across the complete ROI.

### 3.5 Vehicle-class stabilization

Per-frame class predictions may fluctuate because of:

* partial occlusion;
* tile-edge clipping;
* low resolution;
* unusual vehicle geometry;
* detector uncertainty;
* changing viewpoint.

The pipeline applies confidence-weighted temporal stabilization for each vehicle track.

Instead of treating every frame-level class prediction as final, class evidence is accumulated over time and used to derive a more stable track-level vehicle class.

### 3.6 License-plate detection

License-plate detection is performed as a second-stage operation inside tracked vehicle crops.

Each plate detection is associated with:

```text
vehicle_track_id
```

This allows plate detections, crops, and OCR observations to remain linked to the corresponding vehicle trajectory.

Plate detection may be scheduled rather than executed on every frame. Between scheduled detections, the most recent plate box can be projected using the updated vehicle bounding box.

### 3.7 OCR recognition

PaddleOCR is used in recognition-only mode.

Detected plate crops are passed through multiple preprocessing variants to improve OCR robustness. Depending on the configuration, these variants may include:

* original crop;
* resized crop;
* grayscale conversion;
* contrast enhancement;
* sharpening;
* thresholding;
* denoising.

OCR output is normalized to uppercase alphanumeric text:

```text
A-Z
0-9
```

Spaces, punctuation, and unsupported symbols are removed before temporal aggregation.

### 3.8 Temporal OCR consensus

A single OCR observation is not treated as a reliable final plate number.

OCR observations are accumulated for each tracked vehicle and combined using confidence-weighted exact-text voting.

The current implementation supports two output states:

* **Provisional:** the available OCR evidence is not yet strong enough for final confirmation.
* **Confirmed:** the same normalized OCR result has accumulated sufficient temporal support and confidence.

This reduces the impact of isolated OCR errors but does not eliminate systematic recognition mistakes.

---

## 4. Current Capabilities

### Implemented

#### ROI and tiling

* Polygon ROI drawing.
* Freehand ROI drawing.
* Static tile generation.
* Tile reuse across all frames.
* ROI-only tiled inference.
* Configurable tile overlap.
* ROI boundary protection.
* Frame-boundary clipping.
* Global-coordinate restoration.

#### Vehicle detection and tracking

* Ultralytics YOLO26m vehicle detection.
* Cross-tile duplicate merging.
* One ByteTrack instance for the complete frame.
* ByteTrack Kalman motion prediction.
* Low-confidence ByteTrack recovery.
* Configurable track persistence.
* Optional short Kalman-only prediction export.
* Persistent vehicle track IDs.
* Confidence-weighted vehicle-class stabilization.

#### License-plate processing

* Second-stage plate detection within tracked vehicle crops.
* Plate-to-vehicle association.
* Plate records linked through `vehicle_track_id`.
* Scheduled plate detection.
* Cached plate-box projection between detection intervals.
* Plate crop extraction.
* Plate crop saving.

#### OCR

* PaddleOCR recognition-only integration.
* Multiple OCR preprocessing variants.
* OCR confidence collection.
* Uppercase alphanumeric normalization.
* Confidence-weighted exact-text temporal voting.
* Provisional plate-number state.
* Confirmed plate-number state.

#### Outputs

* Annotated MP4 video.
* Vehicle detection and tracking CSV.
* Vehicle detection and tracking JSONL.
* Plate detection CSV.
* Plate detection JSONL.
* Saved plate crops.
* Frame-level OCR observations.
* Final plate-number JSON.
* Final plate-number CSV.
* Processing summary files.

---

## 5. Output Structure

A typical processing run produces an output structure similar to:

```text
outputs/
├── annotated/
│   └── annotated_video.mp4
│
├── vehicles/
│   ├── vehicle_tracks.csv
│   └── vehicle_tracks.jsonl
│
├── plates/
│   ├── plate_detections.csv
│   ├── plate_detections.jsonl
│   └── crops/
│       ├── track_000001/
│       ├── track_000002/
│       └── ...
│
├── ocr/
│   ├── ocr_observations.csv
│   ├── ocr_observations.jsonl
│   ├── final_plate_numbers.csv
│   └── final_plate_numbers.json
│
└── summaries/
    └── processing_summary.json
```

The exact directory and file names may differ based on runtime configuration.

---

## 6. Vehicle Output Data

Vehicle output records may contain fields such as:

| Field                  | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `frame_index`          | Zero-based source-video frame index                              |
| `timestamp_s`          | Frame timestamp in seconds                                       |
| `vehicle_track_id`     | Persistent ByteTrack vehicle identity                            |
| `vehicle_class`        | Stabilized vehicle class                                         |
| `detection_class`      | Current frame-level detector class                               |
| `detection_confidence` | Vehicle detection confidence                                     |
| `x1`                   | Left vehicle-box coordinate                                      |
| `y1`                   | Top vehicle-box coordinate                                       |
| `x2`                   | Right vehicle-box coordinate                                     |
| `y2`                   | Bottom vehicle-box coordinate                                    |
| `box_width`            | Vehicle-box width in pixels                                      |
| `box_height`           | Vehicle-box height in pixels                                     |
| `is_predicted`         | Whether the row came from prediction without a matched detection |
| `track_age`            | Number of frames associated with the track                       |
| `track_state`          | Current tracking state                                           |

---

## 7. Plate Output Data

Plate detection and OCR records may contain:

| Field                        | Description                              |
| ---------------------------- | ---------------------------------------- |
| `frame_index`                | Source-video frame index                 |
| `timestamp_s`                | Frame timestamp                          |
| `vehicle_track_id`           | Associated vehicle identity              |
| `plate_detection_confidence` | Plate detector confidence                |
| `plate_x1`                   | Left plate-box coordinate                |
| `plate_y1`                   | Top plate-box coordinate                 |
| `plate_x2`                   | Right plate-box coordinate               |
| `plate_y2`                   | Bottom plate-box coordinate              |
| `plate_crop_path`            | Saved plate-image path                   |
| `ocr_raw_text`               | Unmodified OCR result                    |
| `ocr_normalized_text`        | Uppercase alphanumeric OCR result        |
| `ocr_confidence`             | OCR confidence score                     |
| `consensus_text`             | Current temporal-voting result           |
| `consensus_state`            | Provisional or confirmed                 |
| `observation_count`          | Number of OCR observations for the track |
| `consensus_score`            | Aggregated support for the selected text |

---

## 8. Known Limitations

### 8.1 Plate resolution

License-plate detection does not guarantee readable OCR.

A detected plate may still contain too few pixels for reliable character recognition. Plate visibility must be evaluated using the original video resolution rather than only the resized model input.

### 8.2 Camera geometry

The system is primarily designed for static or near-static cameras.

Camera movement, vibration, zoom changes, or strong stabilization drift may cause:

* ROI misalignment;
* tile misalignment;
* tracking instability;
* incorrect cached plate-box projection;
* duplicate or missed detections.

### 8.3 Occlusion

Long vehicle occlusions can cause ByteTrack identity fragmentation or ID switches.

The current pipeline does not use appearance-based vehicle ReID.

### 8.4 Plate perspective

The current plate detector produces rectangular bounding boxes.

It does not detect the four physical plate corners and therefore cannot perform accurate projective rectification for strongly skewed plates.

### 8.5 OCR voting

Temporal consensus currently operates on exact normalized strings.

For example, these OCR observations are treated as different candidates:

```text
KL07AB1234
KL07A81234
KL07AB123
```

Character-level alignment and voting are not currently implemented.

### 8.6 Country-specific formatting

The pipeline does not currently include a complete Indian registration-number grammar or state-specific validation system.

OCR output is normalized, but not fully validated against all Indian number-plate formats.

### 8.7 CPU performance

The project targets CPU-based study and prototyping, not real-time processing.

Performance depends on:

* source resolution;
* ROI size;
* tile dimensions;
* tile overlap;
* detector input size;
* frame-skip configuration;
* number of vehicles;
* plate-detection frequency;
* OCR scheduling;
* CPU architecture;
* thread configuration.

---

## 9. Not Implemented Yet

The following capabilities are outside the current `0.3.0` implementation:

* Four-corner license-plate detection.
* Perspective rectification from detected plate corners.
* Character-level temporal OCR consensus.
* Indian-number-plate-specific OCR training.
* Indian registration-format validation.
* Multi-frame plate super-resolution.
* Appearance-based vehicle ReID.
* Automatic lane detection.
* Automatic lane configuration.
* Camera-motion compensation.
* Homography-based world-coordinate projection.
* Vehicle speed estimation.
* Lane-wise traffic counting.
* Wrong-way movement detection.
* Production database integration.
* Distributed processing.
* Production queue management.
* User authentication and access control.
* Privacy-aware retention configuration.
* Plate-image encryption.
* Automatic plate-image deletion.
* Audit logging.
* Enforcement-grade validation.
* Legal-evidence chain of custody.

---

## 10. Planned Development

Potential future development areas include:

### Phase 1 — OCR reliability

* Character-level temporal voting.
* Edit-distance-based OCR grouping.
* Position-aware character confidence.
* Indian plate-format constraints.
* Better OCR preprocessing selection.
* OCR quality scoring.

### Phase 2 — Plate geometry

* Four-corner plate detection.
* Perspective transformation.
* Rectified plate crops.
* Blur and skew quality metrics.
* Multi-frame crop selection.

### Phase 3 — Tracking robustness

* Appearance-based vehicle ReID.
* Improved long-occlusion recovery.
* Class-aware track association.
* Track-fragment merging.
* Camera-motion compensation.

### Phase 4 — Traffic analytics

* Lane configuration.
* Entry and exit lines.
* Direction detection.
* Vehicle counting.
* Time headway.
* Occupancy.
* Spot-speed estimation.
* Homography-based distance measurement.

### Phase 5 — Production engineering

* PostgreSQL integration.
* Object-storage integration.
* Worker queues.
* Job monitoring.
* API services.
* Configurable retention periods.
* Privacy controls.
* Model-version tracking.
* Reproducible processing manifests.

---

## 11. Research and Evaluation Guidance

Evaluation should separate the major pipeline stages.

### Vehicle detection metrics

* Precision.
* Recall.
* F1 score.
* Per-class average precision.
* Small-object recall.
* Duplicate-detection rate.

### Tracking metrics

* IDF1.
* HOTA.
* MOTA.
* ID switches.
* Track fragmentation.
* Track recall.
* Average track duration.

### Plate detection metrics

* Plate detection precision.
* Plate detection recall.
* Plate-to-vehicle association accuracy.
* Plate visibility rate.
* Plate crop pixel dimensions.

### OCR metrics

* Exact plate accuracy.
* Character accuracy.
* Character error rate.
* Confirmed-result precision.
* Confirmation coverage.
* False-confirmation rate.
* Time to confirmation.

### Performance metrics

* Vehicle detector time per frame.
* Plate detector time per scheduled crop.
* OCR time per plate crop.
* End-to-end processing FPS.
* Peak memory use.
* Number of processed tiles.
* Number of saved plate crops.

---

## 12. Recommended Validation Procedure

For a controlled test:

1. Use a static CCTV video with known frame rate and resolution.
2. Draw an ROI limited to the visible road surface.
3. Confirm that all generated tiles remain inside or near the ROI.
4. Inspect cross-tile duplicate merging.
5. Verify vehicle IDs through tile boundaries.
6. Review class stabilization for long tracks.
7. Measure the detected plate dimensions in the original frame.
8. Inspect plate-to-vehicle associations.
9. Compare OCR observations across consecutive frames.
10. Review provisional and confirmed consensus results separately.
11. Manually validate final plate numbers against visible source frames.
12. Record failure cases by blur, skew, lighting, occlusion, and plate size.

Do not evaluate OCR only from the annotated output video because video re-encoding can further reduce plate readability.

---

## 13. Responsible Use

License plates may constitute personal or sensitive information depending on jurisdiction and application.

Users of this repository are responsible for:

* complying with applicable privacy laws;
* defining lawful processing purposes;
* restricting access to plate images and OCR results;
* configuring appropriate retention periods;
* securing exported datasets;
* avoiding unauthorized surveillance;
* obtaining required permissions;
* validating model behavior before deployment.

The repository does not provide legal advice or certify compliance with any regulatory framework.

---

## 14. Disclaimer

This software is experimental.

It is provided for research, education, algorithm evaluation, and controlled prototyping. It is not validated for enforcement, toll collection, legal evidence, automated penalties, security access decisions, or other high-risk operational use.

Vehicle detections, tracking identities, classifications, plate detections, and OCR outputs may be incomplete or incorrect.

All results must be independently reviewed before being used for operational or legal decisions.

---

## 15. Version Status

### Version `0.3.0`

Current focus:

* static ROI-aware tiled vehicle detection;
* full-frame ByteTrack tracking;
* confidence-weighted vehicle-class stabilization;
* tracked-vehicle plate detection;
* PaddleOCR recognition;
* exact-text temporal consensus;
* research-oriented file exports.

Current maturity:

```text
Experimental / Research Prototype
```

Production readiness:

```text
Not production-ready
```
