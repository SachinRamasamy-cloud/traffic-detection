YOLO26m Static ROI-Tile Traffic Tracker with License-Plate Detection and OCR

Version: 0.3.0Primary target: Linux CPU study and prototypingPipeline: YOLO26m vehicle detection + static ROI tiling + ByteTrack + license-plate detection + PaddleOCR temporal consensus

This repository is an experimental traffic-video processing pipeline. It is suitable for research, learning, algorithm evaluation, and controlled prototyping. It is not a certified enforcement, tolling, legal-evidence, or production ANPR system.

1. Purpose

This project processes traffic video and produces:

vehicle detections;

persistent vehicle track IDs;

stabilized vehicle classes;

license-plate bounding boxes associated with vehicle track IDs;

saved plate crops;

OCR observations for detected plates;

provisional or confirmed plate numbers based on temporal voting;

annotated video, CSV, JSONL, and summary files.

The pipeline is designed mainly for:

static CCTV cameras;

fixed or near-fixed traffic scenes;

CPU-only development;

testing ROI-aware tiling;

studying vehicle tracking and ANPR;

evaluating plate visibility before building a production system.

It is not designed to guarantee an accurate registration number from every frame. OCR reliability depends heavily on the original plate pixel size, blur, camera angle, compression, lighting, and model generalization.

2. Current capabilities

Implemented

Ultralytics yolo26m.pt vehicle detection.

Polygon and freehand ROI drawing.

Static tile coordinates generated once and reused for every frame.

ROI-only tiling instead of blindly tiling the complete image.

Tile overlap and ROI boundary protection.

Global-coordinate restoration after tile inference.

Cross-tile duplicate merging.

One ByteTrack instance for the complete frame.

ByteTrack Kalman motion prediction.

Low-confidence ByteTrack recovery.

Optional short Kalman-only prediction export.

Confidence-weighted vehicle-class stabilization.

Second-stage license-plate detection inside tracked vehicle crops.

Plate-to-vehicle association using vehicle_track_id.

Cached plate-box projection between scheduled plate detections.

Plate crop saving.

Recognition-only PaddleOCR integration.

Multiple OCR preprocessing variants.

OCR normalization to uppercase letters and digits.

Confidence-weighted exact-text temporal consensus.

Provisional and confirmed plate-number states.

Annotated MP4 output.

Vehicle CSV and JSONL output.

Plate CSV and JSONL output.

Final plate-number JSON and CSV output.

Not implemented yet

Four-corner plate detection.

Perspective rectification using detected plate corners.

Character-level temporal consensus.

Indian-number-plate-specific OCR training.

Multi-frame super-resolution.

Appearance-based vehicle ReID.

Automatic lane configuration.

Production database integration.

Privacy retention controls.

Enforcement-grade validation.
