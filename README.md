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
