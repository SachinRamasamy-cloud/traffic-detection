from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import cv2
import numpy as np

from traffic_plate_study.config import PlateConfig
from traffic_plate_study.geometry import clamp_bbox, crop_image, laplacian_blur_score
from traffic_plate_study.schemas import BBox, PlateObservation


@dataclass(frozen=True)
class PlateCandidate:
    raw_text: str | None
    normalized_text: str | None
    text_valid: bool
    ocr_confidence: float
    detector_confidence: float
    quality_score: float
    local_bbox: BBox
    plate_width: int
    plate_height: int
    blur_score: float
    crop: np.ndarray


def normalize_plate_text(text: str | None) -> str | None:
    if not text:
        return None
    normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
    return normalized or None


def _mean_confidence(value: float | list[float] | tuple[float, ...] | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        valid = [float(item) for item in value]
        return float(statistics.fmean(valid)) if valid else 0.0
    return float(value)


def _quality_score(
    detector_confidence: float,
    ocr_confidence: float,
    plate_width: int,
    blur_score: float,
    text_valid: bool,
) -> float:
    width_score = min(1.0, plate_width / 160.0)
    sharpness_score = min(1.0, blur_score / 300.0)
    validity_score = 1.0 if text_valid else 0.25
    return float(
        0.25 * detector_confidence
        + 0.35 * ocr_confidence
        + 0.15 * width_score
        + 0.15 * sharpness_score
        + 0.10 * validity_score
    )


class FastAlprEngine:
    def __init__(self, config: PlateConfig) -> None:
        try:
            from fast_alpr import ALPR
        except ImportError as exc:
            raise RuntimeError(
                "FastALPR is not installed. Install requirements-cpu.txt or requirements-gpu.txt."
            ) from exc

        self.config = config
        self.text_regex = re.compile(config.text_pattern)
        self.model = ALPR(
            detector_model=config.detector_model,
            detector_conf_thresh=config.detector_confidence,
            ocr_model=config.ocr_model,
            ocr_device=config.ocr_device,
        )

    def predict(self, vehicle_crop: np.ndarray) -> list[PlateCandidate]:
        if vehicle_crop.size == 0:
            return []

        candidates: list[PlateCandidate] = []
        results = self.model.predict(vehicle_crop)
        height, width = vehicle_crop.shape[:2]

        for result in results:
            detection = result.detection
            bbox = detection.bounding_box
            local_bbox = clamp_bbox((bbox.x1, bbox.y1, bbox.x2, bbox.y2), width, height)
            plate_crop = crop_image(vehicle_crop, local_bbox)
            if plate_crop.size == 0:
                continue

            raw_text = result.ocr.text if result.ocr is not None else None
            normalized = normalize_plate_text(raw_text)
            valid = bool(
                normalized
                and self.config.min_text_length <= len(normalized) <= self.config.max_text_length
                and self.text_regex.fullmatch(normalized)
            )
            ocr_confidence = _mean_confidence(
                result.ocr.confidence if result.ocr is not None else None
            )
            detector_confidence = float(detection.confidence)
            x1, y1, x2, y2 = local_bbox
            plate_width = x2 - x1
            plate_height = y2 - y1
            blur_score = laplacian_blur_score(plate_crop)
            quality = _quality_score(
                detector_confidence=detector_confidence,
                ocr_confidence=ocr_confidence,
                plate_width=plate_width,
                blur_score=blur_score,
                text_valid=valid,
            )

            candidates.append(
                PlateCandidate(
                    raw_text=raw_text,
                    normalized_text=normalized,
                    text_valid=valid,
                    ocr_confidence=ocr_confidence,
                    detector_confidence=detector_confidence,
                    quality_score=quality,
                    local_bbox=local_bbox,
                    plate_width=plate_width,
                    plate_height=plate_height,
                    blur_score=blur_score,
                    crop=plate_crop,
                )
            )

        return sorted(candidates, key=lambda item: item.quality_score, reverse=True)


def candidate_to_observation(
    candidate: PlateCandidate,
    frame_index: int,
    timestamp_ms: float,
    vehicle_bbox: BBox,
    vehicle_crop_path: str | None = None,
    plate_crop_path: str | None = None,
) -> PlateObservation:
    vehicle_x1, vehicle_y1, _, _ = vehicle_bbox
    px1, py1, px2, py2 = candidate.local_bbox
    global_plate_bbox = (
        vehicle_x1 + px1,
        vehicle_y1 + py1,
        vehicle_x1 + px2,
        vehicle_y1 + py2,
    )
    return PlateObservation(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        raw_text=candidate.raw_text,
        normalized_text=candidate.normalized_text,
        text_valid=candidate.text_valid,
        ocr_confidence=candidate.ocr_confidence,
        detector_confidence=candidate.detector_confidence,
        quality_score=candidate.quality_score,
        vehicle_bbox=vehicle_bbox,
        plate_bbox=global_plate_bbox,
        plate_width=candidate.plate_width,
        plate_height=candidate.plate_height,
        blur_score=candidate.blur_score,
        vehicle_crop_path=vehicle_crop_path,
        plate_crop_path=plate_crop_path,
    )
