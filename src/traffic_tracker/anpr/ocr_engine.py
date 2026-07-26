from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..types import PlateDetection

LOGGER = logging.getLogger("traffic_tracker.anpr.ocr")


@dataclass(frozen=True)
class OCRRead:
    vehicle_track_id: int
    frame_index: int
    raw_text: str
    text: str
    confidence: float
    variant: str
    accepted: bool
    plate_confidence: float
    quality_score: float
    sharpness: float
    crop_width: int
    crop_height: int
    weighted_score: float

    def to_record(self) -> dict[str, Any]:
        return {
            "vehicle_track_id": self.vehicle_track_id,
            "frame_index": self.frame_index,
            "raw_text": self.raw_text,
            "text": self.text,
            "confidence": round(float(self.confidence), 6),
            "variant": self.variant,
            "accepted": self.accepted,
            "plate_confidence": round(float(self.plate_confidence), 6),
            "quality_score": round(float(self.quality_score), 6),
            "sharpness": round(float(self.sharpness), 3),
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "weighted_score": round(float(self.weighted_score), 6),
        }


class PlateOCREngine:
    """Recognition-only PaddleOCR wrapper for already-localized plate crops."""

    def __init__(
        self,
        model_name: str = "en_PP-OCRv5_mobile_rec",
        device: str = "cpu",
        cpu_threads: int = 4,
        enable_hpi: bool = False,
        engine: str | None = None,
        batch_size: int = 4,
        target_height: int = 96,
        maximum_width: int = 640,
        minimum_score: float = 0.20,
        minimum_text_length: int = 4,
        maximum_text_length: int = 14,
        minimum_plate_width: int = 12,
        minimum_plate_height: int = 4,
        minimum_sharpness: float = 0.0,
        interval: int = 1,
        maximum_reads_per_track: int = 12,
        pattern: str = "",
        variants: str = "colour,gray,clahe,sharpened",
    ) -> None:
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "OCR was enabled, but PaddleOCR is not installed. Run ./install_ocr.sh "
                "inside the project virtual environment."
            ) from exc

        init_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "cpu_threads": max(1, cpu_threads),
            "enable_hpi": enable_hpi,
        }
        if engine:
            init_kwargs["engine"] = engine

        self.model = TextRecognition(**init_kwargs)
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.target_height = max(16, target_height)
        self.maximum_width = max(32, maximum_width)
        self.minimum_score = float(minimum_score)
        self.minimum_text_length = max(1, minimum_text_length)
        self.maximum_text_length = max(self.minimum_text_length, maximum_text_length)
        self.minimum_plate_width = max(1, minimum_plate_width)
        self.minimum_plate_height = max(1, minimum_plate_height)
        self.minimum_sharpness = max(0.0, minimum_sharpness)
        self.interval = max(1, interval)
        self.maximum_reads_per_track = max(0, maximum_reads_per_track)
        self.pattern = re.compile(pattern) if pattern else None
        self.variant_names = _parse_variants(variants)
        self.last_attempt_frame: dict[int, int] = {}
        self.attempt_counts: dict[int, int] = defaultdict(int)

    def recognize(
        self,
        crop: np.ndarray,
        plate: PlateDetection,
    ) -> OCRRead | None:
        track_id = int(plate.vehicle_track_id)
        if not self._scheduled(track_id, plate.frame_index):
            return None

        plate_width = float(plate.xyxy[2] - plate.xyxy[0])
        plate_height = float(plate.xyxy[3] - plate.xyxy[1])
        if plate_width < self.minimum_plate_width or plate_height < self.minimum_plate_height:
            return None
        if crop.size == 0:
            return None

        crop_height, crop_width = crop.shape[:2]
        sharpness = calculate_sharpness(crop)
        if sharpness < self.minimum_sharpness:
            return None

        self.last_attempt_frame[track_id] = plate.frame_index
        self.attempt_counts[track_id] += 1

        quality = calculate_quality_score(
            crop_width=crop_width,
            crop_height=crop_height,
            sharpness=sharpness,
            plate_confidence=plate.confidence,
        )
        prepared = create_ocr_variants(
            crop,
            target_height=self.target_height,
            maximum_width=self.maximum_width,
            enabled=self.variant_names,
        )
        if not prepared:
            return None

        try:
            results = list(
                self.model.predict(
                    input=[image for _, image in prepared],
                    batch_size=min(self.batch_size, len(prepared)),
                )
            )
        except Exception as exc:
            LOGGER.warning(
                "OCR inference failed for vehicle track %s at frame %s: %s",
                track_id,
                plate.frame_index,
                exc,
            )
            return None

        candidates: list[tuple[str, str, str, float]] = []
        for (variant_name, _), result in zip(prepared, results):
            raw_text, score = unpack_recognition_result(result)
            normalized = normalize_plate_text(raw_text)
            candidates.append((variant_name, raw_text, normalized, score))

        if not candidates:
            return None

        # Prefer a non-empty normalized result, then the highest OCR score.
        variant, raw_text, text, confidence = max(
            candidates,
            key=lambda item: (bool(item[2]), item[3]),
        )
        accepted = self._accept(text, confidence)
        weighted_score = float(confidence) * float(plate.confidence) * float(quality)
        return OCRRead(
            vehicle_track_id=track_id,
            frame_index=plate.frame_index,
            raw_text=raw_text,
            text=text,
            confidence=float(confidence),
            variant=variant,
            accepted=accepted,
            plate_confidence=float(plate.confidence),
            quality_score=float(quality),
            sharpness=float(sharpness),
            crop_width=int(crop_width),
            crop_height=int(crop_height),
            weighted_score=weighted_score,
        )

    def _scheduled(self, track_id: int, frame_index: int) -> bool:
        if (
            self.maximum_reads_per_track > 0
            and self.attempt_counts[track_id] >= self.maximum_reads_per_track
        ):
            return False
        previous = self.last_attempt_frame.get(track_id)
        return previous is None or frame_index - previous >= self.interval

    def _accept(self, text: str, confidence: float) -> bool:
        if confidence < self.minimum_score:
            return False
        if not self.minimum_text_length <= len(text) <= self.maximum_text_length:
            return False
        if self.pattern is not None and self.pattern.fullmatch(text) is None:
            return False
        return True


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def calculate_sharpness(image: np.ndarray) -> float:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_quality_score(
    crop_width: int,
    crop_height: int,
    sharpness: float,
    plate_confidence: float,
) -> float:
    area_score = min(1.0, (crop_width * crop_height) / 3500.0)
    height_score = min(1.0, crop_height / 32.0)
    sharpness_score = min(1.0, sharpness / 250.0)
    confidence_score = min(1.0, max(0.0, float(plate_confidence)))
    return float(
        0.30 * area_score
        + 0.25 * height_score
        + 0.25 * sharpness_score
        + 0.20 * confidence_score
    )


def create_ocr_variants(
    image: np.ndarray,
    target_height: int = 96,
    maximum_width: int = 640,
    enabled: tuple[str, ...] = ("colour", "gray", "clahe", "sharpened"),
) -> list[tuple[str, np.ndarray]]:
    if image.size == 0:
        return []

    padded = _add_padding(image)
    resized = _resize_for_ocr(padded, target_height, maximum_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharpened = cv2.addWeighted(clahe, 1.7, blurred, -0.7, 0)

    available = {
        "colour": resized,
        "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "clahe": cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        "sharpened": cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
    }
    return [(name, available[name]) for name in enabled if name in available]


def unpack_recognition_result(result: Any) -> tuple[str, float]:
    payload: Any
    if isinstance(result, dict):
        payload = result
    else:
        payload = getattr(result, "json", {})
        if callable(payload):
            payload = payload()

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}

    data = payload.get("res", payload)
    if not isinstance(data, dict):
        data = {}
    return str(data.get("rec_text", "")), float(data.get("rec_score", 0.0) or 0.0)


def _add_padding(image: np.ndarray, ratio: float = 0.12) -> np.ndarray:
    height, width = image.shape[:2]
    pad_x = max(3, int(round(width * ratio)))
    pad_y = max(3, int(round(height * ratio)))
    return cv2.copyMakeBorder(
        image,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )


def _resize_for_ocr(
    image: np.ndarray,
    target_height: int,
    maximum_width: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid plate crop dimensions")
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    if target_width > maximum_width:
        target_width = maximum_width
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )


def _parse_variants(raw: str) -> tuple[str, ...]:
    valid = {"colour", "gray", "clahe", "sharpened"}
    output: list[str] = []
    for token in (part.strip().lower() for part in raw.split(",")):
        if not token:
            continue
        if token not in valid:
            raise ValueError(
                f"Unknown OCR preprocessing variant '{token}'. Valid values: {sorted(valid)}"
            )
        if token not in output:
            output.append(token)
    return tuple(output or ["colour"])
