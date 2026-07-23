from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VehicleConfig:
    model: str = "yolo26s.pt"
    tracker_file: str = "bytetrack.yaml"
    image_size: int = 960
    confidence: float = 0.25
    iou: float = 0.60
    device: str | int | None = None
    class_ids: tuple[int, ...] = (2, 3, 5, 7)
    minimum_track_frames: int = 5
    crop_expansion: float = 0.05
    min_crop_width: int = 120
    min_crop_height: int = 80
    min_crop_blur_score: float = 10.0


@dataclass(frozen=True)
class PlateConfig:
    detector_model: str = "yolo-v9-s-608-license-plate-end2end"
    detector_confidence: float = 0.35
    ocr_model: str = "cct-s-v2-global-model"
    ocr_device: str = "auto"
    every_n_frames: int = 2
    max_attempts_per_track: int = 30
    max_observations_per_track: int = 12
    min_text_length: int = 5
    max_text_length: int = 12
    text_pattern: str = r"^[A-Z0-9]{5,12}$"


@dataclass(frozen=True)
class ConsensusConfig:
    minimum_support: int = 3
    minimum_confidence: float = 0.80
    maximum_normalized_edit_distance: float = 0.20


@dataclass(frozen=True)
class RoiConfig:
    polygon: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    save_best_crops: bool = True
    include_all_plate_observations: bool = True
    json_indent: int = 2
    video_codec: str = "mp4v"


@dataclass(frozen=True)
class AppConfig:
    project_name: str = "traffic-plate-study"
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    plate: PlateConfig = field(default_factory=PlateConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level configuration must be a mapping")

    vehicle_data = _section(data, "vehicle")
    tracker_file = str(vehicle_data.get("tracker_file", "bytetrack.yaml"))
    tracker_path = Path(tracker_file)
    if not tracker_path.is_absolute():
        candidate = config_path.parent / tracker_path
        if candidate.exists():
            tracker_file = str(candidate.resolve())

    vehicle = VehicleConfig(
        model=str(vehicle_data.get("model", "yolo26s.pt")),
        tracker_file=tracker_file,
        image_size=int(vehicle_data.get("image_size", 960)),
        confidence=float(vehicle_data.get("confidence", 0.25)),
        iou=float(vehicle_data.get("iou", 0.60)),
        device=vehicle_data.get("device"),
        class_ids=tuple(int(value) for value in vehicle_data.get("class_ids", [2, 3, 5, 7])),
        minimum_track_frames=int(vehicle_data.get("minimum_track_frames", 5)),
        crop_expansion=float(vehicle_data.get("crop_expansion", 0.05)),
        min_crop_width=int(vehicle_data.get("min_crop_width", 120)),
        min_crop_height=int(vehicle_data.get("min_crop_height", 80)),
        min_crop_blur_score=float(vehicle_data.get("min_crop_blur_score", 10.0)),
    )

    plate_data = _section(data, "plate")
    plate = PlateConfig(
        detector_model=str(plate_data.get("detector_model", "yolo-v9-s-608-license-plate-end2end")),
        detector_confidence=float(plate_data.get("detector_confidence", 0.35)),
        ocr_model=str(plate_data.get("ocr_model", "cct-s-v2-global-model")),
        ocr_device=str(plate_data.get("ocr_device", "auto")),
        every_n_frames=max(1, int(plate_data.get("every_n_frames", 2))),
        max_attempts_per_track=max(1, int(plate_data.get("max_attempts_per_track", 30))),
        max_observations_per_track=max(1, int(plate_data.get("max_observations_per_track", 12))),
        min_text_length=max(1, int(plate_data.get("min_text_length", 5))),
        max_text_length=max(1, int(plate_data.get("max_text_length", 12))),
        text_pattern=str(plate_data.get("text_pattern", r"^[A-Z0-9]{5,12}$")),
    )

    consensus_data = _section(data, "consensus")
    consensus = ConsensusConfig(
        minimum_support=max(1, int(consensus_data.get("minimum_support", 3))),
        minimum_confidence=float(consensus_data.get("minimum_confidence", 0.80)),
        maximum_normalized_edit_distance=float(
            consensus_data.get("maximum_normalized_edit_distance", 0.20)
        ),
    )

    roi_data = _section(data, "roi")
    polygon = tuple((int(point[0]), int(point[1])) for point in roi_data.get("polygon", []))
    roi = RoiConfig(polygon=polygon)

    output_data = _section(data, "output")
    output = OutputConfig(
        save_best_crops=bool(output_data.get("save_best_crops", True)),
        include_all_plate_observations=bool(output_data.get("include_all_plate_observations", True)),
        json_indent=int(output_data.get("json_indent", 2)),
        video_codec=str(output_data.get("video_codec", "mp4v")),
    )

    project = _section(data, "project")
    config = AppConfig(
        project_name=str(project.get("name", "traffic-plate-study")),
        vehicle=vehicle,
        plate=plate,
        consensus=consensus,
        roi=roi,
        output=output,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.vehicle.image_size < 320:
        raise ValueError("vehicle.image_size must be at least 320")
    if not 0 < config.vehicle.confidence <= 1:
        raise ValueError("vehicle.confidence must be within (0, 1]")
    if not 0 < config.vehicle.iou <= 1:
        raise ValueError("vehicle.iou must be within (0, 1]")
    if not 0 < config.plate.detector_confidence <= 1:
        raise ValueError("plate.detector_confidence must be within (0, 1]")
    if config.plate.min_text_length > config.plate.max_text_length:
        raise ValueError("plate.min_text_length cannot exceed plate.max_text_length")
    if not 0 <= config.consensus.maximum_normalized_edit_distance <= 1:
        raise ValueError("maximum_normalized_edit_distance must be within [0, 1]")
    if not 0 <= config.consensus.minimum_confidence <= 1:
        raise ValueError("consensus.minimum_confidence must be within [0, 1]")
