from __future__ import annotations

import csv
import json
from pathlib import Path


TRACK_CSV_FIELDS = [
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "state",
    "prediction_gap",
    "inside_exact_roi",
    "raw_class_id",
    "raw_class_name",
    "class_id",
    "class_name",
    "class_stabilized",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height",
    "plate_detected",
    "plate_state",
    "plate_age_frames",
    "plate_confidence",
    "plate_x1",
    "plate_y1",
    "plate_x2",
    "plate_y2",
    "plate_text",
    "plate_text_status",
    "plate_text_confidence",
    "plate_text_observations",
    "plate_text_dominance",
]

PLATE_CSV_FIELDS = [
    "frame_index",
    "timestamp_seconds",
    "vehicle_track_id",
    "vehicle_class_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "vehicle_x1",
    "vehicle_y1",
    "vehicle_x2",
    "vehicle_y2",
    "search_region",
    "crop_path",
    "ocr_attempted",
    "ocr_raw_text",
    "ocr_text",
    "ocr_confidence",
    "ocr_variant",
    "ocr_accepted",
    "ocr_quality_score",
    "ocr_sharpness",
    "plate_text",
    "plate_text_status",
    "plate_text_confidence",
    "plate_text_observations",
    "plate_text_dominance",
]

PLATE_NUMBER_CSV_FIELDS = [
    "vehicle_track_id",
    "plate_text",
    "status",
    "confidence",
    "weighted_score",
    "observation_count",
    "total_accepted_observations",
    "dominance",
    "first_frame",
    "last_frame",
]


class ResultExporter:
    def __init__(self, output_dir: Path) -> None:
        self.jsonl_path = output_dir / "tracks.jsonl"
        self.csv_path = output_dir / "tracks.csv"
        self.plates_jsonl_path = output_dir / "plates.jsonl"
        self.plates_csv_path = output_dir / "plates.csv"
        self.jsonl_file = None
        self.csv_file = None
        self.csv_writer = None
        self.plates_jsonl_file = None
        self.plates_csv_file = None
        self.plates_csv_writer = None

    def __enter__(self) -> "ResultExporter":
        self.jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=TRACK_CSV_FIELDS)
        self.csv_writer.writeheader()

        self.plates_jsonl_file = self.plates_jsonl_path.open("w", encoding="utf-8")
        self.plates_csv_file = self.plates_csv_path.open("w", newline="", encoding="utf-8")
        self.plates_csv_writer = csv.DictWriter(self.plates_csv_file, fieldnames=PLATE_CSV_FIELDS)
        self.plates_csv_writer.writeheader()
        return self

    def write_frame(
        self,
        frame_index: int,
        timestamp: float,
        records: list[dict],
        plate_records: list[dict] | None = None,
    ) -> None:
        plate_records = plate_records or []
        self.jsonl_file.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "objects": records,
                    "plate_detections": plate_records,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        for record in records:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            center_x, center_y = record["center_xy"]
            plate = record.get("plate") or {}
            plate_box = plate.get("bbox_xyxy") or [None, None, None, None]
            self.csv_writer.writerow(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "track_id": record["track_id"],
                    "state": record["state"],
                    "prediction_gap": record["prediction_gap"],
                    "inside_exact_roi": record["inside_exact_roi"],
                    "raw_class_id": record["raw_class_id"],
                    "raw_class_name": record["raw_class_name"],
                    "class_id": record["class_id"],
                    "class_name": record["class_name"],
                    "class_stabilized": record["class_stabilized"],
                    "confidence": record["confidence"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": record["width"],
                    "height": record["height"],
                    "plate_detected": bool(plate),
                    "plate_state": plate.get("state"),
                    "plate_age_frames": plate.get("age_frames"),
                    "plate_confidence": plate.get("confidence"),
                    "plate_x1": plate_box[0],
                    "plate_y1": plate_box[1],
                    "plate_x2": plate_box[2],
                    "plate_y2": plate_box[3],
                    "plate_text": record.get("plate_text"),
                    "plate_text_status": record.get("plate_text_status"),
                    "plate_text_confidence": record.get("plate_text_confidence"),
                    "plate_text_observations": record.get("plate_text_observations"),
                    "plate_text_dominance": record.get("plate_text_dominance"),
                }
            )

        self.plates_jsonl_file.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "plates": plate_records,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for plate in plate_records:
            x1, y1, x2, y2 = plate["bbox_xyxy"]
            vx1, vy1, vx2, vy2 = plate["vehicle_bbox_xyxy"]
            ocr = plate.get("ocr") or {}
            self.plates_csv_writer.writerow(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "vehicle_track_id": plate["vehicle_track_id"],
                    "vehicle_class_id": plate["vehicle_class_id"],
                    "class_id": plate["class_id"],
                    "confidence": plate["confidence"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": plate["width"],
                    "height": plate["height"],
                    "vehicle_x1": vx1,
                    "vehicle_y1": vy1,
                    "vehicle_x2": vx2,
                    "vehicle_y2": vy2,
                    "search_region": plate["search_region"],
                    "crop_path": plate.get("crop_path"),
                    "ocr_attempted": bool(ocr),
                    "ocr_raw_text": ocr.get("raw_text"),
                    "ocr_text": ocr.get("text"),
                    "ocr_confidence": ocr.get("confidence"),
                    "ocr_variant": ocr.get("variant"),
                    "ocr_accepted": ocr.get("accepted"),
                    "ocr_quality_score": ocr.get("quality_score"),
                    "ocr_sharpness": ocr.get("sharpness"),
                    "plate_text": plate.get("plate_text"),
                    "plate_text_status": plate.get("plate_text_status"),
                    "plate_text_confidence": plate.get("plate_text_confidence"),
                    "plate_text_observations": plate.get("plate_text_observations"),
                    "plate_text_dominance": plate.get("plate_text_dominance"),
                }
            )

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in (
            self.jsonl_file,
            self.csv_file,
            self.plates_jsonl_file,
            self.plates_csv_file,
        ):
            if handle:
                handle.close()


def write_plate_numbers(output_dir: Path, records: list[dict]) -> tuple[Path, Path]:
    json_path = output_dir / "plate_numbers.json"
    csv_path = output_dir / "plate_numbers.csv"
    json_path.write_text(
        json.dumps({"plate_numbers": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLATE_NUMBER_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in PLATE_NUMBER_CSV_FIELDS})
    return json_path, csv_path
