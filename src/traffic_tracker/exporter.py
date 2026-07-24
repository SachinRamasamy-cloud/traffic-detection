from __future__ import annotations

import csv
import json
from pathlib import Path


CSV_FIELDS = [
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "state",
    "prediction_gap",
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
]


class ResultExporter:
    def __init__(self, output_dir: Path) -> None:
        self.jsonl_path = output_dir / "tracks.jsonl"
        self.csv_path = output_dir / "tracks.csv"
        self.jsonl_file = None
        self.csv_file = None
        self.csv_writer = None

    def __enter__(self) -> "ResultExporter":
        self.jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.csv_writer.writeheader()
        return self

    def write_frame(self, frame_index: int, timestamp: float, records: list[dict]) -> None:
        self.jsonl_file.write(json.dumps({"frame_index": frame_index, "timestamp_seconds": timestamp, "objects": records}, ensure_ascii=False) + "\n")
        for record in records:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            center_x, center_y = record["center_xy"]
            self.csv_writer.writerow(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "track_id": record["track_id"],
                    "state": record["state"],
                    "prediction_gap": record["prediction_gap"],
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
                }
            )

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.jsonl_file:
            self.jsonl_file.close()
        if self.csv_file:
            self.csv_file.close()
