from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .nms import merge_detections
from .roi import ROI
from .tiling import prepare_tile_image
from .types import Detection, Tile


class YoloTileDetector:
    def __init__(
        self,
        model_name: str,
        device: str,
        imgsz: int,
        confidence: float,
        detector_iou: float,
        merge_iou: float,
        tile_batch_size: int,
        classes_raw: str,
        class_agnostic_merge: bool,
        max_detections: int,
        one_to_many: bool,
        mask_outside_roi: bool,
        mask_dilation: int,
        verbose: bool,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.detector_iou = detector_iou
        self.merge_iou = merge_iou
        self.tile_batch_size = max(1, tile_batch_size)
        self.classes = _resolve_classes(classes_raw, self.model.names)
        self.class_agnostic_merge = class_agnostic_merge
        self.max_detections = max_detections
        self.one_to_many = one_to_many
        self.mask_outside_roi = mask_outside_roi
        self.mask_dilation = mask_dilation
        self.verbose = verbose

    @property
    def names(self):
        return self.model.names

    def detect(self, frame: np.ndarray, tiles: Sequence[Tile], roi: ROI) -> list[Detection]:
        frame_height, frame_width = frame.shape[:2]
        all_detections: list[Detection] = []

        for offset in range(0, len(tiles), self.tile_batch_size):
            batch_tiles = list(tiles[offset : offset + self.tile_batch_size])
            images = [
                prepare_tile_image(
                    frame,
                    tile,
                    roi_mask=roi.mask,
                    mask_outside_roi=self.mask_outside_roi,
                    mask_dilation=self.mask_dilation,
                )
                for tile in batch_tiles
            ]
            results = self.model.predict(
                source=images,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.detector_iou,
                classes=self.classes,
                verbose=self.verbose,
                end2end=not self.one_to_many,
            )

            for tile, result in zip(batch_tiles, results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
                classes = boxes.cls.detach().cpu().numpy().astype(np.int32)

                for local_box, score, class_id in zip(xyxy, confs, classes):
                    local_center_x = float((local_box[0] + local_box[2]) / 2.0)
                    local_center_y = float((local_box[1] + local_box[3]) / 2.0)
                    if local_center_x >= tile.valid_width or local_center_y >= tile.valid_height:
                        continue
                    global_box = local_box.copy()
                    global_box[[0, 2]] += tile.x
                    global_box[[1, 3]] += tile.y
                    global_box[[0, 2]] = np.clip(global_box[[0, 2]], 0, frame_width - 1)
                    global_box[[1, 3]] = np.clip(global_box[[1, 3]], 0, frame_height - 1)
                    if global_box[2] <= global_box[0] or global_box[3] <= global_box[1]:
                        continue
                    if not roi.accepts_box(global_box):
                        continue
                    all_detections.append(
                        Detection(
                            xyxy=global_box,
                            confidence=float(score),
                            class_id=int(class_id),
                            tile_index=tile.index,
                        )
                    )

        return merge_detections(
            all_detections,
            iou_threshold=self.merge_iou,
            class_agnostic=self.class_agnostic_merge,
            max_detections=self.max_detections,
        )


def _resolve_classes(raw: str, names) -> list[int] | None:
    if not raw.strip():
        return None
    mapping = names if isinstance(names, dict) else dict(enumerate(names))
    reverse = {str(name).lower(): int(index) for index, name in mapping.items()}
    output: list[int] = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token.isdigit():
            value = int(token)
            if value not in mapping:
                raise ValueError(f"Unknown class ID: {value}")
        else:
            value = reverse.get(token.lower(), -1)
            if value < 0:
                raise ValueError(f"Unknown class name: {token}")
        if value not in output:
            output.append(value)
    return output
