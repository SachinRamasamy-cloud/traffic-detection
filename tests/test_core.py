from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from traffic_tracker.anpr.memory import PlateMemory
from traffic_tracker.anpr.ocr_engine import OCRRead, normalize_plate_text, unpack_recognition_result
from traffic_tracker.anpr.temporal_consensus import PlateTextConsensus
from traffic_tracker.nms import merge_detections
from traffic_tracker.roi import load_roi
from traffic_tracker.stabilization import TrackClassStabilizer
from traffic_tracker.tile_plan import load_or_build_tile_plan, load_tile_plan
from traffic_tracker.tiling import build_sparse_tiles
from traffic_tracker.tracking import DetectionBatch
from traffic_tracker.types import Detection, PlateDetection


class CoreTests(unittest.TestCase):
    def test_detection_batch_xywh(self):
        batch = DetectionBatch(
            np.asarray([[10, 20, 30, 60]], dtype=np.float32),
            np.asarray([0.9]),
            np.asarray([2]),
        )
        np.testing.assert_allclose(batch.xywh[0], [20, 40, 20, 40])

    def test_cross_tile_nms(self):
        detections = [
            Detection(np.asarray([10, 10, 100, 100], dtype=np.float32), 0.9, 2, 0),
            Detection(np.asarray([12, 12, 102, 102], dtype=np.float32), 0.8, 7, 1),
        ]
        merged = merge_detections(detections, 0.5, class_agnostic=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].tile_index, 0)

    def test_roi_and_sparse_tiles_are_roi_anchored(self):
        payload = {
            "coordinate_space": "normalized",
            "detection_geometry": {
                "type": "Polygon",
                "points": [[0.3, 0.4], [0.8, 0.4], [0.8, 0.9], [0.3, 0.9]],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roi.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            roi = load_roi(path, 1000, 500)
            self.assertTrue(roi.contains_point(500, 300))
            self.assertFalse(roi.contains_point(100, 50))
            tiles = build_sparse_tiles(
                1000,
                500,
                roi,
                tile_size=320,
                overlap=0.25,
                minimum_roi_ratio=0.001,
                roi_padding=50,
            )
            self.assertGreater(len(tiles), 0)
            self.assertTrue(all(tile.roi_ratio > 0 for tile in tiles))
            # The ROI starts at x=300 and padding is 50. At least one entry tile
            # must begin before the exact ROI boundary.
            self.assertLessEqual(min(tile.x for tile in tiles), 250)
            # The static grid should not begin at x=0 for this central ROI.
            self.assertGreater(min(tile.x for tile in tiles), 0)

    def test_tile_plan_is_saved_and_reloaded(self):
        payload = {
            "coordinate_space": "source_pixel",
            "reference_frame": {"width": 640, "height": 360},
            "detection_geometry": {
                "type": "Polygon",
                "points": [[100, 120], [540, 120], [600, 340], [40, 340]],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roi_path = root / "roi.json"
            plan_path = root / "tiles.json"
            roi_path.write_text(json.dumps(payload), encoding="utf-8")
            roi = load_roi(roi_path, 640, 360)
            built = load_or_build_tile_plan(
                path=plan_path,
                frame_width=640,
                frame_height=360,
                roi=roi,
                tile_size=256,
                overlap=0.25,
                minimum_roi_ratio=0.001,
                max_tiles=0,
                roi_padding=40,
                force_boundary_tiles=True,
                rebuild=False,
            )
            self.assertTrue(plan_path.is_file())
            loaded = load_tile_plan(plan_path)
            self.assertEqual(built.tiles, loaded.tiles)
            reused = load_or_build_tile_plan(
                path=plan_path,
                frame_width=640,
                frame_height=360,
                roi=roi,
                tile_size=256,
                overlap=0.25,
                minimum_roi_ratio=0.001,
                max_tiles=0,
                roi_padding=40,
                force_boundary_tiles=True,
                rebuild=False,
            )
            self.assertEqual(loaded.tiles, reused.tiles)

    def test_plate_memory_projects_cached_box_with_vehicle_motion(self):
        memory = PlateMemory(cache_frames=5)
        detection = PlateDetection(
            xyxy=np.asarray([140, 160, 220, 190], dtype=np.float32),
            confidence=0.9,
            class_id=0,
            vehicle_track_id=7,
            vehicle_class_id=2,
            vehicle_xyxy=np.asarray([100, 100, 300, 300], dtype=np.float32),
            frame_index=10,
            search_region="lower_0.75",
        )
        memory.update([detection])
        item = memory.get(
            7,
            frame_index=11,
            current_vehicle_xyxy=np.asarray([200, 150, 400, 350], dtype=np.float32),
        )
        self.assertIsNotNone(item)
        np.testing.assert_allclose(item.detection.xyxy, [240, 210, 320, 240], atol=1e-5)
        self.assertEqual(item.age_frames, 1)
        self.assertFalse(item.is_current)

    def test_plate_text_normalization(self):
        self.assertEqual(normalize_plate_text("kl 07-ab 1234"), "KL07AB1234")
        self.assertEqual(normalize_plate_text("  MH.12 XY 9  "), "MH12XY9")

    def test_unpack_paddle_result(self):
        text, score = unpack_recognition_result(
            {"res": {"rec_text": "KL 07 AB 1234", "rec_score": 0.91}}
        )
        self.assertEqual(text, "KL 07 AB 1234")
        self.assertAlmostEqual(score, 0.91)

    def test_plate_text_consensus_confirms_repeated_read(self):
        consensus = PlateTextConsensus(
            minimum_observations=3,
            minimum_confirm_confidence=0.5,
            minimum_dominance=0.6,
        )
        for frame_index in (10, 12, 14):
            result = consensus.update(
                OCRRead(
                    vehicle_track_id=7,
                    frame_index=frame_index,
                    raw_text="KL 07 AB 1234",
                    text="KL07AB1234",
                    confidence=0.8,
                    variant="clahe",
                    accepted=True,
                    plate_confidence=0.7,
                    quality_score=0.8,
                    sharpness=120.0,
                    crop_width=100,
                    crop_height=30,
                    weighted_score=0.448,
                )
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "KL07AB1234")
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(result.observation_count, 3)

    def test_class_stabilizer_rejects_one_frame_flip(self):
        stabilizer = TrackClassStabilizer(
            history_size=10,
            minimum_observations=3,
            switch_ratio=1.5,
        )
        values = [stabilizer.update(1, 2, 0.9, i) for i in range(4)]
        self.assertEqual(values[-1], 2)
        changed = stabilizer.update(1, 7, 0.3, 5)
        self.assertEqual(changed, 2)


if __name__ == "__main__":
    unittest.main()
