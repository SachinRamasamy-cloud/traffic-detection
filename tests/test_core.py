from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from traffic_tracker.nms import merge_detections
from traffic_tracker.roi import load_roi
from traffic_tracker.stabilization import TrackClassStabilizer
from traffic_tracker.tiling import build_sparse_tiles
from traffic_tracker.tracking import DetectionBatch
from traffic_tracker.types import Detection


class CoreTests(unittest.TestCase):
    def test_detection_batch_xywh(self):
        batch = DetectionBatch(np.asarray([[10, 20, 30, 60]], dtype=np.float32), np.asarray([0.9]), np.asarray([2]))
        np.testing.assert_allclose(batch.xywh[0], [20, 40, 20, 40])

    def test_cross_tile_nms(self):
        detections = [
            Detection(np.asarray([10, 10, 100, 100], dtype=np.float32), 0.9, 2, 0),
            Detection(np.asarray([12, 12, 102, 102], dtype=np.float32), 0.8, 7, 1),
        ]
        merged = merge_detections(detections, 0.5, class_agnostic=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].tile_index, 0)

    def test_roi_and_sparse_tiles(self):
        payload = {
            "coordinate_space": "normalized",
            "detection_geometry": {
                "type": "Polygon",
                "points": [[0.1, 0.4], [0.9, 0.4], [0.9, 0.9], [0.1, 0.9]],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roi.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            roi = load_roi(path, 1000, 500)
            self.assertTrue(roi.contains_point(500, 300))
            self.assertFalse(roi.contains_point(500, 50))
            tiles = build_sparse_tiles(1000, 500, roi, 320, 0.2, 0.001)
            self.assertGreater(len(tiles), 0)
            self.assertTrue(all(tile.roi_ratio > 0 for tile in tiles))

    def test_class_stabilizer_rejects_one_frame_flip(self):
        stabilizer = TrackClassStabilizer(history_size=10, minimum_observations=3, switch_ratio=1.5)
        values = [stabilizer.update(1, 2, 0.9, i) for i in range(4)]
        self.assertEqual(values[-1], 2)
        changed = stabilizer.update(1, 7, 0.3, 5)
        self.assertEqual(changed, 2)


if __name__ == "__main__":
    unittest.main()
