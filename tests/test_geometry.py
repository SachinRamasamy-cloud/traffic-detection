import numpy as np

from traffic_plate_study.geometry import clamp_bbox, expand_bbox, laplacian_blur_score


def test_clamp_bbox() -> None:
    assert clamp_bbox((-10, -5, 110, 120), 100, 100) == (0, 0, 100, 100)


def test_expand_bbox_stays_inside_frame() -> None:
    assert expand_bbox((10, 10, 90, 90), 0.25, 100, 100) == (0, 0, 100, 100)


def test_blank_image_has_zero_blur_score() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert laplacian_blur_score(image) == 0.0
