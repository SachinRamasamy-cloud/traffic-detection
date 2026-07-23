from traffic_plate_study.config import ConsensusConfig
from traffic_plate_study.consensus import build_consensus, levenshtein_distance
from traffic_plate_study.schemas import PlateObservation


def observation(text: str, confidence: float = 0.95) -> PlateObservation:
    return PlateObservation(
        frame_index=1,
        timestamp_ms=40.0,
        raw_text=text,
        normalized_text=text,
        text_valid=True,
        ocr_confidence=confidence,
        detector_confidence=0.95,
        quality_score=0.90,
        vehicle_bbox=(0, 0, 200, 100),
        plate_bbox=(50, 50, 150, 80),
        plate_width=100,
        plate_height=30,
        blur_score=200.0,
    )


def test_levenshtein_distance() -> None:
    assert levenshtein_distance("TN38AB1234", "TN38AB1234") == 0
    assert levenshtein_distance("TN38AB1234", "TN38A81234") == 1


def test_consensus_accepts_repeated_plate_with_one_ocr_error() -> None:
    config = ConsensusConfig(
        minimum_support=3,
        minimum_confidence=0.70,
        maximum_normalized_edit_distance=0.20,
    )
    result = build_consensus(
        [
            observation("TN38AB1234"),
            observation("TN38AB1234"),
            observation("TN38AB1234"),
            observation("TN38A81234", 0.75),
        ],
        config,
    )
    assert result.text == "TN38AB1234"
    assert result.accepted is True
    assert result.support_count == 4


def test_consensus_rejects_single_observation() -> None:
    config = ConsensusConfig(minimum_support=3, minimum_confidence=0.80)
    result = build_consensus([observation("TN38AB1234")], config)
    assert result.accepted is False
    assert result.rejection_reason == "insufficient_support"
