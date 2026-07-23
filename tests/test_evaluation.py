import json
from pathlib import Path

from traffic_plate_study.evaluation import evaluate_predictions


def test_evaluation_exact_accuracy(tmp_path: Path) -> None:
    predictions = {
        "vehicles": [
            {"track_id": 1, "plate": {"text": "TN38AB1234", "accepted": True}},
            {"track_id": 2, "plate": {"text": "KL07CD445S", "accepted": True}},
        ]
    }
    truth = {
        "tracks": [
            {"track_id": 1, "plate_text": "TN38AB1234"},
            {"track_id": 2, "plate_text": "KL07CD4455"},
        ]
    }
    predictions_path = tmp_path / "predictions.json"
    truth_path = tmp_path / "truth.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    truth_path.write_text(json.dumps(truth), encoding="utf-8")

    result = evaluate_predictions(predictions_path, truth_path, target_accuracy=0.80)
    assert result["exact_plate_accuracy"] == 0.5
    assert result["target_reached"] is False
    assert result["false_accepted_results"] == 1
