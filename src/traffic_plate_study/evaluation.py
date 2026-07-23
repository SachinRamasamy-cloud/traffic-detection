from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traffic_plate_study.alpr import normalize_plate_text
from traffic_plate_study.consensus import normalized_similarity


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def evaluate_predictions(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    target_accuracy: float = 0.80,
) -> dict[str, Any]:
    predictions = load_json(predictions_path)
    ground_truth = load_json(ground_truth_path)

    predicted_by_track = {
        int(vehicle["track_id"]): vehicle.get("plate", {})
        for vehicle in predictions.get("vehicles", [])
    }
    truth_by_track = {
        int(item["track_id"]): normalize_plate_text(str(item["plate_text"]))
        for item in ground_truth.get("tracks", [])
    }

    if not truth_by_track:
        raise ValueError("Ground truth contains no tracks")

    exact_correct = 0
    accepted = 0
    false_accepted = 0
    similarities: list[float] = []
    rows: list[dict[str, Any]] = []

    for track_id, expected in sorted(truth_by_track.items()):
        plate_result = predicted_by_track.get(track_id, {})
        predicted = normalize_plate_text(plate_result.get("text"))
        is_accepted = bool(plate_result.get("accepted", False))
        is_exact = bool(predicted and expected and predicted == expected)
        similarity = (
            normalized_similarity(predicted, expected)
            if predicted is not None and expected is not None
            else 0.0
        )
        exact_correct += int(is_exact)
        accepted += int(is_accepted)
        false_accepted += int(is_accepted and not is_exact)
        similarities.append(similarity)
        rows.append(
            {
                "track_id": track_id,
                "expected": expected,
                "predicted": predicted,
                "accepted": is_accepted,
                "exact": is_exact,
                "character_similarity": round(similarity, 6),
            }
        )

    total = len(truth_by_track)
    exact_accuracy = exact_correct / total
    coverage = accepted / total
    mean_similarity = sum(similarities) / total

    return {
        "target_exact_accuracy": target_accuracy,
        "target_reached": exact_accuracy >= target_accuracy,
        "ground_truth_tracks": total,
        "exact_correct": exact_correct,
        "exact_plate_accuracy": round(exact_accuracy, 6),
        "accepted_track_coverage": round(coverage, 6),
        "mean_character_similarity": round(mean_similarity, 6),
        "false_accepted_results": false_accepted,
        "tracks": rows,
    }
