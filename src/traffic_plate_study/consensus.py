from __future__ import annotations

from collections import defaultdict

from traffic_plate_study.config import ConsensusConfig
from traffic_plate_study.schemas import PlateConsensus, PlateObservation


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))
    for row, right_char in enumerate(right, start=1):
        current = [row]
        for column, left_char in enumerate(left, start=1):
            insert_cost = current[column - 1] + 1
            delete_cost = previous[column] + 1
            substitution_cost = previous[column - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, substitution_cost))
        previous = current
    return previous[-1]


def normalized_similarity(left: str, right: str) -> float:
    maximum_length = max(len(left), len(right), 1)
    return 1.0 - levenshtein_distance(left, right) / maximum_length


def _observation_weight(observation: PlateObservation) -> float:
    return max(
        0.01,
        observation.quality_score
        * max(0.10, observation.ocr_confidence)
        * max(0.10, observation.detector_confidence),
    )


def build_consensus(
    observations: list[PlateObservation],
    config: ConsensusConfig,
) -> PlateConsensus:
    valid = [
        observation
        for observation in observations
        if observation.text_valid and observation.normalized_text
    ]
    if not valid:
        return PlateConsensus(
            text=None,
            accepted=False,
            confidence=0.0,
            support_count=0,
            observation_count=len(observations),
            exact_support_count=0,
            rejection_reason="no_valid_ocr_observations",
        )

    exact_weights: dict[str, float] = defaultdict(float)
    exact_counts: dict[str, int] = defaultdict(int)
    for observation in valid:
        assert observation.normalized_text is not None
        exact_weights[observation.normalized_text] += _observation_weight(observation)
        exact_counts[observation.normalized_text] += 1

    fuzzy_scores: dict[str, float] = defaultdict(float)
    fuzzy_counts: dict[str, int] = defaultdict(int)
    for candidate in exact_weights:
        for observation in valid:
            assert observation.normalized_text is not None
            distance = levenshtein_distance(candidate, observation.normalized_text)
            allowed = max(
                1,
                round(
                    max(len(candidate), len(observation.normalized_text))
                    * config.maximum_normalized_edit_distance
                ),
            )
            if distance <= allowed:
                similarity = normalized_similarity(candidate, observation.normalized_text)
                fuzzy_scores[candidate] += _observation_weight(observation) * similarity
                fuzzy_counts[candidate] += 1

    ranked = sorted(fuzzy_scores.items(), key=lambda item: item[1], reverse=True)
    best_text, best_score = ranked[0]
    total_score = sum(score for _, score in ranked)
    score_share = best_score / total_score if total_score > 0 else 0.0

    supporting = [
        observation
        for observation in valid
        if observation.normalized_text is not None
        and levenshtein_distance(best_text, observation.normalized_text)
        <= max(
            1,
            round(
                max(len(best_text), len(observation.normalized_text))
                * config.maximum_normalized_edit_distance
            ),
        )
    ]
    weighted_quality = sum(
        observation.quality_score * _observation_weight(observation)
        for observation in supporting
    ) / max(sum(_observation_weight(item) for item in supporting), 1e-9)
    support_ratio = len(supporting) / len(valid)
    confidence = min(
        1.0,
        0.45 * weighted_quality + 0.35 * support_ratio + 0.20 * score_share,
    )

    support_count = fuzzy_counts[best_text]
    exact_support_count = exact_counts.get(best_text, 0)
    accepted = (
        support_count >= config.minimum_support
        and confidence >= config.minimum_confidence
    )
    rejection_reason = None
    if support_count < config.minimum_support:
        rejection_reason = "insufficient_support"
    elif confidence < config.minimum_confidence:
        rejection_reason = "low_consensus_confidence"

    alternatives = tuple((text, score) for text, score in ranked[:5])
    return PlateConsensus(
        text=best_text,
        accepted=accepted,
        confidence=confidence,
        support_count=support_count,
        observation_count=len(observations),
        exact_support_count=exact_support_count,
        alternatives=alternatives,
        rejection_reason=rejection_reason,
    )
