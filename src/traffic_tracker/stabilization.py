from __future__ import annotations

from collections import defaultdict, deque


class TrackClassStabilizer:
    def __init__(
        self,
        history_size: int = 20,
        minimum_observations: int = 4,
        switch_ratio: float = 1.75,
        stale_after_frames: int = 300,
    ) -> None:
        self.history_size = history_size
        self.minimum_observations = minimum_observations
        self.switch_ratio = switch_ratio
        self.stale_after_frames = stale_after_frames
        self.histories: dict[int, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.stable_classes: dict[int, int] = {}
        self.last_seen: dict[int, int] = {}

    def update(self, track_id: int, class_id: int, confidence: float, frame_index: int) -> int:
        self.histories[track_id].append((class_id, max(float(confidence), 0.001)))
        self.last_seen[track_id] = frame_index
        scores: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for observed_class, observed_confidence in self.histories[track_id]:
            scores[observed_class] += observed_confidence**2
            counts[observed_class] += 1
        candidate = max(scores, key=scores.get)
        current = self.stable_classes.get(track_id)
        if current is None:
            if len(self.histories[track_id]) >= self.minimum_observations:
                self.stable_classes[track_id] = candidate
            return candidate
        if candidate != current:
            if counts[candidate] >= self.minimum_observations and scores[candidate] >= scores.get(current, 0.0) * self.switch_ratio:
                self.stable_classes[track_id] = candidate
        return self.stable_classes[track_id]

    def get(self, track_id: int, fallback: int) -> int:
        return self.stable_classes.get(track_id, fallback)

    def prune(self, current_frame: int) -> None:
        stale = [track_id for track_id, last in self.last_seen.items() if current_frame - last > self.stale_after_frames]
        for track_id in stale:
            self.histories.pop(track_id, None)
            self.stable_classes.pop(track_id, None)
            self.last_seen.pop(track_id, None)
