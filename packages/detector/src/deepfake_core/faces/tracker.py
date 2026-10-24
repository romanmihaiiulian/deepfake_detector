"""Persistent face tracking from geometry alone. There is no identity recognition.

Each track predicts its box with constant velocity. Detections are matched to tracks by
Hungarian assignment on 1 - IoU, gated by IoU or centre distance and by scale change. A
track survives `max_gap_s` without detections, which covers brief occlusions and
detector misses. Tracks shorter than `min_detections` are dropped at the end.

This is our own implementation. SORT and DeepSORT are GPL and are not used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from deepfake_core.config import TrackingConfig
from deepfake_core.faces.types import Box, Detection, box_center, box_size, iou

_BLOCKED = 1e6
_MAX_EXTRAPOLATION_S = 0.5  # beyond this, constant velocity is a worse guess than standing still


@dataclass(slots=True)
class _Track:
    track_id: int
    box: Box
    last_time: float
    first_time: float
    velocity: Box = field(default_factory=lambda: np.zeros(4, np.float32))
    hits: int = 1

    def predict(self, time: float) -> Box:
        dt = min(float(time - self.last_time), _MAX_EXTRAPOLATION_S)
        out: Box = self.box + self.velocity * dt
        return out


@dataclass(frozen=True, slots=True)
class TrackSpan:
    track_id: int
    first_time: float
    last_time: float
    detections: int


class FaceTracker:
    """Online tracker: call `update()` once per sampled frame, in time order."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._active: list[_Track] = []
        self._all: dict[int, _Track] = {}
        self._next_id = 0

    def _gate_cost(self, track: _Track, predicted: Box, detection: Detection) -> float:
        overlap = iou(predicted, detection.box)
        size = box_size(predicted)
        distance = float(np.linalg.norm(box_center(predicted) - box_center(detection.box)))
        ratio = detection.area / max(1e-6, box_size(predicted) ** 2)
        scale_ok = 1 / self.config.max_scale_change <= ratio <= self.config.max_scale_change
        near = overlap >= self.config.min_iou or distance <= self.config.max_center_distance * size
        if not (scale_ok and near):
            return _BLOCKED
        # Prefer overlap; fall back to normalised distance when boxes don't overlap.
        return (1.0 - overlap) + (0.0 if overlap > 0 else distance / max(size, 1e-6))

    def update(self, time: float, detections: list[Detection]) -> list[int]:
        """Assign each detection to a track; returns track IDs in detection order."""
        self._active = [t for t in self._active if time - t.last_time <= self.config.max_gap_s]
        ids = [-1] * len(detections)
        if self._active and detections:
            predicted = [track.predict(time) for track in self._active]
            cost = np.array(
                [
                    [self._gate_cost(track, box, det) for det in detections]
                    for track, box in zip(self._active, predicted, strict=True)
                ]
            )
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols, strict=True):
                if cost[row, col] >= _BLOCKED:
                    continue
                self._extend(self._active[row], time, detections[col])
                ids[col] = self._active[row].track_id
        for index, detection in enumerate(detections):
            if ids[index] == -1:
                track = _Track(self._next_id, detection.box.copy(), time, time)
                self._next_id += 1
                self._active.append(track)
                self._all[track.track_id] = track
                ids[index] = track.track_id
        return ids

    def _extend(self, track: _Track, time: float, detection: Detection) -> None:
        dt = time - track.last_time
        if dt > 0:
            observed = (detection.box - track.box) / dt
            alpha = self.config.velocity_smoothing
            track.velocity = (alpha * track.velocity + (1 - alpha) * observed).astype(np.float32)
        track.box = detection.box.copy()
        track.last_time = time
        track.hits += 1

    def _kept(self) -> list[_Track]:
        return sorted(
            (t for t in self._all.values() if t.hits >= self.config.min_detections),
            key=lambda t: (t.first_time, t.track_id),
        )

    def finalize(self) -> dict[int, TrackSpan]:
        """Tracks long enough to keep, keyed by their final ID (0, 1, 2… in order of first
        appearance). `id_map()` translates the raw IDs returned by `update()`."""
        return {
            new_id: TrackSpan(new_id, t.first_time, t.last_time, t.hits)
            for new_id, t in enumerate(self._kept())
        }

    def id_map(self) -> dict[int, int]:
        """Raw track ID (from `update()`) → final ID, for tracks that survive `finalize()`."""
        return {t.track_id: new_id for new_id, t in enumerate(self._kept())}
