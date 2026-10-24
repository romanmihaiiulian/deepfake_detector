"""Suspicious intervals: stretches of a track where frame scores stay high.

Scores are median-filtered over a few samples, so a single noisy frame is ignored. Runs
above the threshold are measured in time, with each sample covering one sampling
interval. Runs separated by short gaps are merged, and runs shorter than
`min_duration_s` are dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from deepfake_core.config import AggregationConfig


@dataclass(frozen=True, slots=True)
class Interval:
    track_id: int
    start: float
    end: float
    score: float
    """Mean raw frame score inside the interval."""
    peak: float


def median_filter(values: Sequence[float], width: int) -> list[float]:
    """Centred median with a window that shrinks at the edges."""
    if width <= 1:
        return list(values)
    half = width // 2
    array = np.asarray(values, dtype=np.float64)
    return [float(np.median(array[max(0, i - half) : i + half + 1])) for i in range(len(array))]


def suspicious_intervals(
    track_id: int,
    points: Sequence[tuple[float, float]],
    sample_interval: float,
    config: AggregationConfig,
) -> list[Interval]:
    """`points` are (time, p_fake) of scored frames in time order."""
    if not points:
        return []
    times = [t for t, _ in points]
    raw = [p for _, p in points]
    smooth = median_filter(raw, config.interval_smoothing)

    runs: list[list[int]] = []
    for i, value in enumerate(smooth):
        if value < config.interval_threshold:
            continue
        if (
            runs
            and times[i] - (times[runs[-1][-1]] + sample_interval) <= config.interval_merge_gap_s
        ):
            runs[-1].append(i)
        else:
            runs.append([i])

    intervals = []
    for run in runs:
        start, end = times[run[0]], times[run[-1]] + sample_interval
        if end - start < config.interval_min_duration_s:
            continue
        inside = [raw[i] for i in range(run[0], run[-1] + 1)]
        intervals.append(
            Interval(track_id, start, end, float(np.mean(inside)), float(np.max(inside)))
        )
    return intervals
