"""From frame scores to track scores to a video-level manipulation probability.

GenD scores a video as the mean of its frame probabilities, and that is the default track
score. Median, p90 and the maximum `window_s` window mean are kept as well, so the
benchmark can pick the best aggregator with data (a partial manipulation can hide in a
long mean). The video probability is the maximum over eligible tracks, because one
manipulated face is enough.

Quality enters in exactly one place: a track is eligible only if it has enough scored
frames and its quality tier is at least POOR. Quality never scales a score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from deepfake_core.config import AggregationConfig
from deepfake_core.quality.gate import Grade, Tier


@dataclass(frozen=True, slots=True)
class FrameScore:
    time: float
    p_fake: float | None
    """None when the frame was not scored (quality INSUFFICIENT or alignment failed)."""
    tier: Tier


@dataclass(frozen=True, slots=True)
class TrackScore:
    track_id: int
    start: float
    end: float
    detections: int
    scored_frames: int
    quality: Grade
    mean: float | None
    median: float | None
    p90: float | None
    window_max: float | None
    iqr: float | None
    eligible: bool
    timeline: tuple[FrameScore, ...]

    def score(self, method: str) -> float | None:
        return {"mean": self.mean, "median": self.median, "window_max": self.window_max}[method]


def window_max_mean(times: Sequence[float], scores: Sequence[float], window: float) -> float | None:
    """Highest mean score over any window [t, t + window) that starts at a scored frame
    and holds at least 3 scored frames."""
    best: float | None = None
    t = np.asarray(times)
    s = np.asarray(scores)
    for start in t:
        inside = s[(t >= start) & (t < start + window)]
        if inside.size >= 3:
            value = float(inside.mean())
            best = value if best is None else max(best, value)
    return best


def score_track(
    track_id: int,
    frames: Sequence[FrameScore],
    quality: Grade,
    config: AggregationConfig,
) -> TrackScore:
    ordered = tuple(sorted(frames, key=lambda f: f.time))
    scored = [(f.time, f.p_fake) for f in ordered if f.p_fake is not None]
    values = np.array([p for _, p in scored], dtype=np.float64)
    mean: float | None = None
    median: float | None = None
    p90: float | None = None
    iqr: float | None = None
    window: float | None = None
    if values.size:
        mean, median = float(values.mean()), float(np.median(values))
        p90 = float(np.percentile(values, 90))
        q75, q25 = np.percentile(values, [75, 25])
        iqr = float(q75 - q25)
        window = window_max_mean([t for t, _ in scored], values.tolist(), config.window_s)
    eligible = values.size >= config.min_scored_frames and quality.tier >= Tier.POOR
    return TrackScore(
        track_id=track_id,
        start=ordered[0].time if ordered else 0.0,
        end=ordered[-1].time if ordered else 0.0,
        detections=len(ordered),
        scored_frames=int(values.size),
        quality=quality,
        mean=mean,
        median=median,
        p90=p90,
        window_max=window,
        iqr=iqr,
        eligible=eligible,
        timeline=ordered,
    )


def video_probability(
    tracks: Sequence[TrackScore], method: str = "mean"
) -> tuple[float | None, TrackScore | None]:
    """Maximum track score over eligible tracks, and the track that decided it."""
    candidates = [
        (t.score(method), t) for t in tracks if t.eligible and t.score(method) is not None
    ]
    if not candidates:
        return None, None
    score, track = max(candidates, key=lambda item: (item[0], -item[1].track_id))
    return score, track
