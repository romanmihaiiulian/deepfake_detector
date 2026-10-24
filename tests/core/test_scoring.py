from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from deepfake_core.config import AggregationConfig, VerdictConfig
from deepfake_core.quality import Grade, Tier
from deepfake_core.scoring import (
    CandidatePool,
    FrameScore,
    Interval,
    decide,
    score_track,
    select_evidence,
    suspicious_intervals,
    video_probability,
)
from deepfake_core.scoring.aggregate import window_max_mean
from deepfake_core.scoring.intervals import median_filter

AGG = AggregationConfig()
VERDICT = VerdictConfig()
GOOD = Grade(Tier.GOOD)


def frames(
    scores: list[float | None], rate: float = 4.0, tier: Tier = Tier.GOOD
) -> list[FrameScore]:
    return [FrameScore(i / rate, p, tier) for i, p in enumerate(scores)]


# --------------------------------------------------------------------------- aggregation


def test_track_score_statistics() -> None:
    track = score_track(0, frames([0.1, 0.2, 0.3, 0.4, None, 0.5, 0.6, 0.7, 0.8]), GOOD, AGG)
    assert track.scored_frames == 8 and track.detections == 9
    assert track.mean == pytest.approx(0.45)
    assert track.median == pytest.approx(0.45)
    assert track.eligible


def test_eligibility_needs_enough_frames_and_quality() -> None:
    few = score_track(0, frames([0.9] * 5), GOOD, AGG)
    assert not few.eligible
    insufficient = score_track(1, frames([0.9] * 20), Grade(Tier.INSUFFICIENT), AGG)
    assert not insufficient.eligible


def test_window_max_catches_a_partial_manipulation() -> None:
    scores = [0.1] * 40 + [0.95] * 16 + [0.1] * 40  # 4 s burst inside 24 s
    track = score_track(0, frames(scores), GOOD, AGG)
    assert track.mean is not None and track.mean < 0.3
    assert track.window_max == pytest.approx(0.95)
    assert window_max_mean([0.0, 1.0], [0.9, 0.9], 4.0) is None  # fewer than 3 frames


def test_video_probability_is_max_over_eligible_tracks() -> None:
    low = score_track(0, frames([0.2] * 10), GOOD, AGG)
    high = score_track(1, frames([0.8] * 10), GOOD, AGG)
    ineligible = score_track(2, frames([0.99] * 3), GOOD, AGG)
    probability, decisive = video_probability([low, high, ineligible])
    assert probability == pytest.approx(0.8)
    assert decisive is not None and decisive.track_id == 1
    assert video_probability([ineligible]) == (None, None)


# --------------------------------------------------------------------------- intervals


def test_median_filter_ignores_single_spikes() -> None:
    assert median_filter([0.1, 0.9, 0.1, 0.1], 3) == [0.5, 0.1, 0.1, 0.1]


def test_suspicious_interval_boundaries() -> None:
    scores = [0.1] * 8 + [0.9] * 8 + [0.1] * 8  # 2.0 s .. 4.0 s
    points = [(i / 4, p) for i, p in enumerate(scores)]
    intervals = suspicious_intervals(3, points, 0.25, AGG)
    assert len(intervals) == 1
    interval = intervals[0]
    assert (interval.track_id, interval.start, interval.end) == (3, 2.0, 4.0)
    assert interval.score == pytest.approx(0.9)


def test_short_runs_are_dropped_and_close_runs_merged() -> None:
    short = [0.1] * 4 + [0.9] * 2 + [0.1] * 4  # 0.5 s run
    assert suspicious_intervals(0, [(i / 4, p) for i, p in enumerate(short)], 0.25, AGG) == []
    gapped = [0.9] * 6 + [0.1] * 3 + [0.9] * 6  # 0.75 s gap < 1 s merge gap
    merged = suspicious_intervals(0, [(i / 4, p) for i, p in enumerate(gapped)], 0.25, AGG)
    assert len(merged) == 1 and merged[0].end == pytest.approx(3.75)


# --------------------------------------------------------------------------- verdict


@pytest.mark.parametrize(
    ("probability", "tier", "verdict"),
    [
        (0.95, Tier.GOOD, "manipulated_likely"),
        (0.85, Tier.FAIR, "manipulated_likely"),
        (0.85, Tier.POOR, "inconclusive"),
        (0.95, Tier.POOR, "manipulated_likely"),
        (0.95, Tier.INSUFFICIENT, "inconclusive"),
        (0.50, Tier.GOOD, "inconclusive"),
        (0.10, Tier.GOOD, "authentic_likely"),
        (0.20, Tier.POOR, "inconclusive"),
        (None, Tier.GOOD, "inconclusive"),
    ],
)
def test_verdict_bands(probability: float | None, tier: Tier, verdict: str) -> None:
    decision = decide(probability, tier, scored_frames=40, iqr=0.05, config=VERDICT)
    assert decision.verdict == verdict


def test_confidence_rules() -> None:
    strong = decide(0.99, Tier.GOOD, scored_frames=40, iqr=0.05, config=VERDICT)
    assert strong.confidence == "high"
    few_frames = decide(0.99, Tier.GOOD, scored_frames=10, iqr=0.05, config=VERDICT)
    assert few_frames.confidence == "low"
    spread = decide(0.99, Tier.GOOD, scored_frames=40, iqr=0.4, config=VERDICT)
    assert spread.confidence == "medium"
    fair = decide(0.99, Tier.FAIR, scored_frames=40, iqr=0.05, config=VERDICT)
    assert fair.confidence == "medium"
    poor = decide(0.99, Tier.POOR, scored_frames=40, iqr=0.05, config=VERDICT)
    assert poor.confidence == "low"
    assert decide(0.5, Tier.GOOD, scored_frames=40, iqr=0.0, config=VERDICT).confidence == "low"


def test_poor_quality_explains_why_it_held_back() -> None:
    decision = decide(0.85, Tier.POOR, scored_frames=40, iqr=0.05, config=VERDICT)
    assert decision.reasons == ("poor_quality_needs_stronger_evidence",)


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@given(
    st.floats(0, 1, allow_nan=False),
    st.sampled_from(list(Tier)),
    st.sampled_from(list(Tier)),
    st.integers(0, 200),
    st.one_of(st.none(), st.floats(0, 1, allow_nan=False)),
)
def test_worse_quality_never_makes_a_decision_bolder(
    probability: float, tier_a: Tier, tier_b: Tier, frames_scored: int, iqr: float | None
) -> None:
    better, worse = max(tier_a, tier_b), min(tier_a, tier_b)
    at_better = decide(probability, better, scored_frames=frames_scored, iqr=iqr, config=VERDICT)
    at_worse = decide(probability, worse, scored_frames=frames_scored, iqr=iqr, config=VERDICT)
    if at_worse.verdict != "inconclusive":
        assert at_better.verdict == at_worse.verdict
    assert CONFIDENCE_RANK[at_worse.confidence] <= CONFIDENCE_RANK[at_better.confidence]


@given(
    st.lists(st.floats(0, 1, allow_nan=False), min_size=8, max_size=40), st.sampled_from(list(Tier))
)
def test_quality_never_changes_the_probability(scores: list[float], tier: Tier) -> None:
    """The score is computed from frame probabilities only; quality affects eligibility."""
    good = score_track(0, frames(scores, tier=Tier.GOOD), Grade(Tier.GOOD), AGG)
    other = score_track(0, frames(scores, tier=tier), Grade(max(tier, Tier.POOR)), AGG)
    assert good.mean == other.mean


# --------------------------------------------------------------------------- evidence


def crop(value: int) -> np.ndarray:
    return np.full((32, 32, 3), value, np.uint8)


def test_candidate_pool_keeps_the_top_scores() -> None:
    pool = CandidatePool(3)
    for i, p in enumerate([0.1, 0.9, 0.5, 0.7, 0.3]):
        pool.offer(p, 0, float(i), crop(i))
    assert [c.p_fake for c in pool.candidates()] == [0.9, 0.7, 0.5]


def test_evidence_prefers_intervals_and_spreads_picks() -> None:
    pool = CandidatePool(16)
    for i, p in enumerate([0.99, 0.98, 0.97, 0.6, 0.65, 0.7]):
        pool.offer(p, 0, i * 0.25 if i < 3 else 10.0 + i, crop(i))
    interval = Interval(0, 12.5, 16.5, 0.65, 0.7)
    picks = select_evidence(pool.candidates(), [interval], count=4, min_separation_s=1.0)
    assert [round(p.time, 2) for p in picks[:3]] == [15.0, 14.0, 13.0]  # inside the interval
    assert len(picks) == 4  # then the best outside, one per second
    assert picks[0].jpeg[:2] == b"\xff\xd8"
