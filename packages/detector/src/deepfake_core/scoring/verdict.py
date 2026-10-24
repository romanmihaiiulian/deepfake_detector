"""Verdict and confidence from the manipulation probability and the quality tier.

Quality can only make a decision more cautious:

- INSUFFICIENT (or no eligible face): always `inconclusive`.
- POOR: stricter bands, and confidence is low.
- FAIR: normal bands, and confidence is at most medium.

So for a fixed probability, worse quality never produces a decided verdict that better
quality wouldn't, and never raises confidence. The property tests enforce this. The
thresholds are profile `uncalibrated-v0`; M7 recommends values at target false-positive
rates, and step 16 replaces them with calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deepfake_core.config import VerdictConfig
from deepfake_core.quality.gate import Tier

Verdict = Literal["authentic_likely", "inconclusive", "manipulated_likely"]
Confidence = Literal["high", "medium", "low"]

_CONFIDENCE_ORDER: dict[Confidence, int] = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    confidence: Confidence
    reasons: tuple[str, ...]


def bands(tier: Tier, config: VerdictConfig) -> tuple[float, float]:
    """(authentic threshold, manipulated threshold) for a quality tier."""
    if tier <= Tier.POOR:
        return config.poor_authentic_threshold, config.poor_manipulated_threshold
    return config.authentic_threshold, config.manipulated_threshold


def _cap(confidence: Confidence, ceiling: Confidence) -> Confidence:
    return confidence if _CONFIDENCE_ORDER[confidence] <= _CONFIDENCE_ORDER[ceiling] else ceiling


def decide(
    probability: float | None,
    tier: Tier,
    *,
    scored_frames: int,
    iqr: float | None,
    config: VerdictConfig,
) -> Decision:
    if probability is None:
        return Decision("inconclusive", "low", ("no_face",))
    if tier is Tier.INSUFFICIENT:
        return Decision("inconclusive", "low", ("insufficient_quality",))

    authentic_at, manipulated_at = bands(tier, config)
    normal_authentic, normal_manipulated = bands(Tier.GOOD, config)
    if probability >= manipulated_at:
        verdict: Verdict = "manipulated_likely"
        margin = probability - manipulated_at
        reasons: tuple[str, ...] = ("score_above_threshold",)
    elif probability <= authentic_at:
        verdict = "authentic_likely"
        margin = authentic_at - probability
        reasons = ("score_below_threshold",)
    else:
        stricter = probability >= normal_manipulated or probability <= normal_authentic
        reason = "poor_quality_needs_stronger_evidence" if stricter else "score_in_uncertain_band"
        return Decision("inconclusive", "low", (reason,))

    spread = iqr if iqr is not None else 1.0
    if (
        margin >= config.high_confidence_margin
        and scored_frames >= config.high_confidence_min_frames
        and spread <= config.high_confidence_max_iqr
    ):
        confidence: Confidence = "high"
    elif (
        margin >= config.medium_confidence_margin
        and scored_frames >= config.medium_confidence_min_frames
        and spread <= config.medium_confidence_max_iqr
    ):
        confidence = "medium"
    else:
        confidence = "low"

    if tier is Tier.FAIR:
        confidence = _cap(confidence, "medium")
    elif tier is Tier.POOR:
        confidence = "low"
    if tier < Tier.GOOD:
        reasons += ("reduced_video_quality",)
    return Decision(verdict, confidence, reasons)
