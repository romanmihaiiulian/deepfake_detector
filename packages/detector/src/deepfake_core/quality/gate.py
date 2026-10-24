"""Quality tiers and grading.

A metric maps to a tier through its band. The tier of a set of metrics is the worst
metric tier, and each metric below GOOD contributes a human-readable reason.

The gate never touches the manipulation score. `scoring/verdict.py` uses tiers for only
three things: excluding INSUFFICIENT frames, widening verdict bands, and capping
confidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum

from deepfake_core.config import Band, RangeBand


class Tier(IntEnum):
    """Ordered so that a higher value is better quality."""

    INSUFFICIENT = 0
    POOR = 1
    FAIR = 2
    GOOD = 3

    @property
    def label(self) -> str:
        return self.name.lower()


def band_tier(band: Band | RangeBand, value: float) -> Tier:
    if isinstance(band, RangeBand):
        for tier, (low, high) in ((Tier.GOOD, band.good), (Tier.FAIR, band.fair)):
            if low <= value <= high:
                return tier
        low, high = band.poor
        return Tier.POOR if low <= value <= high else Tier.INSUFFICIENT
    sign = 1.0 if band.higher_is_better else -1.0
    for tier, bound in ((Tier.GOOD, band.good), (Tier.FAIR, band.fair), (Tier.POOR, band.poor)):
        if sign * value >= sign * bound:
            return tier
    return Tier.INSUFFICIENT


# metric -> (reason code, message template); templates receive `value`
REASONS: dict[str, tuple[str, str]] = {
    "resolution_short_side": ("low_resolution", "Low video resolution ({value:.0f} px)."),
    "effective_fps": ("low_frame_rate", "Low effective frame rate ({value:.1f} fps)."),
    "bits_per_pixel": ("high_compression", "Strong compression ({value:.3f} bits per pixel)."),
    "frame_availability": (
        "missing_frames",
        "Frames are missing or could not be decoded ({value:.0%} available).",
    ),
    "face_crop_size": ("small_face", "Faces are small in the frame ({value:.0f} px)."),
    "face_sharpness": ("blurry_face", "Faces are blurry."),
    "face_exposure": ("poor_lighting", "Faces are too dark or too bright."),
    "face_clipped_fraction": ("clipped_exposure", "Parts of the face are over- or under-exposed."),
    "face_contrast": ("low_contrast", "Low contrast on the face."),
    "face_detector_score": (
        "uncertain_detection",
        "Face detection is uncertain (occlusion or unusual pose).",
    ),
    "face_yaw": ("turned_face", "Faces are often turned away from the camera."),
    "face_alignment_residual": (
        "inconsistent_landmarks",
        "Facial landmarks are inconsistent (occlusion or extreme pose).",
    ),
    "no_face": ("no_face", "No face was visible long enough to analyze."),
}


@dataclass(frozen=True, slots=True)
class QualityReason:
    code: str
    message: str
    tier: Tier
    metric: str
    value: float | None


@dataclass(frozen=True, slots=True)
class Grade:
    tier: Tier
    reasons: tuple[QualityReason, ...] = ()
    metrics: Mapping[str, float | None] = field(default_factory=dict)


def reason_for(metric: str, value: float | None, tier: Tier) -> QualityReason:
    code, template = REASONS.get(metric, (metric, f"{metric} is low."))
    message = template.format(value=value) if value is not None else template
    return QualityReason(code, message, tier, metric, value)


def grade(metrics: Mapping[str, float | None], bands: Mapping[str, Band | RangeBand]) -> Grade:
    """Grade metrics against their bands. Missing (None) metrics are skipped."""
    tier = Tier.GOOD
    reasons: list[QualityReason] = []
    for name, value in metrics.items():
        if value is None or name not in bands:
            continue
        metric_tier = band_tier(bands[name], value)
        if metric_tier < Tier.GOOD:
            reasons.append(reason_for(name, value, metric_tier))
        tier = min(tier, metric_tier)
    reasons.sort(key=lambda r: (r.tier, r.metric))
    return Grade(tier, tuple(reasons), dict(metrics))


def median_tier(tiers: Iterable[Tier]) -> Tier:
    """The lower median: for an even count the worse middle value (conservative)."""
    ordered = sorted(tiers)
    if not ordered:
        return Tier.INSUFFICIENT
    return ordered[(len(ordered) - 1) // 2]


def combine(*grades: Grade) -> Grade:
    """The worst tier, with reasons from every input and metrics merged."""
    if not grades:
        return Grade(Tier.INSUFFICIENT, (reason_for("no_face", None, Tier.INSUFFICIENT),))
    tier = min(g.tier for g in grades)
    seen: set[str] = set()
    reasons: list[QualityReason] = []
    for reason in sorted((r for g in grades for r in g.reasons), key=lambda r: (r.tier, r.metric)):
        if reason.code not in seen:
            seen.add(reason.code)
            reasons.append(reason)
    metrics: dict[str, float | None] = {}
    for g in grades:
        metrics.update(g.metrics)
    return Grade(tier, tuple(reasons), metrics)
