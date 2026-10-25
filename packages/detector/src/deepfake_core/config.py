"""Pipeline configuration: every threshold the engine uses, in one versioned place.

Values in the `uncalibrated-v0` profile are starting points. The benchmark (M7/M8)
measures them on paired lab data, and `profile` changes whenever they are retuned.
`config_hash()` goes into every result, so a result can always be traced back to the
exact thresholds that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ValidationConfig(_Section):
    min_duration_s: float = 1.0
    max_duration_s: float = 600.0
    max_dimension_px: int = Field(default=4096, description="Longest coded side accepted")


class SamplingConfig(_Section):
    rate_hz: float = Field(default=4.0, gt=0, description="Target samples per second")
    max_samples: int = Field(
        default=960, gt=0, description="Rate is lowered so a video yields at most this"
    )
    duplicate_max_abs_diff: int = Field(
        default=2,
        ge=0,
        description="Max luma difference (0-255) for consecutive frames to be duplicates",
    )


class DetectionConfig(_Section):
    input_long_side: int = Field(default=640, gt=0, description="Frames are downscaled to this")
    score_threshold: float = Field(default=0.7, ge=0, le=1)
    nms_threshold: float = Field(default=0.3, ge=0, le=1)
    max_faces: int = Field(default=8, gt=0, description="Largest faces kept per frame")


class TrackingConfig(_Section):
    max_gap_s: float = Field(default=2.0, gt=0, description="A track survives this long unseen")
    min_iou: float = Field(default=0.1, ge=0, le=1)
    max_center_distance: float = Field(
        default=0.75, gt=0, description="Gate on centre distance, in predicted box sizes"
    )
    max_scale_change: float = Field(default=2.0, gt=1, description="Gate on box area ratio")
    min_detections: int = Field(default=4, gt=0, description="Shorter tracks are dropped")
    velocity_smoothing: float = Field(default=0.5, ge=0, lt=1)


class AlignmentConfig(_Section):
    scale: float = Field(default=1.3, gt=1, description="GenD margin: crop side / face scale")
    model_input: int = Field(default=224, gt=0)


class Band(_Section):
    """Tier boundaries for a metric: the value needed to reach GOOD, FAIR and POOR.

    With `higher_is_better`, a value >= `good` is GOOD, and so on. Otherwise the
    comparisons flip (value <= `good` is GOOD). Anything past `poor` is INSUFFICIENT.
    """

    good: float
    fair: float
    poor: float
    higher_is_better: bool = True


class RangeBand(_Section):
    """Tier boundaries for a two-sided metric: the closed interval for each tier."""

    good: tuple[float, float]
    fair: tuple[float, float]
    poor: tuple[float, float]


class QualityConfig(_Section):
    """Starting thresholds (profile uncalibrated-v0). M8 tunes them on degraded lab data."""

    # video level
    resolution_short_side: Band = Band(good=720, fair=480, poor=144)
    effective_fps: Band = Band(good=20, fair=10, poor=2)
    # Bits per pixel is a weak proxy: simple or static scenes compress very well even at
    # high quality, so it can only flag starved bitrates. Face sharpness and size measure
    # visible degradation directly.
    bits_per_pixel: Band = Band(good=0.02, fair=0.008, poor=0.0)
    frame_availability: Band = Band(good=0.95, fair=0.8, poor=0.5)
    # face level, measured per detection on the aligned crop
    face_crop_size: Band = Band(good=200, fair=120, poor=64)
    face_sharpness: Band = Band(good=60, fair=25, poor=8)
    face_exposure: RangeBand = RangeBand(good=(60, 200), fair=(40, 220), poor=(20, 240))
    face_clipped_fraction: Band = Band(good=0.05, fair=0.15, poor=0.35, higher_is_better=False)
    face_contrast: Band = Band(good=30, fair=18, poor=8)
    face_detector_score: Band = Band(good=0.9, fair=0.8, poor=0.6)
    face_yaw: Band = Band(good=0.15, fair=0.3, poor=0.5, higher_is_better=False)
    face_alignment_residual: Band = Band(good=0.03, fair=0.06, poor=0.1, higher_is_better=False)


class GenDConfig(_Section):
    batch_size: int = Field(default=16, gt=0)


class AggregationConfig(_Section):
    track_score: Literal["mean", "median", "window_max", "temporal"] = Field(
        default="mean",
        description="GenD protocol: mean of frame probabilities ('temporal': step-14 model)",
    )
    min_scored_frames: int = Field(default=8, gt=0, description="Eligibility: ~2 s at 4 Hz")
    window_s: float = Field(default=4.0, gt=0, description="Window for the window_max statistic")
    interval_threshold: float = Field(default=0.6, ge=0, le=1)
    interval_min_duration_s: float = Field(default=1.0, ge=0)
    interval_merge_gap_s: float = Field(default=1.0, ge=0)
    interval_smoothing: int = Field(default=3, ge=1, description="Median filter width (samples)")
    evidence_count: int = Field(default=4, ge=0)
    evidence_min_separation_s: float = Field(default=1.0, ge=0)


class TemporalConfig(_Section):
    """The step-14 temporal model. Off unless a checkpoint trained with `dfbench temporal
    train` is configured; its score is then reported as `detector_scores.temporal`."""

    checkpoint: str | None = Field(default=None, description="Temporal model (.safetensors)")


class VerdictConfig(_Section):
    """Decision bands on the (uncalibrated) manipulation probability, per quality tier.

    The manipulated threshold is deliberately high: GenD gives scores of 0.8-0.98 to
    some real footage outside its training distribution (e.g. Celeb-DF real frames), and
    the product's first objective is a very low false-positive rate. M7 replaces these
    with values chosen at target false-positive rates on lab data.
    """

    manipulated_threshold: float = Field(default=0.80, ge=0, le=1)
    authentic_threshold: float = Field(default=0.30, ge=0, le=1)
    poor_manipulated_threshold: float = Field(default=0.90, ge=0, le=1)
    poor_authentic_threshold: float = Field(default=0.15, ge=0, le=1)
    high_confidence_margin: float = 0.15
    medium_confidence_margin: float = 0.05
    high_confidence_min_frames: int = 24
    medium_confidence_min_frames: int = 12
    high_confidence_max_iqr: float = 0.25
    medium_confidence_max_iqr: float = 0.45


class PipelineConfig(_Section):
    profile: str = "uncalibrated-v0"
    validation: ValidationConfig = ValidationConfig()
    sampling: SamplingConfig = SamplingConfig()
    quality: QualityConfig = QualityConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    gend: GenDConfig = GenDConfig()
    aggregation: AggregationConfig = AggregationConfig()
    temporal: TemporalConfig = TemporalConfig()
    verdict: VerdictConfig = VerdictConfig()

    @model_validator(mode="after")
    def _temporal_needs_checkpoint(self) -> PipelineConfig:
        if self.aggregation.track_score == "temporal" and not self.temporal.checkpoint:
            raise ValueError("aggregation.track_score 'temporal' needs temporal.checkpoint")
        return self

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        """Load overrides from YAML or JSON; unspecified values keep their defaults."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
