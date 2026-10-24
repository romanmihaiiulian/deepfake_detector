"""The public result contract (`AnalysisResult`), shared by the CLI, the API and the web UI.

The handoff doc's fields come first: analysis_id, verdict, manipulation_probability,
confidence, video_quality, face_tracks, detector_scores and suspicious_intervals. The
rest adds the evidence and metadata a reviewer needs.

Breaking changes bump `schema_version`. The result never contains the uploaded file's
name, face embeddings or full frames.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Final = "1.0"

DISCLAIMER = (
    "This is a probabilistic assessment produced by automated models, not proof of "
    "manipulation or of authenticity. It can be wrong, especially for low-quality or heavily "
    "compressed video. Do not use it as the sole basis for any decision about a person; "
    "when in doubt, verify identity through other means."
)

Verdict = Literal["authentic_likely", "inconclusive", "manipulated_likely"]
Confidence = Literal["high", "medium", "low"]
QualityLevel = Literal["good", "fair", "poor", "insufficient"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SuspiciousInterval(_Model):
    start: float = Field(description="Seconds from the first frame")
    end: float
    score: float = Field(description="Mean frame score inside the interval")
    peak: float
    track_id: int


class QualityReason(_Model):
    code: str
    message: str
    level: QualityLevel
    metric: str


class QualityDetails(_Model):
    level: QualityLevel
    reasons: list[QualityReason]
    metrics: dict[str, float | None]


class TimelinePoint(_Model):
    t: float
    score: float | None = Field(description="None when the frame was too poor to score")
    quality: QualityLevel


class TrackResult(_Model):
    track_id: int
    start: float
    end: float
    detections: int
    scored_frames: int
    score: float | None
    eligible: bool = Field(description="Enough scorable frames to count towards the verdict")
    quality: QualityLevel
    quality_reasons: list[QualityReason]
    timeline: list[TimelinePoint]


class EvidenceFrame(_Model):
    evidence_id: str
    track_id: int
    time: float
    score: float
    kind: Literal["face_crop"] = "face_crop"


class VideoMetadata(_Model):
    container: str
    codec: str
    width: int
    height: int
    rotation: int
    duration: float | None
    fps_nominal: float | None
    fps_effective: float | None
    bit_rate: int | None
    has_audio: bool


class ModelInfo(_Model):
    name: str
    role: str
    version: str
    revision: str
    sha256: str
    license: str


class ProcessingInfo(_Model):
    pipeline_version: str
    config_profile: str
    config_hash: str
    analyzer: Literal["real", "fake"] = "real"
    device: str
    sample_rate: float
    frames_decoded: int
    frames_sampled: int
    faces_detected: int
    frames_scored: int
    started_at: datetime
    duration_ms: float
    stage_ms: dict[str, float]


class Calibration(_Model):
    method: Literal["none"] = "none"
    note: str = (
        "Scores are raw detector probabilities, not calibrated likelihoods. Thresholds are "
        "provisional until calibration (roadmap step 16)."
    )


class AnalysisResult(_Model):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    analysis_id: str
    verdict: Verdict
    manipulation_probability: float | None = Field(
        description="Highest track score among eligible face tracks; None if no face qualified"
    )
    confidence: Confidence
    video_quality: QualityLevel
    face_tracks: int
    detector_scores: dict[str, float]
    suspicious_intervals: list[SuspiciousInterval]
    verdict_reasons: list[str]
    quality: QualityDetails
    tracks: list[TrackResult]
    evidence: list[EvidenceFrame]
    video: VideoMetadata
    models: list[ModelInfo]
    processing: ProcessingInfo
    calibration: Calibration = Calibration()
    disclaimer: str = DISCLAIMER
