"""Video input: probing, validation, decoding and deterministic frame sampling."""

from deepfake_core.video.decode import DecodeStats, SampledFrame, VideoReader
from deepfake_core.video.probe import VideoInfo, probe
from deepfake_core.video.sampling import SamplingGrid, plan_sample_rate
from deepfake_core.video.validate import ALLOWED_EXTENSIONS, validate

__all__ = [
    "ALLOWED_EXTENSIONS",
    "DecodeStats",
    "SampledFrame",
    "SamplingGrid",
    "VideoInfo",
    "VideoReader",
    "plan_sample_rate",
    "probe",
    "validate",
]
