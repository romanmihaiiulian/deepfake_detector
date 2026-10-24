"""Video-level quality metrics: resolution, effective frame rate, compression, frame
availability."""

from __future__ import annotations

from deepfake_core.config import QualityConfig
from deepfake_core.quality.gate import Grade, grade
from deepfake_core.video.decode import DecodeStats
from deepfake_core.video.probe import VideoInfo

VIDEO_METRICS = ("resolution_short_side", "effective_fps", "bits_per_pixel", "frame_availability")


def video_metrics(info: VideoInfo, stats: DecodeStats) -> dict[str, float | None]:
    short_side = float(min(info.display_width, info.display_height))
    bits_per_pixel = None
    if info.bit_rate and info.fps:
        bits_per_pixel = info.bit_rate / (info.width * info.height * info.fps)
    availability: float | None = stats.grid_coverage if stats.grid_targets else None
    if info.fps and stats.duration > 0:
        decoded_ratio = min(1.0, stats.frames_decoded / (stats.duration * info.fps))
        availability = decoded_ratio if availability is None else min(availability, decoded_ratio)
    return {
        "resolution_short_side": short_side,
        "effective_fps": stats.effective_fps,
        "bits_per_pixel": bits_per_pixel,
        "frame_availability": availability,
    }


def grade_video(info: VideoInfo, stats: DecodeStats, config: QualityConfig) -> Grade:
    bands = {name: getattr(config, name) for name in VIDEO_METRICS}
    return grade(video_metrics(info, stats), bands)
