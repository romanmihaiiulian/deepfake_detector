"""Upload validation: is this an MP4, MOV or WebM video we can analyze?

Validation is deliberately cheap: it reads metadata and decodes a single frame. Duration
limits for files that don't declare a duration are enforced while decoding
(`VideoReader`), so a hostile file can't bypass them.
"""

from __future__ import annotations

import os
from pathlib import Path

from deepfake_core.config import ValidationConfig
from deepfake_core.errors import VideoValidationError
from deepfake_core.video.probe import VideoInfo, probe_with_first_frame

ALLOWED_EXTENSIONS: dict[str, tuple[str, ...]] = {
    # extension -> FFmpeg demuxer names that may legitimately back it
    ".mp4": ("mp4", "mov"),
    ".mov": ("mov", "mp4"),
    ".webm": ("webm", "matroska"),
}


def validate(
    path: str | os.PathLike[str],
    config: ValidationConfig | None = None,
) -> VideoInfo:
    """Return the video's metadata, or raise `VideoValidationError` with a stable code.

    Codes: unsupported_extension, empty_file, unreadable, container_mismatch,
    no_video_stream, undecodable, too_large_resolution, too_short, too_long.
    """
    config = config or ValidationConfig()
    file = Path(path)
    extension = file.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise VideoValidationError(
            "unsupported_extension", "Supported formats are MP4, MOV and WEBM."
        )
    if not file.is_file() or file.stat().st_size == 0:
        raise VideoValidationError("empty_file", "The file is empty.")

    info, decoded = probe_with_first_frame(file)
    demuxers = set(info.container.split(","))
    if not demuxers & set(ALLOWED_EXTENSIONS[extension]):
        raise VideoValidationError(
            "container_mismatch",
            f"The file content is not a {extension[1:].upper()} container.",
        )
    if info.width <= 0 or info.height <= 0 or not decoded:
        raise VideoValidationError("undecodable", "No video frame could be decoded.")
    if max(info.width, info.height) > config.max_dimension_px:
        raise VideoValidationError(
            "too_large_resolution",
            f"Videos up to {config.max_dimension_px} px on the longest side are supported.",
        )
    if info.duration is not None:
        if info.duration < config.min_duration_s:
            raise VideoValidationError(
                "too_short", f"The video must be at least {config.min_duration_s:g} s long."
            )
        if info.duration > config.max_duration_s:
            raise VideoValidationError(
                "too_long", f"The video must be at most {config.max_duration_s:g} s long."
            )
    return info
