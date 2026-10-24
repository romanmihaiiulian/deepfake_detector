"""Container and stream metadata, read without decoding the whole file."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import av
import av.error
from av.container import InputContainer
from av.video.stream import VideoStream

from deepfake_core.errors import VideoValidationError


@dataclass(frozen=True, slots=True)
class VideoInfo:
    container: str
    """Demuxer name(s) reported by FFmpeg, e.g. 'mov,mp4,m4a,3gp,3g2,mj2' or 'matroska,webm'."""
    codec: str
    width: int
    """Coded width, before display rotation."""
    height: int
    rotation: int
    """Display rotation in degrees counterclockwise (0, 90, 180 or 270)."""
    duration: float | None
    """Seconds. Taken from the stream or container header, otherwise measured from packet
    timestamps (MediaRecorder WebM files often declare none). None only for broken files."""
    duration_source: str
    """'stream', 'container', 'packets' or 'unknown'."""
    fps: float | None
    """Nominal (average) frame rate."""
    bit_rate: int | None
    frame_count: int | None
    has_audio: bool

    @property
    def display_width(self) -> int:
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation in (90, 270) else self.height


def open_input(path: str | os.PathLike[str]) -> InputContainer:
    """Open a container for reading, mapping FFmpeg failures to a validation error."""
    try:
        return av.open(os.fspath(path), mode="r")
    except (av.error.FFmpegError, UnicodeDecodeError) as exc:
        raise VideoValidationError("unreadable", "The file is not a readable video.") from exc


def first_video_stream(container: InputContainer) -> VideoStream | None:
    """The first real video stream, skipping cover art (attached pictures)."""
    for stream in container.streams.video:
        if not stream.disposition & av.stream.Disposition.attached_pic:
            return stream
    return None


def normalize_rotation(degrees: float) -> int:
    return round(degrees / 90.0) % 4 * 90


def _packet_span(path: str | os.PathLike[str]) -> float | None:
    """Duration measured from video packet timestamps (demux only, no decoding)."""
    with open_input(path) as container:
        stream = first_video_stream(container)
        if stream is None or stream.time_base is None:
            return None
        first: int | None = None
        last: int | None = None
        try:
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                end = packet.pts + (packet.duration or 0)
                first = packet.pts if first is None else min(first, packet.pts)
                last = end if last is None else max(last, end)
        except av.error.FFmpegError:
            pass
        if first is None or last is None or last <= first:
            return None
        return float((last - first) * stream.time_base)


def probe(path: str | os.PathLike[str]) -> VideoInfo:
    """Read container and stream metadata; decodes at most one frame (for rotation)."""
    return probe_with_first_frame(path)[0]


def probe_with_first_frame(path: str | os.PathLike[str]) -> tuple[VideoInfo, bool]:
    """Like `probe`, also reporting whether the first video frame decoded."""
    if not Path(path).is_file():
        raise VideoValidationError("unreadable", "The file does not exist.")
    with open_input(path) as container:
        stream = first_video_stream(container)
        if stream is None:
            raise VideoValidationError("no_video_stream", "The file contains no video stream.")
        codec_context = stream.codec_context
        duration: float | None = None
        duration_source = "unknown"
        if stream.duration and stream.time_base:
            duration, duration_source = float(stream.duration * stream.time_base), "stream"
        elif container.duration:
            duration, duration_source = container.duration / av.time_base, "container"
        rate = stream.average_rate or stream.guessed_rate
        rotation = 0
        decoded = False
        try:
            for frame in container.decode(stream):
                rotation = normalize_rotation(frame.rotation)
                decoded = True
                break
        except av.error.FFmpegError:
            pass  # validate() reports undecodable files with a precise code
        info = VideoInfo(
            container=container.format.name,
            codec=codec_context.name,
            width=codec_context.width,
            height=codec_context.height,
            rotation=rotation,
            duration=duration,
            duration_source=duration_source,
            fps=float(rate) if rate else None,
            bit_rate=stream.bit_rate or container.bit_rate or None,
            frame_count=stream.frames or None,
            has_audio=bool(container.streams.audio),
        )
    if info.duration is None:
        span = _packet_span(path)
        if span is not None:
            info = replace(info, duration=span, duration_source="packets")
    return info, decoded
