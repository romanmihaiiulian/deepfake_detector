"""Sequential decoding with deterministic sampling.

Every frame is decoded, because sequential decoding is the only reliable way to get exact
timestamps for variable-frame-rate and broken-index files. Only sampled frames are
converted to RGB. A tiny luma thumbnail of every frame is compared with its predecessor
to count duplicated frames, which gives the effective frame rate. Screen recordings and
frame-rate conversions often repeat frames.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

import av.error
import numpy as np
import numpy.typing as npt
from av.video.frame import VideoFrame

from deepfake_core.config import SamplingConfig, ValidationConfig
from deepfake_core.errors import VideoValidationError
from deepfake_core.video.probe import (
    VideoInfo,
    first_video_stream,
    normalize_rotation,
    open_input,
)
from deepfake_core.video.sampling import SamplingGrid, plan_sample_rate

RGBImage = npt.NDArray[np.uint8]

_LUMA_8BIT = frozenset(
    {
        "gray",
        "nv12",
        "nv21",
        "yuv410p",
        "yuv411p",
        "yuv420p",
        "yuv422p",
        "yuv440p",
        "yuv444p",
        "yuvj420p",
        "yuvj422p",
        "yuvj440p",
        "yuvj444p",
    }
)
_LUMA_10BIT = frozenset({"yuv420p10le", "yuv422p10le", "yuv444p10le"})
_THUMB_ROWS, _THUMB_COLS = 36, 64


@dataclass(frozen=True, slots=True)
class SampledFrame:
    sample_index: int
    frame_index: int
    """Index of the frame in presentation order."""
    time: float
    """Seconds since the first decoded frame."""
    image: RGBImage
    """RGB, uint8, (height, width, 3), in display orientation."""


@dataclass(slots=True)
class DecodeStats:
    sample_rate: float = 0.0
    frames_decoded: int = 0
    duplicate_frames: int = 0
    decode_errors: int = 0
    out_of_order_frames: int = 0
    samples: int = 0
    grid_targets: int = 0
    grid_far_targets: int = 0
    span: float = 0.0
    """Time of the last frame minus time of the first frame."""
    last_frame_duration: float = 0.0

    @property
    def duration(self) -> float:
        return self.span + self.last_frame_duration

    @property
    def effective_fps(self) -> float | None:
        if self.duration <= 0:
            return None
        return (self.frames_decoded - self.duplicate_frames) / self.duration

    @property
    def grid_coverage(self) -> float:
        if self.grid_targets == 0:
            return 0.0
        return 1.0 - self.grid_far_targets / self.grid_targets


def luma_thumbnail(frame: VideoFrame) -> npt.NDArray[np.int16]:
    """A 36x64 grid of luma samples, read straight from the Y plane when possible."""
    name = frame.format.name
    if name in _LUMA_8BIT or name in _LUMA_10BIT:
        plane = frame.planes[0]
        dtype = np.dtype("<u2") if name in _LUMA_10BIT else np.dtype(np.uint8)
        stride = plane.line_size // dtype.itemsize
        rows = np.frombuffer(plane, dtype=dtype)[: plane.height * stride]
        luma = rows.reshape(plane.height, stride)[: frame.height, : frame.width]
        if name in _LUMA_10BIT:
            luma = np.right_shift(luma, 2)
    else:
        luma = frame.to_ndarray(format="gray")
    ys = np.linspace(0, luma.shape[0] - 1, _THUMB_ROWS).astype(np.intp)
    xs = np.linspace(0, luma.shape[1] - 1, _THUMB_COLS).astype(np.intp)
    return luma[np.ix_(ys, xs)].astype(np.int16)


def to_rgb(frame: VideoFrame) -> RGBImage:
    """RGB image in display orientation (display matrix applied)."""
    image: RGBImage = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
    turns = normalize_rotation(frame.rotation) // 90
    if turns:
        image = np.ascontiguousarray(np.rot90(image, k=turns))
    return image


def _exact_time(frame: VideoFrame) -> Fraction | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return Fraction(frame.pts) * frame.time_base


class VideoReader:
    """Iterate the sampled frames of a validated video.

    `stats` is complete once iteration ends. Iteration raises `VideoValidationError` when
    decoding shows that the video is longer or shorter than allowed. Decoding does not
    trust the header's duration for this check.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        info: VideoInfo,
        sampling: SamplingConfig | None = None,
        validation: ValidationConfig | None = None,
    ) -> None:
        self.path = path
        self.info = info
        self.sampling = sampling or SamplingConfig()
        self.validation = validation or ValidationConfig()
        self.rate = plan_sample_rate(info.duration, info.fps, self.sampling)
        self.stats = DecodeStats(sample_rate=float(self.rate))

    def __iter__(self) -> Iterator[SampledFrame]:
        stats = self.stats = DecodeStats(sample_rate=float(self.rate))
        fps = Fraction(self.info.fps).limit_denominator(100_000) if self.info.fps else Fraction(30)
        nominal_interval = 1 / fps
        grid = SamplingGrid(self.rate, end_tolerance=nominal_interval / 2)
        threshold = self.sampling.duplicate_max_abs_diff
        max_duration = Fraction(self.validation.max_duration_s)

        prev: tuple[VideoFrame, int, Fraction] | None = None
        prev_thumb: npt.NDArray[np.int16] | None = None
        first: Fraction | None = None
        last_raw: Fraction | None = None

        def emit(index: int, current: tuple[VideoFrame, int, Fraction]) -> SampledFrame:
            frame, frame_index, time = prev if prev is not None and index == prev[1] else current
            sample = SampledFrame(stats.samples, frame_index, float(time), to_rgb(frame))
            stats.samples += 1
            return sample

        with open_input(self.path) as container:
            stream = first_video_stream(container)
            if stream is None:
                raise VideoValidationError("no_video_stream", "The file contains no video stream.")
            stream.thread_type = "AUTO"
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.error.FFmpegError:
                    stats.decode_errors += 1
                    continue
                for frame in frames:
                    raw = _exact_time(frame)
                    if raw is None:
                        raw = (last_raw if last_raw is not None else Fraction(0)) + nominal_interval
                    if last_raw is not None and raw <= last_raw:
                        stats.out_of_order_frames += 1
                        continue
                    last_raw = raw
                    if first is None:
                        first = raw
                    time = raw - first
                    if time > max_duration:
                        raise VideoValidationError(
                            "too_long",
                            f"The video must be at most {self.validation.max_duration_s:g} s long.",
                        )
                    index = stats.frames_decoded
                    stats.frames_decoded += 1
                    stats.span = float(time)
                    if frame.duration and frame.time_base:
                        stats.last_frame_duration = float(frame.duration * frame.time_base)
                    else:
                        stats.last_frame_duration = float(nominal_interval)

                    thumb = luma_thumbnail(frame)
                    if (
                        prev_thumb is not None
                        and int(np.abs(thumb - prev_thumb).max()) <= threshold
                    ):
                        stats.duplicate_frames += 1
                    prev_thumb = thumb

                    current = (frame, index, time)
                    for selected in grid.push(index, time):
                        yield emit(selected, current)
                    prev = current

            if prev is not None:
                for selected in grid.finish():
                    yield emit(selected, prev)

        stats.grid_targets = grid.targets
        stats.grid_far_targets = grid.far_targets
        if stats.frames_decoded == 0:
            raise VideoValidationError("undecodable", "No video frame could be decoded.")
        if stats.duration < self.validation.min_duration_s:
            raise VideoValidationError(
                "too_short",
                f"The video must be at least {self.validation.min_duration_s:g} s long.",
            )
