"""Shared fixtures: a synthetic video factory built on PyAV, so tests need no media files.

Frames carry their own index as a 10-bit barcode (white/black blocks along the top edge)
that survives lossy encoding. A test can then read back exactly which source frame was
sampled.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import pytest

BARCODE_BITS = 10
BLOCK = 24
DEFAULT_CODECS = {".mp4": "libx264", ".mov": "libx264", ".webm": "libvpx-vp9"}
CODEC_OPTIONS = {
    "libx264": {"crf": "12", "preset": "ultrafast", "bf": "0"},
    "libvpx-vp9": {"crf": "20", "b": "0", "deadline": "realtime", "cpu-used": "8"},
}

Image = npt.NDArray[np.uint8]


def barcode_frame(index: int, size: tuple[int, int] = (320, 240)) -> Image:
    """An RGB frame whose top row of blocks encodes `index`, over a moving gradient."""
    width, height = size
    xs = np.arange(width, dtype=np.int32)[None, :]
    ys = np.arange(height, dtype=np.int32)[:, None]
    gradient = ((xs + ys + 7 * index) % 200 + 28).astype(np.uint8)
    image = np.repeat(gradient[:, :, None], 3, axis=2)
    image[: BLOCK + 8, :] = 128
    for bit in range(BARCODE_BITS):
        value = 255 if (index >> bit) & 1 else 0
        x0 = 4 + bit * BLOCK
        image[4 : 4 + BLOCK - 4, x0 : x0 + BLOCK - 4] = value
    return image


def read_barcode(image: Image) -> int:
    index = 0
    for bit in range(BARCODE_BITS):
        x0 = 4 + bit * BLOCK
        cell = image[4 : 4 + BLOCK - 4, x0 : x0 + BLOCK - 4].astype(np.float32)
        if cell.mean() > 128:
            index |= 1 << bit
    return index


def write_video(
    path: Path,
    *,
    frames: int | Sequence[Image] = 50,
    fps: int = 25,
    size: tuple[int, int] = (320, 240),
    times: Sequence[float] | None = None,
    codec: str | None = None,
    rotation: int = 0,
    container_options: dict[str, str] | None = None,
    image_for: Callable[[int], Image] | None = None,
) -> Path:
    """Encode a synthetic video. `times` (seconds) makes it variable frame rate."""
    codec = codec or DEFAULT_CODECS[path.suffix.lower()]
    count = frames if isinstance(frames, int) else len(frames)
    time_base = Fraction(1, 1000) if times is not None else Fraction(1, fps)
    with av.open(str(path), "w", options=container_options or {}) as out:
        stream = out.add_stream(codec, rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.options = CODEC_OPTIONS.get(codec, {})
        if rotation:
            stream.set_display_rotation(rotation)
        for i in range(count):
            if isinstance(frames, int):
                image = image_for(i) if image_for else barcode_frame(i, size)
            else:
                image = frames[i]
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = round(times[i] * 1000) if times is not None else i
            frame.time_base = time_base
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def write_audio_only(path: Path, seconds: float = 2.0) -> Path:
    rate = 44100
    with av.open(str(path), "w") as out:
        stream = out.add_stream("aac", rate=rate)
        samples = int(seconds * rate)
        for start in range(0, samples, 1024):
            data = np.zeros((1, 1024), dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(data, format="fltp", layout="mono")
            frame.sample_rate = rate
            frame.pts = start
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    counter = iter(range(1_000_000))

    def factory(name: str = "clip.mp4", **kwargs: object) -> Path:
        return write_video(tmp_path / f"{next(counter)}-{name}", **kwargs)  # type: ignore[arg-type]

    return factory


requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg CLI not found")
