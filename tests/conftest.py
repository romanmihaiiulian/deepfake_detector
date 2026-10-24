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
    codec_options: dict[str, str] | None = None,
    image_for: Callable[[int], Image] | None = None,
) -> Path:
    """Encode a synthetic video. `times` (seconds) makes it variable frame rate."""
    codec = codec or DEFAULT_CODECS[path.suffix.lower()]
    count = frames if isinstance(frames, int) else len(frames)
    if not isinstance(frames, int) and count:
        size = (frames[0].shape[1], frames[0].shape[0])  # never let the encoder rescale
    time_base = Fraction(1, 1000) if times is not None else Fraction(1, fps)
    with av.open(str(path), "w", options=container_options or {}) as out:
        stream = out.add_stream(codec, rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.options = {**CODEC_OPTIONS.get(codec, {}), **(codec_options or {})}
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


# --------------------------------------------------------------------------- real faces

PORTRAIT_SIDE = 1200


@pytest.fixture(scope="session")
def portraits() -> dict[str, Image]:
    """Public-domain NASA portraits (RGB, longest side 1200 px), keyed by short name."""
    from PIL import Image as PILImage

    from tools.fetch_fixtures import FIXTURES_DIR, PORTRAITS

    loaded: dict[str, Image] = {}
    for fixture in PORTRAITS:
        path = FIXTURES_DIR / fixture.path
        if not path.is_file():
            pytest.skip("public-domain portrait fixtures missing: run `make fixtures`")
        with PILImage.open(path) as image:
            image.draft("RGB", (PORTRAIT_SIDE, PORTRAIT_SIDE))
            rgb = image.convert("RGB")
            rgb.thumbnail((PORTRAIT_SIDE, PORTRAIT_SIDE))
            loaded[Path(fixture.path).stem] = np.asarray(rgb)
    return loaded


@pytest.fixture(scope="session")
def yunet():  # type: ignore[no-untyped-def]
    from deepfake_core.errors import ModelMissingError
    from deepfake_core.faces import YuNetDetector

    try:
        return YuNetDetector()
    except ModelMissingError:
        pytest.skip("YuNet weights missing: run `make models`")


def face_patch(portrait: Image, detector, side: int = 240) -> Image:  # type: ignore[no-untyped-def]
    """A square patch centred on the portrait's face (face about 55% of the patch)."""
    import cv2

    detection = detector.detect(portrait)[0]
    x1, y1, x2, y2 = detection.box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = 0.9 * max(x2 - x1, y2 - y1)
    h, w = portrait.shape[:2]
    left, top = int(max(0, cx - half)), int(max(0, cy - half))
    right, bottom = int(min(w, cx + half)), int(min(h, cy + half))
    return cv2.resize(portrait[top:bottom, left:right], (side, side), interpolation=cv2.INTER_AREA)


def animate(
    patches: Sequence[tuple[Image, tuple[float, float], tuple[float, float]]],
    *,
    seconds: float = 3.0,
    fps: int = 25,
    canvas: tuple[int, int] = (640, 480),
    zoom: float = 0.06,
    roll_degrees: float = 4.0,
    occlusions: Sequence[tuple[int, float, float]] = (),
) -> list[Image]:
    """Frames with each patch moving from `start` at `velocity` (px/s), with gentle zoom
    and roll. `occlusions` are (patch index, start s, end s) during which that face is
    covered by a black rectangle."""
    import cv2

    width, height = canvas
    background = barcode_frame(0, canvas) // 3 + 60
    frames = []
    for i in range(round(seconds * fps)):
        t = i / fps
        frame = background.copy()
        for k, (patch, (x0, y0), (vx, vy)) in enumerate(patches):
            side = patch.shape[0]
            scale = 1.0 + zoom * np.sin(2 * np.pi * t / 2.0 + k)
            angle = roll_degrees * np.sin(2 * np.pi * t / 3.0 + k)
            matrix = cv2.getRotationMatrix2D((side / 2, side / 2), angle, scale)
            matrix[:, 2] += (x0 + vx * t - side / 2, y0 + vy * t - side / 2)
            cv2.warpAffine(
                patch, matrix, (width, height), dst=frame, borderMode=cv2.BORDER_TRANSPARENT
            )
            for index, start, end in occlusions:
                if index == k and start <= t < end:
                    cx, cy = int(x0 + vx * t), int(y0 + vy * t)
                    frame[
                        max(0, cy - side // 2) : cy + side // 2,
                        max(0, cx - side // 2) : cx + side // 2,
                    ] = 0
        frames.append(frame)
    return frames
