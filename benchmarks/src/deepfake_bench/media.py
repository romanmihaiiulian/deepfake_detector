"""ffmpeg invocations for the lab. Every encode records its exact argv and ffmpeg version.

The lab uses the system `ffmpeg` CLI (Homebrew's build is GPL). It is an internal tool and
is never distributed.
"""

from __future__ import annotations

import functools
import hashlib
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from deepfake_core.video import probe

FFMPEG = "ffmpeg"


class MediaError(RuntimeError):
    pass


@functools.cache
def ffmpeg_version() -> str:
    if shutil.which(FFMPEG) is None:
        raise MediaError("ffmpeg not found on PATH (macOS: `brew install ffmpeg`)")
    first = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, check=True)
    return first.stdout.splitlines()[0]


def run_ffmpeg(argv: Sequence[str]) -> None:
    ffmpeg_version()
    completed = subprocess.run(list(argv), capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-5:])
        raise MediaError(f"ffmpeg failed ({completed.returncode}): {tail}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Encoded:
    path: Path
    argv: list[str]
    ffmpeg_version: str
    width: int
    height: int
    fps: float | None
    duration: float | None
    sha256: str


def describe(path: Path, argv: list[str]) -> Encoded:
    info = probe(path)
    return Encoded(
        path=path,
        argv=argv,
        ffmpeg_version=ffmpeg_version(),
        width=info.display_width,
        height=info.display_height,
        fps=info.fps,
        duration=info.duration,
        sha256=sha256_of(path),
    )


def canonicalize(
    source: Path, dest: Path, *, max_seconds: float, max_fps: float, max_height: int
) -> Encoded:
    """Near-lossless, constant-frame-rate, audio-free intermediate from a source video.

    Both the real sample and the FaceFusion input derive from this file, so codec history
    is identical up to the final encode.
    """
    info = probe(source)
    fps = min(info.fps or max_fps, max_fps)
    height = min(info.display_height, max_height)
    scale = (
        f"scale=-2:{height - height % 2}"
        if info.display_height > max_height
        else ("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    )
    argv = [
        FFMPEG, "-nostdin", "-loglevel", "error", "-y",
        "-i", str(source),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-t", f"{max_seconds:g}",
        "-vf", f"fps={fps:.6g},{scale}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv420p",
        "-map_metadata", "-1",
        str(dest),
    ]  # fmt: skip
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(argv)
    return describe(dest, argv)
