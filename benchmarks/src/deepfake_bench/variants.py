"""Encoding profiles. Real and fake members of a pair always get identical recipes.

`pristine` is the final encode of every lab sample, made from the near-lossless
intermediates. If real videos kept their original encoding while fakes were re-encoded,
a detector could learn codec history instead of manipulation (encoding parity). The other
profiles reproduce the robustness conditions of the handoff doc (M8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from deepfake_bench.media import FFMPEG, Encoded, describe, run_ffmpeg

Builder = Callable[[Path, Path], list[list[str]]]

_HEAD = [FFMPEG, "-nostdin", "-loglevel", "error", "-y"]
_TAIL = ["-an", "-sn", "-dn", "-map_metadata", "-1", "-movflags", "+faststart"]


def _x264(crf: int, preset: str = "medium", extra: list[str] | None = None) -> list[str]:
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"] + (
        extra or []
    )


def single(video_filter: str | None, codec: list[str]) -> Builder:
    def build(source: Path, dest: Path) -> list[list[str]]:
        vf = ["-vf", video_filter] if video_filter else []
        return [[*_HEAD, "-i", str(source), "-map", "0:v:0", *vf, *codec, *_TAIL, str(dest)]]

    return build


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    description: str
    build: Builder


PROFILES: dict[str, Profile] = {
    "pristine": Profile(
        "pristine",
        "Final high-quality encode (x264 crf 18) of the near-lossless intermediate",
        single(None, _x264(18)),
    ),
}


def apply_profile(profile: Profile, source: Path, dest: Path) -> Encoded:
    """Run every ffmpeg pass of a profile; returns the final file's description."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    passes = profile.build(source, dest)
    for argv in passes:
        run_ffmpeg(argv)
    flat = [arg for argv in passes for arg in [*argv, "&&"]][:-1]
    for intermediate in {Path(a[-1]) for a in passes[:-1]}:
        intermediate.unlink(missing_ok=True)
    return describe(dest, flat)
