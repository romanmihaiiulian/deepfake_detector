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


def conferencing(height: int, fps: int, kbps: int) -> Builder:
    """A WebRTC-like chain: realtime VP8 at a capped bitrate (the call), then an H.264
    re-encode at the same bitrate (the meeting recording)."""

    def build(source: Path, dest: Path) -> list[list[str]]:
        call = dest.with_name(dest.stem + ".call.webm")
        vp8 = [
            "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8",
            "-b:v", f"{kbps}k", "-minrate", f"{kbps}k", "-maxrate", f"{kbps}k",
            "-bufsize", f"{kbps}k", "-g", str(fps * 3), "-error-resilient", "1",
        ]  # fmt: skip
        h264 = [
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k",
            "-pix_fmt", "yuv420p",
        ]  # fmt: skip
        scale = f"scale=-2:'min({height},ih)',fps={fps}"
        return [
            [*_HEAD, "-i", str(source), "-map", "0:v:0", "-vf", scale, *vp8, "-an", str(call)],
            [*_HEAD, "-i", str(call), "-map", "0:v:0", *h264, *_TAIL, str(dest)],
        ]

    return build


def screen_recording(source: Path, dest: Path) -> list[list[str]]:
    """The call shown in a window on a 1080p desktop, captured by a screen recorder."""
    graph = (
        "[0:v]scale=w=1152:h=648:force_original_aspect_ratio=decrease,"
        "pad=1152:648:(ow-iw)/2:(oh-ih)/2:black,eq=gamma=1.05:saturation=0.95[window];"
        "color=c=0x202428:s=1920x1080:r=30[desktop];"
        "[desktop][window]overlay=x=384:y=216:shortest=1,format=yuv420p[out]"
    )
    codec = _x264(28, "veryfast", ["-tune", "zerolatency", "-r", "30"])
    return [
        [
            *_HEAD,
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            *codec,
            *_TAIL,
            str(dest),
        ]
    ]


PROFILES: dict[str, Profile] = {
    "pristine": Profile(
        "pristine",
        "Final high-quality encode (x264 crf 18) of the near-lossless intermediate",
        single(None, _x264(18)),
    ),
    "h264_crf23": Profile("h264_crf23", "H.264 recompression, crf 23", single(None, _x264(23))),
    "h264_crf28": Profile("h264_crf28", "H.264 recompression, crf 28", single(None, _x264(28))),
    "h264_crf35": Profile(
        "h264_crf35", "H.264 heavy recompression, crf 35", single(None, _x264(35))
    ),
    "h265_crf28": Profile(
        "h265_crf28",
        "H.265/HEVC recompression, crf 28",
        single(
            None,
            [
                *("-c:v", "libx265", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p"),
                *("-tag:v", "hvc1", "-x265-params", "log-level=error"),
            ],
        ),
    ),
    "res_1080p": Profile(
        "res_1080p", "At most 1080 lines", single("scale=-2:'min(1080,ih)'", _x264(23))
    ),
    "res_720p": Profile(
        "res_720p", "At most 720 lines", single("scale=-2:'min(720,ih)'", _x264(23))
    ),
    "res_480p": Profile(
        "res_480p", "At most 480 lines", single("scale=-2:'min(480,ih)'", _x264(23))
    ),
    "fps_30": Profile("fps_30", "Resampled to 30 fps", single("fps=30", _x264(23))),
    "fps_15": Profile("fps_15", "Resampled to 15 fps", single("fps=15", _x264(23))),
    "crop_10": Profile(
        "crop_10",
        "Centre crop to 90% of width and height",
        single("crop=trunc(iw*0.9/2)*2:trunc(ih*0.9/2)*2", _x264(23)),
    ),
    "letterbox": Profile(
        "letterbox",
        "Black bars added above and below (4:3 frame around the picture)",
        single("pad=iw:trunc(ih*4/3/2)*2:0:(oh-ih)/2:black", _x264(23)),
    ),
    "screen_recording": Profile(
        "screen_recording",
        "Call window on a 1080p desktop, screen-recorder settings (x264 crf 28, 30 fps)",
        screen_recording,
    ),
    "conf_low": Profile(
        "conf_low",
        "Conference low: 360p, 15 fps, 300 kbps VP8 then H.264",
        conferencing(360, 15, 300),
    ),
    "conf_medium": Profile(
        "conf_medium", "Conference medium: 540p, 24 fps, 800 kbps", conferencing(540, 24, 800)
    ),
    "conf_high": Profile(
        "conf_high", "Conference high: 720p, 30 fps, 1.5 Mbps", conferencing(720, 30, 1500)
    ),
}

ROBUSTNESS_PROFILES = tuple(name for name in PROFILES if name != "pristine")


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
