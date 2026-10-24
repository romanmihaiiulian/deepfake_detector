from __future__ import annotations

import subprocess
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepfake_core.config import SamplingConfig, ValidationConfig
from deepfake_core.errors import VideoValidationError
from deepfake_core.video import SamplingGrid, VideoReader, plan_sample_rate, probe, validate
from tests.conftest import read_barcode, requires_ffmpeg, write_audio_only

MakeVideo = Callable[..., Path]


def sampled(path: Path, **sampling: float) -> list[tuple[int, float, int]]:
    """(frame_index, time, barcode) of every sampled frame."""
    info = validate(path)
    reader = VideoReader(path, info, SamplingConfig(**sampling))  # type: ignore[arg-type]
    return [(s.frame_index, s.time, read_barcode(s.image)) for s in reader]


def nearest_reference(times: list[Fraction], rate: Fraction, tolerance: Fraction) -> list[int]:
    """Brute-force nearest-frame selection: argmin distance, ties to the earlier frame."""
    chosen: list[int] = []
    k = 0
    while k / rate <= times[-1] + tolerance:
        target = k / rate
        best = min(range(len(times)), key=lambda i: (abs(times[i] - target), i))
        if not chosen or chosen[-1] != best:
            chosen.append(best)
        k += 1
    return chosen


# --------------------------------------------------------------------------- probe/validate


@pytest.mark.parametrize("name", ["clip.mp4", "clip.mov", "clip.webm"])
def test_probe_reads_metadata(make_video: MakeVideo, name: str) -> None:
    path = make_video(name, frames=50, fps=25, size=(320, 240))
    info = validate(path)
    assert (info.width, info.height, info.rotation) == (320, 240, 0)
    assert info.fps == pytest.approx(25)
    assert info.duration == pytest.approx(2.0, abs=0.05)
    assert info.codec in {"h264", "vp9"}


def test_probe_measures_duration_when_header_has_none(make_video: MakeVideo) -> None:
    # MediaRecorder-style WebM: muxed as a live stream, so no Duration element
    path = make_video("live.webm", frames=40, fps=20, container_options={"live": "1"})
    info = probe(path)
    assert info.duration_source == "packets"
    assert info.duration == pytest.approx(2.0, abs=0.06)


def test_probe_reads_display_rotation(make_video: MakeVideo) -> None:
    info = probe(make_video("rot.mp4", frames=10, rotation=90, size=(320, 240)))
    assert info.rotation == 90
    assert (info.display_width, info.display_height) == (240, 320)


def test_validate_rejects_wrong_extension(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=30)
    renamed = path.with_suffix(".avi")
    path.rename(renamed)
    with pytest.raises(VideoValidationError) as error:
        validate(renamed)
    assert error.value.code == "unsupported_extension"


def test_validate_rejects_renamed_text_file(tmp_path: Path) -> None:
    fake = tmp_path / "notes.mp4"
    fake.write_text("these are not the frames you are looking for\n" * 100)
    with pytest.raises(VideoValidationError) as error:
        validate(fake)
    assert error.value.code == "unreadable"


def test_validate_rejects_image_renamed_to_mp4(tmp_path: Path) -> None:
    fake = tmp_path / "photo.mp4"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(fake, format="PNG")
    with pytest.raises(VideoValidationError) as error:
        validate(fake)
    assert error.value.code in {"container_mismatch", "unreadable"}


def test_validate_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.webm"
    empty.touch()
    with pytest.raises(VideoValidationError) as error:
        validate(empty)
    assert error.value.code == "empty_file"


def test_validate_rejects_audio_only(tmp_path: Path) -> None:
    path = write_audio_only(tmp_path / "voice.mp4")
    with pytest.raises(VideoValidationError) as error:
        validate(path)
    assert error.value.code == "no_video_stream"


def test_validate_rejects_webm_content_named_mp4(make_video: MakeVideo) -> None:
    path = make_video("clip.webm", frames=30)
    renamed = path.with_suffix(".mp4")
    path.rename(renamed)
    with pytest.raises(VideoValidationError) as error:
        validate(renamed)
    assert error.value.code == "container_mismatch"


def test_validate_duration_limits(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=75, fps=25)  # 3 s
    with pytest.raises(VideoValidationError) as error:
        validate(path, ValidationConfig(max_duration_s=2.0))
    assert error.value.code == "too_long"
    with pytest.raises(VideoValidationError) as error:
        validate(path, ValidationConfig(min_duration_s=5.0))
    assert error.value.code == "too_short"


def test_validate_rejects_huge_resolution(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=25, size=(320, 240))
    with pytest.raises(VideoValidationError) as error:
        validate(path, ValidationConfig(max_dimension_px=256, min_duration_s=0.5))
    assert error.value.code == "too_large_resolution"


def test_reader_enforces_duration_even_if_header_lies(make_video: MakeVideo) -> None:
    from dataclasses import replace

    path = make_video("clip.mp4", frames=75, fps=25)  # 3 s
    lying = replace(validate(path), duration=1.0)
    reader = VideoReader(path, lying, validation=ValidationConfig(max_duration_s=2.0))
    with pytest.raises(VideoValidationError) as error:
        list(reader)
    assert error.value.code == "too_long"


# --------------------------------------------------------------------------- sampling


def test_plan_sample_rate() -> None:
    config = SamplingConfig(rate_hz=4.0, max_samples=960)
    assert plan_sample_rate(60.0, 30.0, config) == 4
    assert float(plan_sample_rate(480.0, 30.0, config)) == pytest.approx(2.0)
    assert plan_sample_rate(10.0, 2.0, config) == 2  # never faster than the video


def test_grid_breaks_ties_towards_the_earlier_frame() -> None:
    grid = SamplingGrid(Fraction(4))
    times = [Fraction(i, 25) for i in range(26)]  # 25 fps: target 0.5 sits between 12 and 13
    selected = [index for i, t in enumerate(times) for index in grid.push(i, t)]
    assert selected[:3] == [0, 6, 12]
    assert selected == nearest_reference(times, Fraction(4), Fraction(0))


def test_grid_counts_far_targets_when_frames_are_missing() -> None:
    grid = SamplingGrid(Fraction(4))
    for i, t in enumerate([Fraction(0), Fraction(1, 10), Fraction(3), Fraction(31, 10)]):
        grid.push(i, t)
    grid.finish()
    # targets 0.25 .. 2.75 have no frame within 0.125 s
    assert grid.far_targets >= 9
    assert grid.targets == 13


def test_cfr_sampling_matches_reference(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=100, fps=25)
    samples = sampled(path)
    times = [Fraction(i, 25) for i in range(100)]
    expected = nearest_reference(times, Fraction(4), Fraction(1, 50))
    assert [frame for frame, _, _ in samples] == expected
    assert all(code == frame for frame, _, code in samples)  # the right pixels came out
    assert all(abs(t - frame / 25) < 1e-9 for frame, t, _ in samples)


def test_vfr_sampling_uses_timestamps_not_indices(make_video: MakeVideo) -> None:
    rng = np.random.default_rng(7)
    gaps = rng.uniform(0.01, 0.12, size=80)
    times_s = np.concatenate([[0.0], np.cumsum(gaps)])[:80]
    times_ms = [round(t * 1000) / 1000 for t in times_s]
    path = make_video("vfr.mp4", frames=80, times=times_ms, fps=30)
    samples = sampled(path)
    exact = [Fraction(round(t * 1000), 1000) for t in times_ms]
    fps = Fraction(validate(path).fps or 30).limit_denominator(100_000)
    rate = plan_sample_rate(validate(path).duration, validate(path).fps, SamplingConfig())
    expected = nearest_reference(exact, rate, 1 / fps / 2)
    assert [frame for frame, _, _ in samples] == expected
    assert all(code == frame for frame, _, code in samples)


def test_sampling_is_deterministic(make_video: MakeVideo) -> None:
    path = make_video("clip.webm", frames=60, fps=30)
    assert sampled(path) == sampled(path)


def test_rotation_is_applied_to_sampled_frames(make_video: MakeVideo) -> None:
    path = make_video("rot.mp4", frames=30, rotation=90, size=(320, 240))
    reader = VideoReader(path, validate(path, ValidationConfig(min_duration_s=0.5)))
    first = next(iter(reader))
    assert first.image.shape == (320, 240, 3)
    # rotating back recovers the barcode
    assert read_barcode(np.rot90(first.image, k=-1)) == 0


def test_stats_report_effective_fps_and_duplicates(make_video: MakeVideo) -> None:
    from tests.conftest import barcode_frame

    # every source frame shown twice: 30 fps container, 15 fps of real motion
    path = make_video("dup.mp4", frames=90, fps=30, image_for=lambda i: barcode_frame(i // 2))
    reader = VideoReader(path, validate(path))
    list(reader)
    stats = reader.stats
    assert stats.frames_decoded == 90
    assert stats.duplicate_frames == pytest.approx(45, abs=2)
    assert stats.effective_fps == pytest.approx(15, abs=1)
    assert stats.duration == pytest.approx(3.0, abs=0.05)
    assert stats.grid_coverage == 1.0


def test_sample_cap_lowers_the_rate(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=100, fps=25)  # 4 s
    samples = sampled(path, max_samples=8)
    assert 8 <= len(samples) <= 9


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_rotation_metadata_written_by_ffmpeg(make_video: MakeVideo, tmp_path: Path) -> None:
    source = make_video("clip.mp4", frames=30)
    rotated = tmp_path / "phone.mov"
    argv = ["ffmpeg", "-loglevel", "error", "-y", "-display_rotation:v:0", "-90"]
    subprocess.run([*argv, "-i", str(source), "-c", "copy", str(rotated)], check=True)
    info = validate(rotated, ValidationConfig(min_duration_s=0.5))
    assert info.rotation == 270
    first = next(iter(VideoReader(rotated, info)))
    assert read_barcode(np.rot90(first.image, k=1)) == 0
