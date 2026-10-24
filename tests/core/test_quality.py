from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from deepfake_core.config import Band, QualityConfig
from deepfake_core.quality import (
    Tier,
    band_tier,
    combine,
    face_metrics,
    grade,
    grade_face,
    grade_video,
    median_tier,
)
from deepfake_core.quality.face_quality import yaw_proxy
from deepfake_core.video.decode import DecodeStats
from deepfake_core.video.probe import VideoInfo

CONFIG = QualityConfig()
FRONTAL = np.array([[87, 118], [169, 118], [128, 164], [95, 210], [161, 210]], np.float32)


def info(**overrides: object) -> VideoInfo:
    base: dict[str, object] = {
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec": "h264",
        "width": 1280,
        "height": 720,
        "rotation": 0,
        "duration": 10.0,
        "duration_source": "stream",
        "fps": 30.0,
        "bit_rate": 2_500_000,
        "frame_count": 300,
        "has_audio": False,
    }
    base.update(overrides)
    return VideoInfo(**base)  # type: ignore[arg-type]


def stats(**overrides: object) -> DecodeStats:
    base: dict[str, object] = {
        "sample_rate": 4.0,
        "frames_decoded": 300,
        "span": 299 / 30,
        "last_frame_duration": 1 / 30,
        "grid_targets": 40,
    }
    base.update(overrides)
    return DecodeStats(**base)  # type: ignore[arg-type]


def textured_face(seed: int = 0, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 216, size=(size // 4, size // 4, 3), dtype=np.uint8)
    return cv2.resize(base, (size, size), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- bands


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (300, Tier.GOOD),
        (200, Tier.GOOD),
        (150, Tier.FAIR),
        (64, Tier.POOR),
        (63, Tier.INSUFFICIENT),
    ],
)
def test_higher_is_better_band(value: float, expected: Tier) -> None:
    assert band_tier(Band(good=200, fair=120, poor=64), value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, Tier.GOOD), (0.1, Tier.FAIR), (0.3, Tier.POOR), (0.5, Tier.INSUFFICIENT)],
)
def test_lower_is_better_band(value: float, expected: Tier) -> None:
    band = Band(good=0.05, fair=0.15, poor=0.35, higher_is_better=False)
    assert band_tier(band, value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(128, Tier.GOOD), (45, Tier.FAIR), (215, Tier.FAIR), (25, Tier.POOR), (5, Tier.INSUFFICIENT)],
)
def test_range_band(value: float, expected: Tier) -> None:
    assert band_tier(CONFIG.face_exposure, value) is expected


@given(
    st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=3, max_size=3),
    st.floats(-2e3, 2e3, allow_nan=False),
    st.floats(0, 1e3, allow_nan=False),
    st.booleans(),
)
def test_band_tier_is_monotone(
    bounds: list[float], value: float, step: float, higher_is_better: bool
) -> None:
    """Moving a metric in its 'worse' direction never improves the tier."""
    poor, fair, good = sorted(bounds)
    if not higher_is_better:
        good, fair, poor = poor, fair, good
    band = Band(good=good, fair=fair, poor=poor, higher_is_better=higher_is_better)
    worse = value - step if higher_is_better else value + step
    assert band_tier(band, worse) <= band_tier(band, value)


@given(st.floats(-50, 300, allow_nan=False), st.floats(0, 100, allow_nan=False))
def test_range_band_is_monotone_away_from_the_good_interval(value: float, step: float) -> None:
    band = CONFIG.face_exposure
    centre = sum(band.good) / 2
    further = value - step if value < centre else value + step
    assert band_tier(band, further) <= band_tier(band, value)


# --------------------------------------------------------------------------- grading


def test_grade_takes_the_worst_tier_and_explains_it() -> None:
    result = grade(
        {"face_crop_size": 100.0, "face_sharpness": 500.0, "face_yaw": None},
        {"face_crop_size": CONFIG.face_crop_size, "face_sharpness": CONFIG.face_sharpness},
    )
    assert result.tier is Tier.POOR
    assert [r.code for r in result.reasons] == ["small_face"]
    assert "100 px" in result.reasons[0].message


def test_median_tier_is_the_lower_median() -> None:
    assert median_tier([Tier.GOOD, Tier.POOR]) is Tier.POOR
    assert median_tier([Tier.GOOD, Tier.GOOD, Tier.POOR]) is Tier.GOOD
    assert median_tier([]) is Tier.INSUFFICIENT


def test_combine_is_worst_of_and_deduplicates_reasons() -> None:
    small = grade({"face_crop_size": 100.0}, {"face_crop_size": CONFIG.face_crop_size})
    tiny = grade({"face_crop_size": 70.0}, {"face_crop_size": CONFIG.face_crop_size})
    merged = combine(small, tiny)
    assert merged.tier is Tier.POOR
    assert [r.code for r in merged.reasons] == ["small_face"]
    assert combine().tier is Tier.INSUFFICIENT


# --------------------------------------------------------------------------- face metrics


def metrics_for(crop: np.ndarray, **overrides: float) -> dict[str, float | None]:
    values: dict[str, float] = {
        "crop_size": 260.0,
        "detector_score": 0.95,
        "alignment_residual": 0.01,
    }
    values.update(overrides)
    return face_metrics(crop, landmarks=FRONTAL, **values)


def test_sharp_face_is_good_and_blurred_face_is_not() -> None:
    sharp = textured_face()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=6)
    sharp_metrics, blurred_metrics = metrics_for(sharp), metrics_for(blurred)
    sharp_value = sharp_metrics["face_sharpness"]
    blurred_value = blurred_metrics["face_sharpness"]
    assert sharp_value is not None and blurred_value is not None
    assert sharp_value > 10 * blurred_value
    assert grade_face(sharp_metrics, CONFIG).tier is Tier.GOOD
    assert grade_face(blurred_metrics, CONFIG).tier <= Tier.POOR


def test_dark_face_is_graded_down_for_lighting() -> None:
    dark = (textured_face() * 0.08).astype(np.uint8)
    result = grade_face(metrics_for(dark), CONFIG)
    assert result.tier <= Tier.POOR
    assert "poor_lighting" in {r.code for r in result.grade.reasons}


def test_small_face_is_insufficient() -> None:
    result = grade_face(metrics_for(textured_face(), crop_size=40.0), CONFIG)
    assert result.tier is Tier.INSUFFICIENT


def test_yaw_proxy_is_zero_for_frontal_and_roll_invariant() -> None:
    assert yaw_proxy(FRONTAL) == pytest.approx(0.0, abs=1e-6)
    turned = FRONTAL.copy()
    turned[2, 0] += 0.4 * (FRONTAL[1, 0] - FRONTAL[0, 0])
    angle = math.radians(35)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], np.float32
    )
    rolled = (turned - 128) @ rotation.T + 128
    assert yaw_proxy(turned) == pytest.approx(0.4, abs=1e-4)
    assert yaw_proxy(rolled) == pytest.approx(0.4, abs=1e-4)


# --------------------------------------------------------------------------- video metrics


def test_good_video_is_good() -> None:
    result = grade_video(info(), stats(), CONFIG)
    assert result.tier is Tier.GOOD, result.reasons


def test_conferencing_like_video_is_poor() -> None:
    conf = info(width=640, height=360, fps=15.0, bit_rate=25_000)
    result = grade_video(conf, stats(frames_decoded=150, span=149 / 15), CONFIG)
    assert result.tier is Tier.POOR
    assert {"low_resolution", "high_compression"} <= {r.code for r in result.reasons}


def test_slideshow_is_insufficient() -> None:
    result = grade_video(info(), stats(duplicate_frames=295), CONFIG)
    assert result.tier is Tier.INSUFFICIENT
    assert result.reasons[0].code == "low_frame_rate"


def test_missing_frames_lower_availability() -> None:
    result = grade_video(info(), stats(frames_decoded=120, grid_far_targets=20), CONFIG)
    assert result.metrics["frame_availability"] == pytest.approx(0.4, abs=0.01)
    assert result.tier is Tier.INSUFFICIENT


def test_rotated_video_uses_display_short_side() -> None:
    portrait = info(width=1920, height=1080, rotation=90)
    assert grade_video(portrait, stats(), CONFIG).metrics["resolution_short_side"] == 1080
