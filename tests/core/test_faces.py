from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deepfake_core.config import QualityConfig, TrackingConfig
from deepfake_core.faces import (
    Detection,
    FaceTracker,
    align_face,
    model_input,
    native_crop_size,
)
from deepfake_core.faces.align import TEMPLATE, template_points
from deepfake_core.quality import Tier, face_metrics, grade_face
from deepfake_core.video import VideoReader, validate
from tests.conftest import animate, face_patch

MakeVideo = Callable[..., Path]


def det(x: float, y: float, size: float = 80.0, score: float = 0.95) -> Detection:
    box = np.array([x, y, x + size, y + size], np.float32)
    landmarks = (TEMPLATE * size + np.array([x, y], np.float32)).astype(np.float32)
    return Detection(box, landmarks, score)


def run_tracker(
    frames: list[tuple[float, list[Detection]]], config: TrackingConfig | None = None
) -> tuple[FaceTracker, list[list[int]]]:
    tracker = FaceTracker(config)
    ids = [tracker.update(time, detections) for time, detections in frames]
    return tracker, ids


# --------------------------------------------------------------------------- tracker


def test_single_moving_face_is_one_track() -> None:
    frames = [(k / 4, [det(100 + 20 * k, 100 + 5 * k)]) for k in range(20)]
    tracker, ids = run_tracker(frames)
    assert {i for row in ids for i in row} == {0}
    assert list(tracker.finalize()) == [0]


def test_crossing_faces_keep_their_identities() -> None:
    frames = []
    for k in range(17):
        t = k / 4
        frames.append((t, [det(100 + 100 * t, 200), det(500 - 100 * t, 205)]))
    tracker, ids = run_tracker(frames)
    mapping = tracker.id_map()
    left_to_right = [mapping[row[0]] for row in ids]
    right_to_left = [mapping[row[1]] for row in ids]
    assert len(set(left_to_right)) == 1 and len(set(right_to_left)) == 1
    assert left_to_right[0] != right_to_left[0]
    assert len(tracker.finalize()) == 2


@pytest.mark.parametrize(("gap", "tracks"), [(1.5, 1), (3.0, 2)])
def test_occlusion_gap(gap: float, tracks: int) -> None:
    before = [(k / 4, [det(200, 200)]) for k in range(8)]
    resume = before[-1][0] + gap
    after = [(resume + k / 4, [det(204, 200)]) for k in range(8)]
    tracker, _ = run_tracker(before + after)
    assert len(tracker.finalize()) == tracks


def test_short_spurious_tracks_are_dropped_and_ids_renumbered() -> None:
    frames: list[tuple[float, list[Detection]]] = []
    for k in range(12):
        detections = [det(300, 300)]
        if k == 3:
            detections.append(det(20, 20, size=40))  # a one-off false positive
        frames.append((k / 4, detections))
    tracker, ids = run_tracker(frames)
    kept = tracker.finalize()
    assert list(kept) == [0]
    assert kept[0].detections == 12
    assert tracker.id_map() == {ids[0][0]: 0}


def test_far_jump_starts_a_new_track() -> None:
    frames = [(k / 4, [det(100, 100)]) for k in range(6)]
    frames += [(1.5 + k / 4, [det(500, 400)]) for k in range(6)]
    tracker, _ = run_tracker(frames)
    assert len(tracker.finalize()) == 2


def test_scale_gate_separates_faces_of_very_different_size() -> None:
    frames = [(k / 4, [det(100, 100, size=60)]) for k in range(6)]
    frames += [(1.5 + k / 4, [det(100, 100, size=200)]) for k in range(6)]
    tracker, _ = run_tracker(frames)
    assert len(tracker.finalize()) == 2


# --------------------------------------------------------------------------- alignment


def test_landmarks_on_the_template_are_reproduced() -> None:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, size=(400, 400, 3), dtype=np.uint8)
    side = 200
    offset = np.array([90, 60], np.float32)
    landmarks = template_points(side) + offset
    aligned = align_face(image, landmarks, size=side)
    assert aligned is not None
    mapped = landmarks @ aligned.matrix[:, :2].T + aligned.matrix[:, 2]
    assert np.abs(mapped - template_points(side)).max() < 0.5
    assert aligned.residual < 1e-3
    np.testing.assert_allclose(aligned.matrix[:, :2], np.eye(2), atol=1e-3)
    np.testing.assert_array_equal(aligned.crop, image[60 : 60 + side, 90 : 90 + side])


def test_alignment_is_deterministic_and_rotation_invariant() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, size=(500, 500, 3), dtype=np.uint8)
    base = template_points(180) + 150
    angle = np.deg2rad(20)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = ((base - 240) @ rotation.T + 240).astype(np.float32)
    first = align_face(image, rotated)
    second = align_face(image, rotated)
    assert first is not None and second is not None
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.crop, second.crop)
    assert first.residual < 1e-3
    mapped = rotated @ first.matrix[:, :2].T + first.matrix[:, 2]
    assert np.abs(mapped - template_points(first.size)).max() < 0.5


def test_native_crop_size_scales_with_the_face() -> None:
    small = native_crop_size(TEMPLATE * 100)
    large = native_crop_size(TEMPLATE * 200)
    assert small == 130  # GenD: mean(landmark distances / template distances) * 1.3
    assert large == 260


def test_model_input_resizes_to_224() -> None:
    crop = np.zeros((300, 300, 3), np.uint8)
    assert model_input(crop).shape == (224, 224, 3)
    exact = np.zeros((224, 224, 3), np.uint8)
    assert model_input(exact) is exact


# --------------------------------------------------------------------------- real faces


@pytest.mark.models
def test_yunet_finds_each_portrait_face_with_template_landmark_order(
    portraits: dict[str, np.ndarray], yunet: Any
) -> None:
    for name, image in portraits.items():
        detections = yunet.detect(image)
        assert len(detections) == 1, name
        face = detections[0]
        assert face.score > 0.8
        eyes, nose, mouth = face.landmarks[:2], face.landmarks[2], face.landmarks[3:]
        assert eyes[0, 0] < eyes[1, 0] and mouth[0, 0] < mouth[1, 0], name  # image-left first
        assert eyes[:, 1].max() < nose[1] < mouth[:, 1].min(), name


@pytest.mark.models
def test_aligned_portrait_faces_land_on_the_template(
    portraits: dict[str, np.ndarray], yunet: Any
) -> None:
    for name, image in portraits.items():
        face = yunet.detect(image)[0]
        aligned = align_face(image, face.landmarks)
        assert aligned is not None
        assert aligned.residual < 0.05, name
        metrics = face_metrics(
            model_input(aligned.crop),
            crop_size=aligned.size,
            detector_score=face.score,
            landmarks=face.landmarks,
            alignment_residual=aligned.residual,
        )
        assert grade_face(metrics, QualityConfig()).tier >= Tier.FAIR, (name, metrics)


def track_clip(path: Path, detector: Any) -> FaceTracker:
    tracker = FaceTracker()
    for sample in VideoReader(path, validate(path)):
        tracker.update(sample.time, detector.detect(sample.image))
    return tracker


@pytest.mark.models
def test_a_moving_face_in_video_is_one_persistent_track(
    portraits: dict[str, np.ndarray], yunet: Any, make_video: MakeVideo
) -> None:
    patch = face_patch(portraits["ellen_ochoa"], yunet)
    frames = animate([(patch, (220, 240), (60, 0))], seconds=4.0)
    tracker = track_clip(make_video("pan.mp4", frames=frames), yunet)
    spans = tracker.finalize()
    assert len(spans) == 1
    assert spans[0].detections >= 15  # 4 s at 4 Hz, allowing a few misses


@pytest.mark.models
@pytest.mark.parametrize(("occlusion", "expected"), [(1.5, 2), (3.0, 3)])
def test_two_faces_with_an_occlusion(
    portraits: dict[str, np.ndarray],
    yunet: Any,
    make_video: MakeVideo,
    occlusion: float,
    expected: int,
) -> None:
    left = face_patch(portraits["mae_jemison"], yunet, side=220)
    right = face_patch(portraits["neil_armstrong"], yunet, side=220)
    frames = animate(
        [(left, (170, 250), (10, 0)), (right, (470, 250), (-10, 0))],
        seconds=7.0,
        occlusions=[(1, 2.0, 2.0 + occlusion)],
    )
    tracker = track_clip(make_video("pair.mp4", frames=frames), yunet)
    assert len(tracker.finalize()) == expected
