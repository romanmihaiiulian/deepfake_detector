"""Did the swap actually happen? Compare the fake with its source, frame by frame.

A face swapper can silently miss a face, for example on profile views, small faces or
fast motion. A "fake" that is mostly the original video would be a mislabelled sample, so
each fake's coverage is measured: the fraction of sampled frames where the face region
changed clearly more than the background did. Fakes below `min_coverage` are rejected.
Both files are near-lossless intermediates, so unchanged regions differ by almost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

from deepfake_core.faces import FaceDetector


@dataclass(frozen=True, slots=True)
class Coverage:
    sampled_frames: int
    frames_with_face: int
    swapped_frames: int
    face_diff_median: float
    background_diff_median: float
    frame_count_delta: int

    @property
    def fraction(self) -> float:
        return self.swapped_frames / self.frames_with_face if self.frames_with_face else 0.0


def _frames(path: Path) -> av.container.InputContainer:
    return av.open(str(path))


def _shrink(
    box: npt.NDArray[np.float32], factor: float, shape: tuple[int, ...]
) -> tuple[slice, slice]:
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor, (y2 - y1) * factor
    height, width = shape[:2]
    rows = slice(int(max(0, cy - h / 2)), int(min(height, cy + h / 2)))
    cols = slice(int(max(0, cx - w / 2)), int(min(width, cx + w / 2)))
    return rows, cols


def measure_coverage(
    original: Path,
    fake: Path,
    detector: FaceDetector,
    *,
    sample_hz: float = 4.0,
    diff_threshold: float = 6.0,
) -> Coverage:
    with _frames(original) as a, _frames(fake) as b:
        stream_a, stream_b = a.streams.video[0], b.streams.video[0]
        fps = float(stream_a.average_rate or 25)
        step = max(1, round(fps / sample_hz))
        count_a = count_b = sampled = with_face = swapped = 0
        face_diffs: list[float] = []
        background_diffs: list[float] = []
        frames_a, frames_b = a.decode(stream_a), b.decode(stream_b)
        while True:
            frame_a, frame_b = next(frames_a, None), next(frames_b, None)
            count_a += frame_a is not None
            count_b += frame_b is not None
            if frame_a is None or frame_b is None:
                break
            if (count_a - 1) % step:
                continue
            sampled += 1
            image_a = np.asarray(frame_a.to_ndarray(format="rgb24"), dtype=np.uint8)
            image_b = np.asarray(frame_b.to_ndarray(format="rgb24"), dtype=np.uint8)
            if image_a.shape != image_b.shape:
                continue
            faces = detector.detect(image_a)
            if not faces:
                continue
            with_face += 1
            rows, cols = _shrink(faces[0].box, 0.8, image_a.shape)
            diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16)).mean(axis=2)
            face_diff = float(diff[rows, cols].mean()) if diff[rows, cols].size else 0.0
            mask = np.ones(diff.shape, bool)
            outer_rows, outer_cols = _shrink(faces[0].box, 1.6, image_a.shape)
            mask[outer_rows, outer_cols] = False
            background = float(diff[mask].mean()) if mask.any() else 0.0
            face_diffs.append(face_diff)
            background_diffs.append(background)
            if face_diff >= max(diff_threshold, 3 * background + 1):
                swapped += 1
        count_a += sum(1 for _ in frames_a)  # whatever is left when lengths differ
        count_b += sum(1 for _ in frames_b)
    return Coverage(
        sampled_frames=sampled,
        frames_with_face=with_face,
        swapped_frames=swapped,
        face_diff_median=float(np.median(face_diffs)) if face_diffs else 0.0,
        background_diff_median=float(np.median(background_diffs)) if background_diffs else 0.0,
        frame_count_delta=count_b - count_a,
    )
