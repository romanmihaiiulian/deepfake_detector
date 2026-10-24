"""Donor stills: one frontal, sharp, well-sized frame of each donor identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from deepfake_core.config import SamplingConfig
from deepfake_core.faces import FaceDetector
from deepfake_core.quality.face_quality import yaw_proxy
from deepfake_core.video import VideoReader, validate


@dataclass(frozen=True, slots=True)
class DonorStill:
    path: Path
    time: float
    score: float


def pick_donor_still(
    video: Path, dest: Path, detector: FaceDetector, sample_hz: float = 2.0
) -> DonorStill:
    """Choose the frame whose face is the most confident, frontal and large; save it as PNG."""
    info = validate(video)
    best: tuple[float, float, Image.Image] | None = None
    for sample in VideoReader(video, info, SamplingConfig(rate_hz=sample_hz, max_samples=240)):
        faces = detector.detect(sample.image)
        if len(faces) != 1:
            continue  # donors must show exactly one face
        face = faces[0]
        size = min(1.0, (face.box[2] - face.box[0]) / 200.0)
        frontal = max(0.0, 1.0 - 2.0 * yaw_proxy(face.landmarks))
        score = face.score * frontal * size
        if best is None or score > best[0]:
            best = (score, sample.time, Image.fromarray(sample.image))
    if best is None:
        raise ValueError(f"no frame with exactly one face in {video.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    best[2].save(dest, format="PNG")
    return DonorStill(dest, best[1], best[0])
