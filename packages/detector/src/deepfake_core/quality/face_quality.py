"""Face-level quality metrics, measured on each aligned face crop.

The crop is the 224x224 image the detector actually sees. Size is measured before
resizing (the native crop side, in source pixels), because upsampling a small face makes
it look smooth rather than small.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from deepfake_core.config import QualityConfig
from deepfake_core.quality.gate import Grade, Tier, grade

FACE_METRICS = (
    "face_crop_size",
    "face_sharpness",
    "face_exposure",
    "face_clipped_fraction",
    "face_contrast",
    "face_detector_score",
    "face_yaw",
    "face_alignment_residual",
)


# The face itself inside an aligned crop: the template's eyes sit at about 0.37/0.63 of the
# width and its mouth at 0.78 of the height. Photometric metrics are measured here, so a
# bright wall or a white collar at the crop margin doesn't count as a badly lit face.
FACE_ROI = (slice(0.22, 0.86), slice(0.22, 0.78))


def face_region(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    rows, cols = FACE_ROI
    height, width = image.shape[:2]
    return image[
        int(rows.start * height) : int(rows.stop * height),
        int(cols.start * width) : int(cols.stop * width),
    ]


def luminance(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """BT.601 luma in [0, 255]."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.asarray(rgb, dtype=np.float32) @ weights


def sharpness(gray: npt.NDArray[np.float32]) -> float:
    """Variance of the Laplacian: high for sharp detail, near zero for blur."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def yaw_proxy(landmarks: npt.NDArray[np.float32]) -> float:
    """Nose offset from the eye midpoint, along the eye axis, in inter-ocular distances.

    About 0 for a frontal face and around 0.5 or more for a profile. Roll does not affect
    it, because the offset is projected onto the eye axis.
    """
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    axis = right_eye - left_eye
    distance = float(np.hypot(axis[0], axis[1]))
    if distance < 1e-6:
        return 1.0
    offset = nose - (left_eye + right_eye) / 2.0
    return abs(float(offset @ axis)) / (distance * distance)


@dataclass(frozen=True, slots=True)
class FaceQuality:
    tier: Tier
    grade: Grade


def face_metrics(
    crop: npt.NDArray[np.uint8],
    *,
    crop_size: float,
    detector_score: float,
    landmarks: npt.NDArray[np.float32],
    alignment_residual: float,
) -> dict[str, float | None]:
    gray = luminance(crop)
    face = face_region(gray)
    return {
        "face_crop_size": crop_size,
        "face_sharpness": sharpness(face),
        "face_exposure": float(face.mean()),
        "face_clipped_fraction": float(np.mean((face <= 5) | (face >= 250))),
        "face_contrast": float(face.std()),
        "face_detector_score": detector_score,
        "face_yaw": yaw_proxy(landmarks),
        "face_alignment_residual": alignment_residual,
    }


def grade_face(metrics: dict[str, float | None], config: QualityConfig) -> FaceQuality:
    bands = {name: getattr(config, name) for name in FACE_METRICS}
    result = grade(metrics, bands)
    return FaceQuality(result.tier, result)
