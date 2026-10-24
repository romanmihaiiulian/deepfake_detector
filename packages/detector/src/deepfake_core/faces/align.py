# Face alignment ported from GenD (https://github.com/yermandy/GenD), detector.py:align_face,
# commit 387a42266dd385fbe8f7626c5d3aa03eea3bfaab. Changes: typed API, returns the transform,
# its residual and the native crop size; no mask handling.
#
# MIT License
#
# Copyright (c) 2025 Andy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""5-point similarity alignment to GenD's face template.

GenD crops at the face's native resolution, with a 1.3x scale margin, and CLIP's processor
then resizes the crop to 224 px with bicubic interpolation. We reproduce both steps so that
the detector sees crops geometrically equivalent to its training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from deepfake_core.faces.types import Landmarks

TEMPLATE = np.array(
    [[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32
)
MIN_CROP = 8

_UPPER = np.triu_indices(5, k=1)
_TEMPLATE_DISTANCES = np.linalg.norm(TEMPLATE[:, None, :] - TEMPLATE[None, :, :], axis=-1)[_UPPER]


@dataclass(frozen=True, slots=True)
class AlignedFace:
    crop: npt.NDArray[np.uint8]
    """RGB square crop at native resolution (`size` x `size`)."""
    size: int
    matrix: npt.NDArray[np.float64]
    """2x3 similarity transform from source pixels to crop pixels."""
    residual: float
    """Mean distance between transformed landmarks and template points, / size."""


def native_crop_size(landmarks: Landmarks, scale: float = 1.3) -> int:
    """GenD's `target_size=None` rule: the face's scale relative to the template, times
    the margin factor."""
    distances = np.linalg.norm(landmarks[:, None, :] - landmarks[None, :, :], axis=-1)[_UPPER]
    approx = int(np.round(np.mean(distances / _TEMPLATE_DISTANCES) * scale))
    return max(approx, MIN_CROP)


def template_points(size: int, scale: float = 1.3) -> npt.NDArray[np.float32]:
    """Template landmark positions inside a `size` x `size` crop with margin."""
    dst = TEMPLATE.copy() * size
    margin = size * (scale - 1) / 2.0
    dst += margin
    dst *= size / (size + 2 * margin)
    return dst.astype(np.float32)


def align_face(
    image: npt.NDArray[np.uint8],
    landmarks: Landmarks,
    *,
    scale: float = 1.3,
    size: int | None = None,
) -> AlignedFace | None:
    """Warp the face to the template. Returns None if no transform can be estimated."""
    side = size if size is not None else native_crop_size(landmarks, scale)
    dst = template_points(side, scale)
    src = np.asarray(landmarks, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    crop = cv2.warpAffine(image, matrix, (side, side), flags=cv2.INTER_LINEAR)
    mapped = src @ matrix[:, :2].T + matrix[:, 2]
    residual = float(np.linalg.norm(mapped - dst, axis=1).mean() / side)
    return AlignedFace(
        np.asarray(crop, dtype=np.uint8), side, np.asarray(matrix, dtype=np.float64), residual
    )


def model_input(crop: npt.NDArray[np.uint8], size: int = 224) -> npt.NDArray[np.uint8]:
    """Resize a square crop the way CLIPImageProcessor does: PIL bicubic to `size`."""
    if crop.shape[0] == size and crop.shape[1] == size:
        return crop
    resized = Image.fromarray(crop).resize((size, size), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)
