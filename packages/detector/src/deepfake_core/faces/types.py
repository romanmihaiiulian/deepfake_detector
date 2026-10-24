"""Face detections and boxes, in source-frame pixel coordinates (display orientation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Box = npt.NDArray[np.float32]
"""(x1, y1, x2, y2)."""

Landmarks = npt.NDArray[np.float32]
"""(5, 2) in image-space order: image-left eye, image-right eye, nose tip,
image-left mouth corner, image-right mouth corner. This is the order of GenD's
alignment template."""


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    landmarks: Landmarks
    score: float

    @property
    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    if inter <= 0.0:
        return 0.0
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0


def box_center(box: Box) -> npt.NDArray[np.float32]:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)


def box_size(box: Box) -> float:
    """Geometric mean of width and height."""
    return float(np.sqrt(max(1e-6, float(box[2] - box[0]) * float(box[3] - box[1]))))
