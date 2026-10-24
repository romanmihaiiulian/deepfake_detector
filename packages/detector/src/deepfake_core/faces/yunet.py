"""YuNet face detector (OpenCV Zoo, MIT) behind the `FaceDetector` protocol.

YuNet replaces the InsightFace SCRFD detector used in GenD's preprocessing, because
InsightFace's pretrained models are non-commercial (ADR 0002). It returns the same 5
landmarks in the same image-space order.
"""

from __future__ import annotations

import os
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from deepfake_core.config import DetectionConfig
from deepfake_core.faces.types import Detection
from deepfake_core.models import YUNET, require


class FaceDetector(Protocol):
    def detect(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        """Faces in an RGB image, largest first."""
        ...


class YuNetDetector:
    def __init__(
        self,
        config: DetectionConfig | None = None,
        models_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        path = require(YUNET, models_dir)
        self._net = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            self.config.score_threshold,
            self.config.nms_threshold,
            5000,
        )
        self._input_size: tuple[int, int] | None = None

    def detect(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        height, width = image.shape[:2]
        scale = min(1.0, self.config.input_long_side / max(height, width))
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            small = image
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size
        _, raw = self._net.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        faces: object = raw  # OpenCV returns None when no face is found (stubs say otherwise)
        if faces is None:
            return []
        detections = []
        for row in np.asarray(faces, dtype=np.float32):
            x, y, w, h = row[:4] / scale
            box = np.array([x, y, x + w, y + h], np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            landmarks = (row[4:14].reshape(5, 2) / scale).astype(np.float32)
            detections.append(Detection(box, landmarks, float(row[14])))
        detections.sort(key=lambda d: (-d.area, float(d.box[0]), float(d.box[1])))
        return detections[: self.config.max_faces]
