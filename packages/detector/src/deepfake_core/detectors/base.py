"""Frame-level detector interface shared by GenD and future detectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class FramePredictions:
    p_fake: npt.NDArray[np.float32]
    """(N,) manipulation probability per face crop (uncalibrated)."""
    embeddings: npt.NDArray[np.float32]
    """(N, D) L2-normalised features, input for the temporal model (step 14)."""


class FrameDetector(Protocol):
    name: str
    """Key used in `detector_scores`."""

    def predict(self, crops: Sequence[npt.NDArray[np.uint8]]) -> FramePredictions:
        """Score aligned RGB face crops (any square size; resized internally)."""
        ...
