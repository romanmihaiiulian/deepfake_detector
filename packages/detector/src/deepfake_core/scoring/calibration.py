"""Calibration (roadmap step 16): raw detector scores → P(manipulated).

- `platt`: logistic regression on the logits of the inputs. With `gend` and `temporal`
  as inputs, this also fuses the two detectors.
- `isotonic`: a monotone step function of one input, interpolated linearly.

Both are fitted by `dfbench calibrate` on val with classes weighted equally, so the
fitted probability assumes a 50% prior. That is deliberate: the lab's ratio of fakes to
reals (several swappers and variants per real video) says nothing about deployment.
`prior` then shifts the log-odds to the base rate an operator expects.

Calibration is applied per track, and the video probability is the maximum over eligible
tracks. Frame scores, suspicious intervals and evidence stay on the raw GenD scale.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from deepfake_core.config import CalibrationConfig

_EPS = 1e-4


def logit(p: float) -> float:
    q = min(max(p, _EPS), 1 - _EPS)
    return float(np.log(q) - np.log1p(-q))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def calibrate(scores: Mapping[str, float | None], config: CalibrationConfig) -> float | None:
    """Calibrated probability from raw scores keyed by input name; None when an input is
    missing. With method `none`, the first input's raw score is returned unchanged."""
    values = [scores.get(name) for name in config.inputs]
    if any(v is None for v in values):
        return None
    raw = [float(v) for v in values if v is not None]
    if config.method == "none":
        return raw[0]
    if config.method == "platt":
        log_odds = config.intercept + sum(
            c * logit(v) for c, v in zip(config.coefficients, raw, strict=True)
        )
    else:
        log_odds = logit(float(np.interp(raw[0], config.isotonic_x, config.isotonic_y)))
    return sigmoid(log_odds + logit(config.prior))
