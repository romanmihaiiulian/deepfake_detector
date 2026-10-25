from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deepfake_core.config import CalibrationConfig, PipelineConfig
from deepfake_core.errors import ModelIntegrityError
from deepfake_core.models import GEND
from deepfake_core.pipeline import Analyzer
from deepfake_core.schemas import UNCALIBRATED_NOTE
from deepfake_core.scoring.calibration import calibrate, logit, sigmoid
from tests.core.test_pipeline import ScriptedScores, StubFaces
from tests.core.test_temporal import write_checkpoint

MakeVideo = Callable[..., Path]

PLATT: dict[str, Any] = {
    "method": "platt",
    "inputs": ["gend"],
    "intercept": -2.0,
    "coefficients": [0.5],
    "model_sha256": [GEND.sha256],
    "fitted_on": "run r1, val: 40 real / 40 manipulated",
}


def test_identity_platt_isotonic_and_prior() -> None:
    assert calibrate({"gend": 0.42}, CalibrationConfig()) == 0.42
    platt = CalibrationConfig.model_validate(PLATT)
    assert calibrate({"gend": 0.9}, platt) == pytest.approx(sigmoid(-2.0 + 0.5 * logit(0.9)))
    assert calibrate({"gend": None}, platt) is None
    assert calibrate({}, platt) is None

    rare = platt.model_copy(update={"prior": 0.01})
    shifted = sigmoid(-2.0 + 0.5 * logit(0.9) + logit(0.01))
    assert calibrate({"gend": 0.9}, rare) == pytest.approx(shifted)

    iso = CalibrationConfig(
        method="isotonic", isotonic_x=(0.2, 0.6, 0.9), isotonic_y=(0.1, 0.3, 0.95)
    )
    assert calibrate({"gend": 0.4}, iso) == pytest.approx(0.2, abs=1e-6)
    assert calibrate({"gend": 0.0}, iso) == pytest.approx(0.1, abs=1e-6)  # clipped
    assert calibrate({"gend": 1.0}, iso) == pytest.approx(0.95, abs=1e-6)

    fused = CalibrationConfig(
        method="platt", inputs=("gend", "temporal"), intercept=0.1, coefficients=(0.3, 1.2)
    )
    expected = sigmoid(0.1 + 0.3 * logit(0.7) + 1.2 * logit(0.2))
    assert calibrate({"gend": 0.7, "temporal": 0.2}, fused) == pytest.approx(expected)
    assert calibrate({"gend": 0.7, "temporal": None}, fused) is None


def test_platt_is_monotone_for_a_positive_coefficient() -> None:
    platt = CalibrationConfig.model_validate(PLATT)
    values = [calibrate({"gend": p}, platt) for p in np.linspace(0, 1, 101)]
    assert all(b >= a for a, b in pairwise(values))  # type: ignore[operator]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"method": "platt", "coefficients": [1.0, 2.0]}, "one coefficient per input"),
        ({"method": "isotonic", "isotonic_x": [0.1], "isotonic_y": [0.2]}, "breakpoints"),
        (
            {"method": "isotonic", "isotonic_x": [0.1, 0.2], "isotonic_y": [0.5, 0.4]},
            "non-decreasing",
        ),
        (
            {
                "method": "isotonic",
                "inputs": ["gend", "temporal"],
                "isotonic_x": [0, 1],
                "isotonic_y": [0, 1],
            },
            "exactly one input",
        ),
        ({"inputs": []}, "non-empty"),
        ({"prior": 1.0}, "less than 1"),
    ],
)
def test_invalid_calibration_configs(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CalibrationConfig.model_validate(data)


def test_temporal_input_needs_a_temporal_model() -> None:
    with pytest.raises(ValueError, match=r"needs temporal\.checkpoint"):
        PipelineConfig.model_validate(
            {"calibration": {**PLATT, "inputs": ["gend", "temporal"], "coefficients": [1, 1]}}
        )


def calibrated_analyzer(calibration: dict[str, Any], scores: list[float], **extra: Any) -> Analyzer:
    config = PipelineConfig.model_validate({"calibration": calibration, **extra})
    return Analyzer(config, face_detector=StubFaces(), frame_detector=ScriptedScores(scores))


def test_pipeline_applies_calibration(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=150, fps=25, size=(1280, 720))  # 24 samples, good
    scores = [0.9] * 24
    verdict = {"manipulated_threshold": 0.25, "authentic_threshold": 0.05}
    result = calibrated_analyzer(PLATT, scores, verdict=verdict).analyze(path).result
    expected = sigmoid(-2.0 + 0.5 * logit(0.9))
    assert result.manipulation_probability == pytest.approx(expected, abs=1e-4)
    assert result.detector_scores == {"gend": pytest.approx(0.9)}  # raw scores stay raw
    assert result.tracks[0].score == pytest.approx(expected, abs=1e-4)
    assert result.tracks[0].timeline[0].score == pytest.approx(0.9)
    calibration = result.calibration
    assert (calibration.method, calibration.inputs, calibration.prior) == ("platt", ["gend"], 0.5)
    assert calibration.raw_probability == pytest.approx(0.9)
    assert calibration.fitted_on == PLATT["fitted_on"]
    assert "P(manipulated) = 0.5" in calibration.note and "run r1" in calibration.note
    assert result.decision_thresholds.manipulated == 0.25
    assert result.verdict == "manipulated_likely"  # 0.29 >= 0.25: the calibrated value decides


def test_uncalibrated_results_say_so(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=150, fps=25, size=(320, 240))
    result = calibrated_analyzer({}, [0.5] * 24).analyze(path).result
    assert result.calibration.method == "none" and result.calibration.note == UNCALIBRATED_NOTE
    assert result.calibration.raw_probability is None


def test_pipeline_fuses_gend_and_temporal(tmp_path: Path, make_video: MakeVideo) -> None:
    checkpoint = write_checkpoint(tmp_path / "t.safetensors")
    temporal_sha = (
        Analyzer(
            PipelineConfig.model_validate({"temporal": {"checkpoint": str(checkpoint)}}),
            face_detector=StubFaces(),
            frame_detector=ScriptedScores([]),
        )
        .models[-1]
        .sha256
    )
    calibration = {
        "method": "platt",
        "inputs": ["gend", "temporal"],
        "intercept": 0.2,
        "coefficients": [0.4, 0.8],
        "model_sha256": [GEND.sha256, temporal_sha],
    }
    path = make_video("clip.mp4", frames=150, fps=25, size=(320, 240))
    result = (
        calibrated_analyzer(calibration, [0.7] * 24, temporal={"checkpoint": str(checkpoint)})
        .analyze(path)
        .result
    )
    gend, temporal = result.detector_scores["gend"], result.detector_scores["temporal"]
    expected = sigmoid(0.2 + 0.4 * logit(gend) + 0.8 * logit(temporal))
    assert result.manipulation_probability == pytest.approx(expected, abs=1e-4)


def test_pipeline_rejects_calibration_for_other_models() -> None:
    with pytest.raises(ModelIntegrityError, match="not loaded"):
        calibrated_analyzer({**PLATT, "model_sha256": ["f" * 64]}, [])
    with pytest.raises(ModelIntegrityError, match="not loaded"):
        calibrated_analyzer({**PLATT, "model_sha256": []}, [])
