from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from deepfake_core.config import PipelineConfig
from deepfake_core.detectors.base import FramePredictions
from deepfake_core.errors import ModelMissingError
from deepfake_core.faces import Detection
from deepfake_core.faces.align import TEMPLATE
from deepfake_core.pipeline import Analyzer
from deepfake_core.schemas import AnalysisResult
from tests.conftest import animate, face_patch

MakeVideo = Callable[..., Path]
VOLATILE = {"analysis_id", "processing"}


class StubFaces:
    """One fixed, well-sized face in every frame."""

    def __init__(self, box: tuple[float, float, float, float] = (60, 20, 260, 220)) -> None:
        x1, y1, x2, _ = box
        self.box = np.array(box, np.float32)
        self.landmarks = (TEMPLATE * (x2 - x1) + np.array([x1, y1], np.float32)).astype(np.float32)

    def detect(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        return [Detection(self.box.copy(), self.landmarks.copy(), 0.97)]


class NoFaces:
    def detect(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        return []


class ScriptedScores:
    """Returns pre-set scores in call order (one face per frame, so order = time)."""

    name = "gend"

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls = 0

    def predict(self, crops: Sequence[npt.NDArray[np.uint8]]) -> FramePredictions:
        out = np.array(self.scores[self.calls : self.calls + len(crops)], np.float32)
        self.calls += len(crops)
        embeddings = np.ones((len(crops), 8), np.float32) / np.sqrt(8)
        return FramePredictions(out, embeddings)


def analyzer(faces: Any, scores: Sequence[float]) -> Analyzer:
    return Analyzer(PipelineConfig(), face_detector=faces, frame_detector=ScriptedScores(scores))


def comparable(result: AnalysisResult) -> dict[str, Any]:
    data = json.loads(result.model_dump_json())
    for key in VOLATILE:
        data.pop(key)
    return data


def test_end_to_end_with_stub_detectors(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=150, fps=25, size=(320, 240))  # 6 s → 24 samples
    scores = [0.1] * 8 + [0.95] * 8 + [0.1] * 8  # suspicious from 2 s to 4 s
    output = analyzer(StubFaces(), scores).analyze(path)
    result = output.result

    assert result.face_tracks == 1
    assert result.processing.frames_sampled == 24
    assert result.manipulation_probability == pytest.approx(np.mean(scores), abs=1e-4)
    assert result.verdict == "inconclusive"
    [interval] = result.suspicious_intervals
    assert interval.start == pytest.approx(2.0, abs=0.05)
    assert interval.end == pytest.approx(4.0, abs=0.05)
    assert result.evidence and all(2.0 <= e.time <= 4.0 for e in result.evidence[:2])
    assert {e.evidence_id for e in result.evidence} == {img.evidence_id for img in output.evidence}
    assert result.detector_scores == {"gend": result.manipulation_probability}
    assert [m.role for m in result.models] == ["face_detector", "frame_detector"]


def test_consistently_high_scores_on_good_video_flag_manipulation(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=150, fps=25, size=(1280, 720))
    result = analyzer(StubFaces((400, 150, 700, 450)), [0.97] * 24).analyze(path).result
    assert result.video_quality == "good"
    assert (result.verdict, result.confidence) == ("manipulated_likely", "high")


def test_poor_video_needs_stronger_evidence(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=150, fps=25, size=(320, 240))  # 240 px: POOR
    result = analyzer(StubFaces(), [0.85] * 24).analyze(path).result
    assert result.video_quality == "poor"
    assert result.manipulation_probability == pytest.approx(0.85, abs=1e-4)  # never altered
    assert result.verdict == "inconclusive"
    assert result.verdict_reasons == ["poor_quality_needs_stronger_evidence"]


def test_no_faces_is_inconclusive(make_video: MakeVideo) -> None:
    path = make_video("clip.webm", frames=60, fps=20)
    result = analyzer(NoFaces(), []).analyze(path).result
    assert result.verdict == "inconclusive"
    assert result.manipulation_probability is None
    assert result.video_quality == "insufficient"
    assert result.face_tracks == 0
    assert "no_face" in {r.code for r in result.quality.reasons}
    assert result.verdict_reasons == ["no_face"]


def test_results_are_deterministic_and_never_contain_the_filename(make_video: MakeVideo) -> None:
    path = make_video("candidate-jane-doe-interview.mp4", frames=100, fps=25)
    scores = list(np.linspace(0.2, 0.6, 16))
    first = analyzer(StubFaces(), scores).analyze(path).result
    second = analyzer(StubFaces(), scores).analyze(path).result
    assert comparable(first) == comparable(second)
    assert "jane" not in first.model_dump_json().lower()


def test_trace_round_trip(make_video: MakeVideo, tmp_path: Path) -> None:
    path = make_video("clip.mp4", frames=100, fps=25)
    output = analyzer(StubFaces(), [0.3] * 16).analyze(path)
    target = tmp_path / "trace.npz"
    output.trace.save_npz(target)
    with np.load(target) as data:
        assert data["track0_times"].shape == (16,)
        assert data["track0_embeddings"].shape == (16, 8)
        np.testing.assert_allclose(data["track0_p_fake"], 0.3, atol=1e-6)


def test_progress_reports_stages_in_order(make_video: MakeVideo) -> None:
    path = make_video("clip.mp4", frames=100, fps=25)
    seen: list[str] = []
    analyzer(StubFaces(), [0.3] * 16).analyze(path, progress=lambda s, f: seen.append(s))
    stages = [s for i, s in enumerate(seen) if i == 0 or seen[i - 1] != s]
    assert stages == ["validating", "analyzing", "scoring", "done"]


@pytest.mark.models
def test_real_models_end_to_end_is_deterministic(
    portraits: dict[str, np.ndarray], yunet: Any, make_video: MakeVideo
) -> None:
    patch = face_patch(portraits["ellen_ochoa"], yunet, side=300)
    path = make_video("ochoa.mp4", frames=animate([(patch, (330, 260), (15, 0))], seconds=4.0))
    try:
        real = Analyzer()
    except ModelMissingError:
        pytest.skip("model weights missing: run `make models`")
    first, second = real.analyze(path), real.analyze(path)
    result = first.result
    assert result.face_tracks == 1 and result.tracks[0].eligible
    assert result.video_quality in {"good", "fair"}
    assert result.manipulation_probability is not None
    assert comparable(first.result) == comparable(second.result)
    assert first.evidence and first.evidence[0].jpeg[:2] == b"\xff\xd8"


def test_decision_thresholds_follow_the_quality_tier(make_video: MakeVideo) -> None:
    poor = make_video("clip.mp4", frames=100, fps=25, size=(320, 240))
    result = analyzer(StubFaces(), [0.5] * 16).analyze(poor).result
    assert result.video_quality == "poor"
    assert (result.decision_thresholds.authentic, result.decision_thresholds.manipulated) == (
        0.15,
        0.90,
    )
