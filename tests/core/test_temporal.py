from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from deepfake_core.config import PipelineConfig
from deepfake_core.errors import ModelIntegrityError, ModelMissingError
from deepfake_core.models import GEND
from deepfake_core.pipeline import Analyzer
from deepfake_core.temporal import (
    CheckpointMeta,
    TemporalHParams,
    TemporalNet,
    TemporalScorer,
    make_windows,
    save_checkpoint,
    score_sequence,
    sequence_features,
    window_starts,
)
from tests.conftest import animate, face_patch
from tests.core.test_pipeline import ScriptedScores, StubFaces

MakeVideo = Callable[..., Path]
DIM = 8


def meta(hparams: TemporalHParams, sha: str = GEND.sha256) -> CheckpointMeta:
    return CheckpointMeta(
        hparams=hparams,
        window=8,
        stride=4,
        embedding_model_sha256=sha,
        name="unit",
        trained_at="2026-10-25T00:00:00+00:00",
        training_set_sha256="0" * 64,
    )


def write_checkpoint(path: Path, *, arch: str = "tcn", sha: str = GEND.sha256) -> Path:
    torch.manual_seed(0)
    hparams = TemporalHParams(embedding_dim=DIM, arch=arch, hidden=16, layers=2)  # type: ignore[arg-type]
    save_checkpoint(TemporalNet(hparams), meta(hparams, sha), path)
    return path


def test_window_starts_cover_every_row() -> None:
    assert window_starts(5, 8, 4) == [0]
    assert window_starts(16, 8, 4) == [0, 4, 8]
    assert window_starts(18, 8, 4) == [0, 4, 8, 10]
    for n in range(1, 40):
        starts = window_starts(n, 8, 3)
        covered = {i for s in starts for i in range(s, min(s + 8, n))}
        assert covered == set(range(n))


def test_sequence_features_drop_unscored_frames_and_sort() -> None:
    times = np.array([0.5, 0.0, 0.25, 3.5], np.float32)
    p = np.array([0.5, 0.9, np.nan, 0.1], np.float32)
    emb = np.eye(4, DIM, dtype=np.float32)
    t, features = sequence_features(times, p, emb)
    assert t.tolist() == [0.0, 0.5, 3.5]
    assert features.shape == (3, DIM + 4)
    logits = np.log([9.0, 1.0, 1 / 9])
    np.testing.assert_allclose(features[:, DIM], logits, rtol=1e-4)
    assert features[:, DIM + 1].tolist() == [0.0, 0.5, 2.0]  # gap clipped at 2 s
    assert features[:, DIM + 2].tolist() == [0.0, 1.0, 1.0]  # orthogonal neighbours
    np.testing.assert_allclose(features[:, DIM + 3], [0.0, np.log(9), np.log(9)], rtol=1e-4)
    np.testing.assert_array_equal(features[0, :DIM], emb[1])

    windows, mask, starts = make_windows(features, 8, 4)
    assert windows.shape == (1, 8, DIM + 4) and mask.sum() == 3 and starts == [0]


@pytest.mark.parametrize("arch", ["tcn", "transformer"])
def test_padding_is_exact(arch: str) -> None:
    torch.manual_seed(1)
    net = TemporalNet(TemporalHParams(embedding_dim=DIM, arch=arch, hidden=16, layers=2))  # type: ignore[arg-type]
    for parameter in net.head.parameters():  # a non-trivial head, so the encoder matters
        torch.nn.init.normal_(parameter)
    net.eval()
    rows = torch.randn(1, 5, DIM + 4)
    padded = torch.cat([rows, torch.randn(1, 3, DIM + 4)], dim=1)
    mask = torch.tensor([[True] * 5 + [False] * 3])
    with torch.inference_mode():
        exact = net(rows, torch.ones(1, 5, dtype=torch.bool))
        with_pad = net(padded, mask)
    torch.testing.assert_close(exact, with_pad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("arch", ["tcn", "transformer"])
def test_untrained_network_reproduces_the_gend_mean_logit(arch: str) -> None:
    net = TemporalNet(TemporalHParams(embedding_dim=DIM, arch=arch, hidden=16)).eval()  # type: ignore[arg-type]
    p = np.array([0.2, 0.6, 0.9, 0.7, 0.4], np.float32)
    _, features = sequence_features(np.arange(5) * 0.25, p, np.ones((5, DIM), np.float32))
    scored = score_sequence(net, 8, 4, np.arange(5) * 0.25, p, np.ones((5, DIM), np.float32))
    expected = 1 / (1 + np.exp(-features[:, DIM].mean()))
    assert scored is not None and scored.score == pytest.approx(expected, abs=1e-5)
    assert scored.windows == ((0.0, 1.0, pytest.approx(expected, abs=1e-5)),)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path / "t.safetensors", arch="transformer")
    scorer = TemporalScorer(path)
    assert scorer.meta.window == 8 and scorer.meta.hparams.arch == "transformer"
    assert scorer.embedding_model_sha256 == GEND.sha256
    assert scorer.info.role == "temporal" and len(scorer.info.sha256) == 64
    rng = np.random.default_rng(0)
    times, p, emb = np.arange(30) * 0.25, rng.random(30), rng.normal(size=(30, DIM))
    p[3] = np.nan
    first, second = scorer.score(times, p, emb), TemporalScorer(path).score(times, p, emb)
    assert first is not None and first == second
    assert len(first.windows) == 7  # 29 scored rows: starts 0, 4, ..., 20, 21
    assert scorer.score(times[:2], [np.nan, np.nan], emb[:2]) is None
    with pytest.raises(ModelIntegrityError, match="dimensions"):
        scorer.score(times, p, emb[:, :4])


def test_bad_checkpoints_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelMissingError):
        TemporalScorer(tmp_path / "missing.safetensors")
    foreign = tmp_path / "foreign.safetensors"
    save_file({"w": torch.zeros(2)}, str(foreign), metadata={"format": "something-else"})
    with pytest.raises(ModelIntegrityError, match="not a temporal-model checkpoint"):
        TemporalScorer(foreign)
    hparams = TemporalHParams(embedding_dim=DIM, hidden=16)
    mismatched = tmp_path / "mismatched.safetensors"
    save_file({"w": torch.zeros(2)}, str(mismatched), metadata=meta(hparams).to_strings())
    with pytest.raises(ModelIntegrityError, match="does not match"):
        TemporalScorer(mismatched)


def test_config_requires_a_checkpoint_for_temporal_track_score() -> None:
    with pytest.raises(ValueError, match=r"temporal\.checkpoint"):
        PipelineConfig.model_validate({"aggregation": {"track_score": "temporal"}})
    assert PipelineConfig().temporal.checkpoint is None


def temporal_analyzer(checkpoint: Path, scores: list[float], track_score: str) -> Analyzer:
    config = PipelineConfig.model_validate(
        {"temporal": {"checkpoint": str(checkpoint)}, "aggregation": {"track_score": track_score}}
    )
    return Analyzer(config, face_detector=StubFaces(), frame_detector=ScriptedScores(scores))


def test_pipeline_reports_the_temporal_score(tmp_path: Path, make_video: MakeVideo) -> None:
    checkpoint = write_checkpoint(tmp_path / "t.safetensors")
    path = make_video("clip.mp4", frames=150, fps=25, size=(320, 240))  # 24 samples
    result = temporal_analyzer(checkpoint, [0.7] * 24, "mean").analyze(path).result
    # an untrained network reproduces the mean logit, so a constant 0.7 stays 0.7
    assert result.detector_scores == {"gend": pytest.approx(0.7), "temporal": pytest.approx(0.7)}
    assert result.manipulation_probability == pytest.approx(0.7)
    assert [m.role for m in result.models] == ["face_detector", "frame_detector", "temporal"]

    scores = [0.2] * 12 + [0.9] * 12
    result = temporal_analyzer(checkpoint, scores, "temporal").analyze(path).result
    assert result.manipulation_probability == result.detector_scores["temporal"]
    assert result.detector_scores["gend"] == pytest.approx(np.mean(scores), abs=1e-4)
    assert result.tracks[0].score == result.detector_scores["temporal"]


def test_pipeline_rejects_a_checkpoint_from_another_frame_model(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "t.safetensors", sha="f" * 64)
    with pytest.raises(ModelIntegrityError, match="different frame model"):
        temporal_analyzer(checkpoint, [], "mean")


@pytest.mark.models
def test_real_models_with_an_untrained_temporal_model(
    tmp_path: Path, portraits: dict[str, np.ndarray], yunet: Any, make_video: MakeVideo
) -> None:
    patch = face_patch(portraits["ellen_ochoa"], yunet, side=300)
    path = make_video("ochoa.mp4", frames=animate([(patch, (330, 260), (15, 0))], seconds=4.0))
    hparams = TemporalHParams(embedding_dim=1024)  # GenD's embedding size
    checkpoint = tmp_path / "t.safetensors"
    save_checkpoint(TemporalNet(hparams), meta(hparams), checkpoint)
    config = PipelineConfig.model_validate({"temporal": {"checkpoint": str(checkpoint)}})
    try:
        real = Analyzer(config)
    except ModelMissingError:
        pytest.skip("model weights missing: run `make models`")
    output = real.analyze(path)
    [track] = output.trace.tracks
    _, features = sequence_features(track.times, track.p_fake, track.embeddings)
    windows, mask, _ = make_windows(features, 8, 4)
    mean_logits = (windows[..., 1024] * mask).sum(axis=1) / mask.sum(axis=1)
    expected = float(np.mean(1 / (1 + np.exp(-mean_logits))))  # untrained: GenD mean logit
    assert output.result.detector_scores["temporal"] == pytest.approx(expected, abs=1e-3)
    assert output.result.models[-1].role == "temporal"
