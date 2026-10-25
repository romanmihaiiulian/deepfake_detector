"""Temporal-model training and evaluation on synthetic traces.

In the synthetic data, real and fake tracks get frame scores from the same distribution,
so every frame-score aggregator is at chance. Fakes differ only in time: their embeddings
flicker from frame to frame, while real ones drift slowly. Only a model that looks at
consecutive frames can separate them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deepfake_bench import temporal
from deepfake_bench.cli import main
from deepfake_bench.lab import LabError, delete_identity
from deepfake_bench.manifest import Identity, Registry
from deepfake_bench.paths import LabPaths
from deepfake_core.config import AggregationConfig
from deepfake_core.models import GEND
from deepfake_core.pipeline import AnalysisTrace, TraceTrack
from deepfake_core.quality.gate import Tier

DIM = 16
FAST = temporal.TrainConfig(window=8, stride=4, hidden=32, layers=2, epochs=40, patience=10)


def synthetic_track(
    rng: np.random.Generator, fake: bool, frames: int = 32, track_id: int = 0
) -> TraceTrack:
    base = rng.normal(size=DIM)
    drift = np.cumsum(rng.normal(scale=0.05, size=(frames, DIM)), axis=0)
    jitter = rng.normal(scale=0.6 if fake else 0.05, size=(frames, DIM))
    emb = base + drift + jitter
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return TraceTrack(
        track_id=track_id,
        times=(np.arange(frames) * 0.25).astype(np.float32),
        p_fake=rng.uniform(0.3, 0.7, frames).astype(np.float32),
        tiers=np.full(frames, int(Tier.GOOD), np.int8),
        embeddings=emb.astype(np.float16),
    )


def synthetic_run(
    root: Path,
    pairs: dict[str, int] | None = None,
    *,
    seed: int = 0,
    extra: list[tuple[dict[str, Any], TraceTrack]] = (),  # type: ignore[assignment]
) -> Path:
    run_dir = LabPaths(root).runs / "traces"
    (run_dir / "traces").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    entries: list[tuple[dict[str, Any], TraceTrack]] = list(extra)
    for split, count in (pairs or {"train": 24, "val": 10, "test": 20}).items():
        for pair in range(count):
            for label in ("real", "manipulated"):
                fake = label == "manipulated"
                record = {
                    "sample_id": f"{split}-{label[:4]}{pair}",
                    "key": f"k-{split}-{label}{pair}",
                    "label": label,
                    "split": split,
                    "pair_id": f"{split}-video{pair}",
                    "transformation": "pristine",
                    "source_identity": f"{split}-donor{pair}" if fake else None,
                    "target_identity": f"{split}-person{pair}",
                    "error": None,
                }
                entries.append((record, synthetic_track(rng, fake)))
    lines = []
    for record, track in entries:
        AnalysisTrace(4.0, (track,)).save_npz(run_dir / "traces" / f"{record['sample_id']}.npz")
        lines.append(json.dumps({**record, "score": float(np.nanmean(track.p_fake))}) + "\n")
    (run_dir / "predictions.jsonl").write_text("".join(lines))
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "traces",
                "pipeline_version": "test",
                "config_hash": "c" * 16,
                "models": [
                    {"name": GEND.name, "role": "frame_detector", "sha256": GEND.sha256},
                ],
            }
        )
    )
    return run_dir


def test_trace_round_trip_and_eligibility(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    good = synthetic_track(rng, fake=False, track_id=3)
    short = synthetic_track(rng, fake=False, frames=5, track_id=4)
    poor = synthetic_track(rng, fake=False, track_id=7)
    poor = TraceTrack(
        7, poor.times, poor.p_fake, np.full(32, int(Tier.INSUFFICIENT)), poor.embeddings
    )
    path = tmp_path / "t.npz"
    AnalysisTrace(4.0, (good, short, poor)).save_npz(path)
    loaded = temporal.load_trace(path)
    assert [t.track_id for t in loaded] == [3, 4, 7]
    np.testing.assert_array_equal(loaded[0].embeddings, good.embeddings)
    config = AggregationConfig()
    assert [temporal.is_eligible(t, config) for t in loaded] == [True, False, False]


def test_training_learns_temporal_cues_the_baselines_miss(tmp_path: Path) -> None:
    root = tmp_path / "data"
    run_dir = synthetic_run(root)
    result = temporal.train(run_dir, LabPaths(root).temporal, FAST, log=lambda _: None)
    assert result.checkpoint.is_file() and result.record.is_file()
    assert result.best_epoch > 0
    assert result.val_auc is not None and result.val_auc >= 0.85
    assert result.baseline_val_auc is not None and result.baseline_val_auc < 0.8

    record = json.loads(result.record.read_text())
    assert len(record["train_samples"]) == 48 and len(record["val_samples"]) == 20
    assert all(not i.startswith("test-") for i in record["identities"])
    assert record["history"][0]["epoch"] == 0

    report = temporal.evaluate(run_dir, result.checkpoint, bootstrap=200)
    test = report["splits"]["test"]
    assert set(report["splits"]) == {"val", "test"}
    assert test["aggregators"]["temporal"]["roc_auc"] >= 0.85
    assert test["aggregators"]["mean"]["roc_auc"] < 0.8
    assert test["auc_delta_ci95"] is not None and test["auc_delta_ci95"][0] > 0
    assert test["auc_by_transformation"]["pristine"]["temporal"] >= 0.85
    out = run_dir / "temporal" / result.name
    markdown = (out / "report.md").read_text()
    assert "AUC(temporal) - AUC(mean)" in markdown and "| temporal |" in markdown
    assert json.loads((out / "report.json").read_text())["excluded_for_overlap"] == 0


def test_training_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "data"
    run_dir = synthetic_run(root, {"train": 6, "val": 3})
    config = temporal.TrainConfig(window=8, stride=4, hidden=16, layers=1, epochs=3, patience=5)
    first = temporal.train(run_dir, tmp_path / "a", config, log=lambda _: None)
    second = temporal.train(run_dir, tmp_path / "b", config, log=lambda _: None)
    assert first.history == second.history
    assert first.name == second.name == "tcn-traces-s0"


def test_training_needs_train_and_val_traces(tmp_path: Path) -> None:
    root = tmp_path / "data"
    with pytest.raises(LabError, match="--split train"):
        temporal.train(synthetic_run(root, {"val": 3, "test": 3}), tmp_path / "out", FAST)
    with pytest.raises(LabError, match="no val-split traces"):
        temporal.train(synthetic_run(tmp_path / "other", {"train": 3}), tmp_path / "out", FAST)


def test_evaluation_excludes_samples_that_share_an_identity_with_training(tmp_path: Path) -> None:
    root = tmp_path / "data"
    rng = np.random.default_rng(5)
    leaked = {
        "sample_id": "test-leak",
        "key": "k-leak",
        "label": "real",
        "split": "test",
        "pair_id": "test-leak",
        "transformation": "pristine",
        "source_identity": None,
        "target_identity": "train-person0",
        "error": None,
    }
    run_dir = synthetic_run(
        root, {"train": 4, "val": 2, "test": 2}, extra=[(leaked, synthetic_track(rng, False))]
    )
    config = temporal.TrainConfig(window=8, stride=4, hidden=16, layers=1, epochs=1)
    result = temporal.train(run_dir, LabPaths(root).temporal, config, log=lambda _: None)
    report = temporal.evaluate(run_dir, result.checkpoint, bootstrap=0)
    assert report["excluded_for_overlap"] == 1
    assert report["splits"]["test"]["aggregators"]["mean"]["n_real"] == 2


def test_evaluation_rejects_a_run_from_another_frame_model(tmp_path: Path) -> None:
    root = tmp_path / "data"
    run_dir = synthetic_run(root, {"train": 3, "val": 2})
    config = temporal.TrainConfig(window=8, stride=4, hidden=16, layers=1, epochs=1)
    result = temporal.train(run_dir, LabPaths(root).temporal, config, log=lambda _: None)
    run = json.loads((run_dir / "run.json").read_text())
    run["models"][0]["sha256"] = "f" * 64
    (run_dir / "run.json").write_text(json.dumps(run))
    with pytest.raises(LabError, match="different frame model"):
        temporal.evaluate(run_dir, result.checkpoint, bootstrap=0)


def test_withdrawing_consent_deletes_temporal_models(tmp_path: Path) -> None:
    root = tmp_path / "data"
    registry = Registry(LabPaths(root))
    registry.identities["alice"] = Identity(identity_id="alice", consent_ref="form-1")
    registry.save()
    models = registry.paths.temporal
    models.mkdir(parents=True)
    for name, identities in (("with-alice", ["alice", "bob"]), ("without", ["bob"])):
        (models / f"{name}.safetensors").write_bytes(b"weights")
        (models / f"{name}.json").write_text(json.dumps({"identities": identities}))
    report = delete_identity(registry, "alice")
    assert report.temporal_models == 1
    assert sorted(p.name for p in models.iterdir()) == ["without.json", "without.safetensors"]


def test_cli_train_and_eval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"
    synthetic_run(root, {"train": 4, "val": 2, "test": 2})
    base = ["--data-root", str(root), "temporal"]
    assert main([*base, "train", "traces", "--window", "8", "--epochs", "2", "--name", "m"]) == 0
    assert (root / "bench" / "temporal" / "m.safetensors").is_file()
    assert main([*base, "eval", "traces", "--checkpoint", "m", "--bootstrap", "0"]) == 0
    assert (root / "bench" / "runs" / "traces" / "temporal" / "m" / "report.md").is_file()
    assert main([*base, "eval", "traces", "--checkpoint", "nope"]) == 1
    assert "no temporal checkpoint" in capsys.readouterr().err
