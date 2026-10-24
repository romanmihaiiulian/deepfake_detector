from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from deepfake_bench import metrics as m
from deepfake_bench.manifest import Registry, Sample
from deepfake_bench.media import sha256_of
from deepfake_bench.paths import LabPaths
from deepfake_bench.report import build_report, evaluate, recommend_thresholds
from deepfake_bench.run import Selection, load_predictions, run_benchmark
from deepfake_bench.slices import build_slices, resolution_bucket
from deepfake_core.config import PipelineConfig
from deepfake_core.detectors.base import FramePredictions
from deepfake_core.pipeline import AnalysisOutput, Analyzer
from tests.bench.test_lab import StubFaces
from tests.conftest import barcode_frame, write_video

# --------------------------------------------------------------------------- metrics


def test_auc_and_average_precision_match_sklearn_with_ties() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        labels = rng.integers(0, 2, 60).astype(float)
        if labels.min() == labels.max():
            continue
        scores = np.round(rng.random(60), 1)  # lots of ties
        assert m.roc_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))
        assert m.average_precision(labels, scores) == pytest.approx(
            average_precision_score(labels, scores)
        )
    assert m.roc_auc(np.zeros(3), np.ones(3)) is None


def test_tpr_at_fpr() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1], float)
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    assert m.tpr_at_fpr(labels, scores, 0.0) == (0.5, 0.95)
    assert m.tpr_at_fpr(labels, scores, 0.25) == (1.0, 0.8)


def test_confusion_and_rates() -> None:
    labels = np.array([0, 0, 1, 1], float)
    conf = m.confusion(labels, np.array([0.1, 0.9, 0.2, 0.95]), 0.5)
    assert (conf.tp, conf.fp, conf.tn, conf.fn) == (1, 1, 1, 1)
    assert (conf.precision, conf.recall, conf.fpr, conf.fnr, conf.f1) == (0.5, 0.5, 0.5, 0.5, 0.5)


def test_calibration_error() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random(20000)
    calibrated = (rng.random(20000) < scores).astype(float)
    assert m.expected_calibration_error(calibrated, scores) < 0.02
    overconfident = (rng.random(20000) < 0.5).astype(float)
    assert m.expected_calibration_error(overconfident, np.where(scores > 0.5, 0.99, 0.01)) > 0.4


def test_paired_win_rate_counts_ties_as_half() -> None:
    assert m.paired_win_rate([(0.2, [0.9, 0.1]), (0.5, [0.5])]) == pytest.approx(0.5)
    assert m.paired_win_rate([]) is None


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat([0.0, 1.0], 50)
    scores = np.r_[rng.normal(0.3, 0.15, 50), rng.normal(0.7, 0.15, 50)]
    groups = [f"g{i % 25}" for i in range(100)]
    stat = lambda idx: m.roc_auc(labels[idx], scores[idx])  # noqa: E731
    first = m.bootstrap_ci(groups, stat, iterations=300, seed=7)
    assert first == m.bootstrap_ci(groups, stat, iterations=300, seed=7)
    assert first is not None and first[0] <= m.roc_auc(labels, scores) <= first[1]  # type: ignore[operator]


# --------------------------------------------------------------------------- slices


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "sample_id": "s",
        "label": "real",
        "split": "test",
        "pair_id": "p",
        "transformation": "pristine",
        "swapper_model": None,
        "dataset": "lab",
        "resolution": "1280x720",
        "score": 0.2,
        "verdict": "authentic_likely",
        "video_quality": "good",
    }
    base.update(overrides)
    return base


def test_slices_keep_reals_for_fake_only_filters() -> None:
    records = [
        record(sample_id="r1"),
        record(sample_id="f1", label="manipulated", swapper_model="inswapper_128"),
        record(sample_id="f2", label="manipulated", swapper_model="ghost_2_256"),
        record(sample_id="r2", split="val"),
        record(sample_id="x1", split="external", dataset="celeb-df", label="manipulated"),
    ]
    slices = {s.name: s for s in build_slices(records, held_out_swappers=["ghost_2_256"])}
    names = lambda name: {r["sample_id"] for r in slices[name].select(records)}  # noqa: E731
    assert names("swapper:ghost_2_256") == {"r1", "f2", "r2"}
    assert names("settings:unseen") == {"r1", "f2"}
    assert names("settings:seen") == {"r1", "f1"}
    assert names("tool:celeb-df") == {"x1"}
    assert names("identities:unseen") == {"r1", "f1", "f2"}
    assert resolution_bucket("1920x1080") == ">=1080p" and resolution_bucket("640x360") == "<480p"


def test_threshold_recommendation_uses_val_only_per_quality_band() -> None:
    val = [
        record(split="val", score=s, pair_id=f"v{i}") for i, s in enumerate(np.linspace(0, 0.6, 10))
    ]
    val += [record(split="val", label="manipulated", score=0.9, pair_id=f"v{i}") for i in range(10)]
    poor = [record(split="val", score=0.95, video_quality="poor", pair_id="p")]
    poor += [
        record(split="val", label="manipulated", score=0.97, video_quality="poor", pair_id="p")
    ]
    test = [record(split="test", score=0.99, pair_id="t")]  # would wreck a test-tuned threshold
    recs = {(r["setting"], r["target_fpr"]): r for r in recommend_thresholds(val + poor + test)}
    normal = recs[("manipulated_threshold", 0.01)]
    assert normal["threshold"] == pytest.approx(0.9) and normal["val_tpr"] == 1.0
    assert normal["reliable"] is False and normal["n_real"] == 10
    assert recs[("poor_manipulated_threshold", 0.01)]["threshold"] == pytest.approx(0.97)


# --------------------------------------------------------------------------- run + report


class Scripted:
    name = "gend"

    def __init__(self) -> None:
        self.current = 0.5

    def predict(self, crops: Sequence[npt.NDArray[np.uint8]]) -> FramePredictions:
        n = len(crops)
        return FramePredictions(
            np.full(n, self.current, np.float32), np.full((n, 4), 0.5, np.float32)
        )


class ScoreBySample:
    """The real pipeline with stub detectors; the frame score is set per sample."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.frames = Scripted()
        self.inner = Analyzer(
            PipelineConfig(), face_detector=StubFaces(), frame_detector=self.frames
        )
        self.models = self.inner.models
        self.config = self.inner.config
        self.calls = 0

    def analyze(self, path: Path, *, analysis_id: str | None = None) -> AnalysisOutput:
        self.calls += 1
        self.frames.current = self.scores[path.parent.name]
        return self.inner.analyze(path, analysis_id=analysis_id)


@pytest.fixture
def lab(tmp_path: Path) -> tuple[Registry, dict[str, float]]:
    registry = Registry(LabPaths(tmp_path / "data"))
    scores: dict[str, float] = {}
    for pair in range(6):
        split = "val" if pair < 3 else "test"
        for label, score in (("real", 0.85 if pair == 0 else 0.2), ("manipulated", 0.92)):
            sample_id = f"{label[:4]}{pair}"
            path = registry.paths.samples / sample_id / "pristine.mp4"
            path.parent.mkdir(parents=True)
            offset = pair * 1000 + (500 if label == "manipulated" else 0)
            write_video(path, frames=75, fps=25, image_for=lambda k, o=offset: barcode_frame(k + o))
            registry.samples[sample_id] = Sample(
                sample_id=sample_id,
                pair_id=f"video{pair}",
                label=label,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
                path=registry.paths.relative(path),
                sha256=sha256_of(path),
                generator="facefusion" if label == "manipulated" else "none",
                swapper_model="inswapper_128" if label == "manipulated" else None,
                source_video=f"video{pair}",
                resolution="320x240",
                fps=25.0,
                duration=2.0,
            )
            scores[sample_id] = score
    registry.save()
    return registry, scores


def test_run_caches_predictions_and_saves_traces(lab: tuple[Registry, dict[str, float]]) -> None:
    registry, scores = lab
    analyzer = ScoreBySample(scores)
    summary = run_benchmark(
        registry, analyzer, "r1", Selection(), save_traces=True, log=lambda _: None
    )
    assert (summary.analysed, summary.cached, summary.failed) == (12, 0, [])
    records = load_predictions(summary.run_dir)
    assert len(records) == 12
    first = next(
        r for r in records if r["sample_id"] == "manipulated0" or r["sample_id"] == "mani0"
    )
    assert first["score"] == pytest.approx(0.92, abs=1e-3)
    assert set(first["scores"]) == {"mean", "median", "window_max"}
    assert len(list((summary.run_dir / "traces").glob("*.npz"))) == 12

    again = run_benchmark(
        registry, analyzer, "r2", Selection(), save_traces=False, log=lambda _: None
    )
    assert (again.analysed, again.cached) == (0, 12)
    assert analyzer.calls == 12


def test_report_end_to_end(lab: tuple[Registry, dict[str, float]]) -> None:
    registry, scores = lab
    run = run_benchmark(registry, ScoreBySample(scores), "r1", Selection(), log=lambda _: None)
    report = build_report(run.run_dir, bootstrap=200)
    overall = report["slices"]["all"]
    assert (overall["n_real"], overall["n_fake"]) == (6, 6)
    assert overall["roc_auc"] == 1.0
    assert overall["fpr"] == pytest.approx(1 / 6, abs=1e-3)  # the 0.85 real is above 0.80
    assert overall["paired_win_rate"] == 1.0
    assert overall["ci95"]["roc_auc"] is not None
    assert {"split:val", "split:test", "swapper:inswapper_128", "identities:unseen"} <= set(
        report["slices"]
    )
    assert report["recommendations"][0]["reliable"] is False
    for name in ("report.json", "report.md", "roc.png", "reliability.png", "scores.png"):
        assert (run.run_dir / name).is_file(), name
    markdown = (run.run_dir / "report.md").read_text()
    assert "TPR@1%FPR" in markdown and "manipulated_threshold" in markdown
    assert json.loads((run.run_dir / "report.json").read_text())["counts"]["samples"] == 12


def test_evaluate_handles_failures_and_single_class() -> None:
    rows = [record(score=None, verdict=None), record(score=0.3)]
    result = evaluate(rows, 0.8, bootstrap=0)
    assert (result["n_real"], result["n_fake"], result["n_failed"]) == (1, 0, 1)
    assert result["roc_auc"] is None and result["tpr_at_fpr_5pct"] is None
