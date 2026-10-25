"""Calibration on synthetic predictions where the raw score is badly calibrated: GenD-like
over-confidence, with real videos at about 0.6 and fakes at about 0.85."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deepfake_bench import metrics as m
from deepfake_bench.calibrate import authentic_threshold, run_calibration
from deepfake_bench.cli import main
from deepfake_bench.lab import LabError
from deepfake_bench.manifest import Registry
from deepfake_bench.paths import LabPaths
from deepfake_bench.run import Selection, run_benchmark
from deepfake_core.config import PipelineConfig
from deepfake_core.models import GEND
from deepfake_core.pipeline import Analyzer
from tests.bench.test_benchmark import ScoreBySample, lab  # noqa: F401  (fixture)
from tests.bench.test_lab import StubFaces


def synthetic_run(
    root: Path,
    *,
    pairs: int = 60,
    temporal: bool = False,
    poor: bool = False,
    seed: int = 0,
    pipeline_config: dict[str, Any] | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    run_dir = LabPaths(root).runs / "r1"
    run_dir.mkdir(parents=True)
    lines = []
    for split in ("val", "test"):
        for pair in range(pairs):
            quality = "poor" if poor and pair % 3 == 0 else "good"
            for label in ("real", "manipulated"):
                fake = label == "manipulated"
                gend = float(np.clip(rng.normal(0.85 if fake else 0.6, 0.12), 0.01, 0.99))
                scores: dict[str, float] = {"mean": gend, "median": gend, "window_max": gend}
                if temporal:  # an independent, informative second opinion
                    scores["temporal"] = float(
                        np.clip(rng.normal(0.7 if fake else 0.3, 0.15), 0.01, 0.99)
                    )
                lines.append(
                    json.dumps(
                        {
                            "sample_id": f"{split}-{label[:4]}{pair}",
                            "label": label,
                            "split": split,
                            "pair_id": f"{split}-v{pair}",
                            "transformation": "pristine",
                            "dataset": "lab",
                            "video_quality": quality,
                            "score": gend,
                            "scores": scores,
                            "verdict": "inconclusive",
                            "error": None,
                        }
                    )
                    + "\n"
                )
    (run_dir / "predictions.jsonl").write_text("".join(lines))
    models = [{"name": GEND.name, "role": "frame_detector", "sha256": GEND.sha256}]
    config = pipeline_config or PipelineConfig().model_dump(mode="json")
    if temporal:
        models.append({"name": "Temporal tcn", "role": "temporal", "sha256": "t" * 64})
        config["temporal"] = {"checkpoint": "/lab/bench/temporal/tcn.safetensors"}
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "r1", "models": models, "pipeline_config": config})
    )
    return run_dir


def test_balanced_metrics() -> None:
    labels = np.array([0, 0, 0, 1], float)
    weights = m.balanced_weights(labels)
    assert weights.tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 6, 0.5])
    scores = np.array([0.2, 0.2, 0.2, 0.8])
    assert m.brier(labels, scores, weights) == pytest.approx(0.04)
    assert m.expected_calibration_error(labels, scores, weights=weights) == pytest.approx(0.2)
    balanced = np.array([0, 1], float)
    assert m.brier(balanced, np.array([0.3, 0.6])) == m.brier(
        balanced, np.array([0.3, 0.6]), m.balanced_weights(balanced)
    )
    assert m.log_loss(np.array([1.0]), np.array([1.0])) == pytest.approx(0, abs=1e-5)


def test_authentic_threshold_clears_at_most_the_target() -> None:
    fakes = [i / 100 for i in range(1, 101)]  # 0.01 ... 1.00
    threshold = authentic_threshold(fakes, 0.05)
    assert threshold is not None and sum(f <= threshold for f in fakes) == 5
    assert authentic_threshold(fakes[:10], 0.05) == pytest.approx(0.0099, abs=1e-4)
    assert authentic_threshold([], 0.05) is None


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_improves_test_calibration(tmp_path: Path, method: str) -> None:
    run_dir = synthetic_run(tmp_path / "data")
    result = run_calibration(run_dir, method=method)  # type: ignore[arg-type]
    report = result.report
    raw, calibrated = report["test"]["raw"], report["test"]["calibrated"]
    assert calibrated["ece_balanced"] < raw["ece_balanced"] / 2
    assert calibrated["brier_balanced"] < raw["brier_balanced"]
    assert report["reliable"] is True and report["inputs"] == ["gend"]

    config = PipelineConfig.from_file(result.profile)
    assert config.profile == f"calibrated-r1-{method}"
    assert config.calibration.method == method
    assert config.calibration.model_sha256 == (GEND.sha256,)
    assert "val: 60 real / 60 manipulated" in config.calibration.fitted_on
    v = config.verdict
    assert v.authentic_threshold < v.manipulated_threshold
    assert v.poor_manipulated_threshold >= v.manipulated_threshold
    assert v.poor_authentic_threshold <= v.authentic_threshold
    assert report["thresholds"]["poor"]["derived_from_normal_band"] is True
    for name in (f"{method}.yaml", f"{method}.report.json", f"{method}.report.md"):
        assert (run_dir / "calibration" / name).is_file()
    assert "| all, calibrated |" in (run_dir / "calibration" / f"{method}.report.md").read_text()


def test_thresholds_hold_the_val_targets(tmp_path: Path) -> None:
    run_dir = synthetic_run(tmp_path / "data", pairs=200, poor=True)
    result = run_calibration(run_dir, target_fpr=0.05, target_cleared_fake=0.05)
    v = result.config.verdict
    records = [
        json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()
    ]
    val = [r for r in records if r["split"] == "val" and r["video_quality"] == "good"]
    from deepfake_core.scoring.calibration import calibrate

    def calibrated(r: dict[str, Any]) -> float:
        value = calibrate({"gend": r["scores"]["mean"]}, result.config.calibration)
        assert value is not None
        return value

    reals = [calibrated(r) for r in val if r["label"] == "real"]
    fakes = [calibrated(r) for r in val if r["label"] == "manipulated"]
    assert np.mean([s >= v.manipulated_threshold for s in reals]) <= 0.05
    assert np.mean([s <= v.authentic_threshold for s in fakes]) <= 0.05
    assert result.report["thresholds"]["poor"]["derived_from_normal_band"] is False


def test_fusion_uses_both_scores_and_the_prior_shifts(tmp_path: Path) -> None:
    run_dir = synthetic_run(tmp_path / "data", temporal=True)
    result = run_calibration(run_dir)
    calibration = result.config.calibration
    assert calibration.inputs == ("gend", "temporal")
    assert len(calibration.coefficients) == 2 and min(calibration.coefficients) > 0
    assert calibration.model_sha256 == (GEND.sha256, "t" * 64)
    only_gend = run_calibration(run_dir, inputs=("gend",))
    assert (
        result.report["test"]["calibrated"]["roc_auc"]
        > only_gend.report["test"]["calibrated"]["roc_auc"]
    )

    rare = run_calibration(run_dir, inputs=("gend",), prior=0.01)
    assert rare.config.calibration.prior == 0.01
    assert (
        rare.config.verdict.manipulated_threshold < only_gend.config.verdict.manipulated_threshold
    )


def test_calibration_needs_a_recorded_config_and_both_classes(tmp_path: Path) -> None:
    run_dir = synthetic_run(tmp_path / "a")
    run = json.loads((run_dir / "run.json").read_text())
    del run["pipeline_config"]
    (run_dir / "run.json").write_text(json.dumps(run))
    with pytest.raises(LabError, match="pipeline config"):
        run_calibration(run_dir)

    run_dir = synthetic_run(tmp_path / "b")
    lines = (run_dir / "predictions.jsonl").read_text().splitlines()
    kept = [x for x in lines if not ('"val"' in x and '"manipulated"' in x)]
    (run_dir / "predictions.jsonl").write_text("\n".join(kept) + "\n")
    with pytest.raises(LabError, match="both classes"):
        run_calibration(run_dir)


def test_profile_round_trip_through_the_real_pipeline(
    lab: tuple[Registry, dict[str, float]],  # noqa: F811
) -> None:
    registry, scores = lab
    run = run_benchmark(registry, ScoreBySample(scores), "r1", Selection(), log=lambda _: None)
    result = run_calibration(run.run_dir)
    sample = registry.samples["mani0"]
    analyzer = ScoreBySample(scores)
    analyzer.inner = Analyzer(
        PipelineConfig.from_file(result.profile),
        face_detector=StubFaces(),
        frame_detector=analyzer.frames,
    )
    output = analyzer.analyze(registry.paths.absolute(sample.path))
    assert output.result.calibration.method == "platt"
    assert output.result.calibration.raw_probability == pytest.approx(0.92, abs=1e-3)
    assert output.result.processing.config_profile == "calibrated-r1-platt"


def test_cli_calibrate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"
    synthetic_run(root, pairs=10)
    assert main(["--data-root", str(root), "calibrate", "r1", "--method", "isotonic"]) == 0
    assert "unreliable" in capsys.readouterr().out
    assert (root / "bench" / "runs" / "r1" / "calibration" / "isotonic.yaml").is_file()
    assert main(["--data-root", str(root), "calibrate", "missing"]) == 1
