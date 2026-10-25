"""Roadmap step 16: fit calibration on val, choose verdict thresholds, report on test.

    dfbench calibrate RUN [--method platt|isotonic] [--input gend --input temporal]

1. **Fit on val.** The inputs are the video-level raw scores of the run: `gend` is the
   run's track aggregator, and `temporal` is the temporal model's score when the run has
   one. Classes are weighted equally, so the result is P(manipulated) at a 50% prior.
   `--prior` shifts it to an expected base rate.
2. **Thresholds on val**, on the calibrated scale, per quality band:
   - `manipulated` keeps the FPR on val reals at `--target-fpr` (default 1%);
   - `authentic` keeps the share of val fakes cleared as authentic at most
     `--target-cleared-fake` (default 5%).
   When there is no val data for the poor band, its thresholds are derived from the
   normal band by a fixed rule: stricter on both sides. The report says when this
   happened.
3. **Report on test** (never used for fitting). It shows class-balanced ECE, Brier and
   log loss, raw vs calibrated, overall and per quality tier. It also shows verdict
   rates under the run's thresholds and under the new ones.

The output is a complete pipeline profile, `calibration/<method>.yaml`: the run's
config plus calibration and thresholds. It is bound to the checksums of the scoring
models. Use it with `deepfake-core analyze --config`, the API's config file or
`dfbench run --pipeline-config`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from deepfake_bench import metrics as m
from deepfake_bench.lab import LabError
from deepfake_bench.run import load_predictions
from deepfake_bench.slices import Record
from deepfake_core.config import CalibrationConfig, PipelineConfig, VerdictConfig
from deepfake_core.scoring.calibration import calibrate, logit

Method = Literal["platt", "isotonic"]
TIERS = ("good", "fair", "poor")
MIN_PER_CLASS = 20
"""Below this many val reals or fakes, the fit is marked unreliable."""


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    config: PipelineConfig
    profile: Path
    report: dict[str, Any]


def _pipeline_config(run: dict[str, Any]) -> PipelineConfig:
    data = run.get("pipeline_config")
    if not data:
        raise LabError(
            f"run {run['run_id']} does not record its pipeline config: re-run it with this "
            "version of dfbench"
        )
    return PipelineConfig.model_validate(data)


def raw_inputs(record: Record, gend_method: str) -> dict[str, float | None]:
    scores = record.get("scores") or {}
    return {"gend": scores.get(gend_method), "temporal": scores.get("temporal")}


def _usable(records: Sequence[Record], inputs: Sequence[str], gend_method: str) -> list[Record]:
    return [
        r
        for r in records
        if r.get("error") is None
        and r.get("score") is not None
        and all(raw_inputs(r, gend_method)[name] is not None for name in inputs)
    ]


def _labels(records: Sequence[Record]) -> m.Array:
    return np.array([1.0 if r["label"] == "manipulated" else 0.0 for r in records])


def fit_calibration(
    records: Sequence[Record],
    inputs: Sequence[str],
    gend_method: str,
    method: Method,
) -> dict[str, Any]:
    """Calibration parameters (CalibrationConfig fields) fitted on `records`."""
    labels = _labels(records)
    if len(set(labels.tolist())) < 2:
        raise LabError("calibration needs val videos of both classes")
    features = np.array(
        [[logit(float(raw_inputs(r, gend_method)[n] or 0.0)) for n in inputs] for r in records]
    )
    weights = m.balanced_weights(labels)
    if method == "platt":
        model = LogisticRegression(C=1.0)
        model.fit(features, labels, sample_weight=weights * len(labels))
        return {
            "intercept": float(model.intercept_[0]),
            "coefficients": tuple(float(c) for c in model.coef_[0]),
        }
    if len(inputs) != 1:
        raise LabError("isotonic calibration takes exactly one input")
    raw = np.array([float(raw_inputs(r, gend_method)[inputs[0]] or 0.0) for r in records])
    iso = IsotonicRegression(y_min=1e-3, y_max=1 - 1e-3, out_of_bounds="clip")
    iso.fit(raw, labels, sample_weight=weights * len(labels))
    x, y = iso.X_thresholds_, iso.y_thresholds_
    if len(x) < 2:
        raise LabError("isotonic calibration found a single level: too little val data")
    return {
        "isotonic_x": tuple(float(v) for v in x),
        "isotonic_y": tuple(float(v) for v in y),
    }


def _round_down(value: float, digits: int = 4) -> float:
    scale = 10.0**digits
    return math.floor(value * scale) / scale


def authentic_threshold(fake_scores: Sequence[float], target: float) -> float | None:
    """Highest threshold that clears at most `target` of the fakes (score <= threshold)."""
    if not fake_scores:
        return None
    ordered = sorted(fake_scores)
    allowed = math.floor(target * len(ordered))
    if allowed >= len(ordered):
        return _round_down(ordered[-1])
    return max(0.0, _round_down(ordered[allowed] - 1e-6))


def manipulated_threshold(
    real_scores: Sequence[float], fake_scores: Sequence[float], target_fpr: float
) -> tuple[float, float | None] | None:
    """(threshold, TPR) keeping the FPR on the reals at most `target_fpr`."""
    if not real_scores:
        return None
    labels = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    scores = np.r_[np.asarray(real_scores, float), np.asarray(fake_scores, float)]
    found = m.tpr_at_fpr(labels, scores, target_fpr)
    if found is None or not np.isfinite(found[1]):
        return None
    return round(float(found[1]), 4), float(found[0]) if fake_scores else None


def choose_thresholds(
    val: Sequence[Record], target_fpr: float, target_cleared_fake: float, base: VerdictConfig
) -> tuple[VerdictConfig, dict[str, Any]]:
    """Verdict bands on calibrated val scores (`score` holds the calibrated value).

    The normal band uses good and fair val videos, and every tier when there are none.
    The poor band uses poor val videos. Without them, it is derived from the normal band:
    the manipulated threshold moves halfway to 1 and the authentic threshold halves.
    Poor video never decides more readily than normal video."""

    def scores(label: str, tiers: Sequence[str] | None) -> list[float]:
        return [
            float(r["score"])
            for r in val
            if r["label"] == label and (tiers is None or r.get("video_quality") in tiers)
        ]

    normal_tiers: Sequence[str] | None = ("good", "fair")
    if not scores("real", normal_tiers) or not scores("manipulated", normal_tiers):
        normal_tiers = None
    reals, fakes = scores("real", normal_tiers), scores("manipulated", normal_tiers)
    found = manipulated_threshold(reals, fakes, target_fpr)
    authentic = authentic_threshold(fakes, target_cleared_fake)
    if found is None or authentic is None:
        raise LabError("val needs real and manipulated videos to choose thresholds")
    manipulated, val_tpr = found
    authentic = min(authentic, _round_down(manipulated - 1e-4))
    notes: dict[str, Any] = {
        "normal": {
            "manipulated": manipulated,
            "authentic": authentic,
            "val_tpr": _r(val_tpr),
            "n_real": len(reals),
            "n_fake": len(fakes),
            "quality_tiers": list(normal_tiers) if normal_tiers else "all",
            "reliable": len(reals) >= round(1 / target_fpr),
        }
    }

    poor_found = manipulated_threshold(
        scores("real", ("poor",)), scores("manipulated", ("poor",)), target_fpr
    )
    poor_authentic = authentic_threshold(scores("manipulated", ("poor",)), target_cleared_fake)
    derived = poor_found is None or poor_authentic is None
    poor_manipulated = poor_found[0] if poor_found else round(1 - (1 - manipulated) / 2, 4)
    if poor_authentic is None:
        poor_authentic = _round_down(authentic / 2)
    notes["poor"] = {
        "manipulated": max(poor_manipulated, manipulated),
        "authentic": min(poor_authentic, authentic),
        "derived_from_normal_band": derived,
    }
    verdict = base.model_copy(
        update={
            "manipulated_threshold": manipulated,
            "authentic_threshold": authentic,
            "poor_manipulated_threshold": notes["poor"]["manipulated"],
            "poor_authentic_threshold": notes["poor"]["authentic"],
        }
    )
    return verdict, notes


def _verdict(score: float, quality: str | None, verdict: VerdictConfig) -> str:
    if quality in (None, "insufficient"):
        return "inconclusive"
    poor = quality == "poor"
    manipulated = verdict.poor_manipulated_threshold if poor else verdict.manipulated_threshold
    authentic = verdict.poor_authentic_threshold if poor else verdict.authentic_threshold
    if score >= manipulated:
        return "manipulated_likely"
    if score <= authentic:
        return "authentic_likely"
    return "inconclusive"


def _quality_metrics(records: Sequence[Record], scores: m.Array) -> dict[str, Any]:
    labels = _labels(records)
    if not len(records):
        return {"n_real": 0, "n_fake": 0}
    weights = m.balanced_weights(labels)
    both = len(set(labels.tolist())) == 2
    return {
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "ece_balanced": _r(m.expected_calibration_error(labels, scores, weights=weights))
        if both
        else None,
        "brier_balanced": _r(m.brier(labels, scores, weights)) if both else None,
        "log_loss_balanced": _r(m.log_loss(labels, scores, weights)) if both else None,
        "roc_auc": _r(m.roc_auc(labels, scores)),
    }


def _verdict_rates(
    records: Sequence[Record], scores: m.Array, verdict: VerdictConfig
) -> dict[str, Any]:
    decided = [
        (_verdict(float(s), r.get("video_quality"), verdict), r["label"])
        for r, s in zip(records, scores, strict=True)
    ]
    reals = [v for v, label in decided if label == "real"]
    fakes = [v for v, label in decided if label == "manipulated"]
    return {
        "flagged_real_rate": _r(reals.count("manipulated_likely") / len(reals)) if reals else None,
        "cleared_fake_rate": _r(fakes.count("authentic_likely") / len(fakes)) if fakes else None,
        "detected_fake_rate": _r(fakes.count("manipulated_likely") / len(fakes)) if fakes else None,
        "abstention_rate": _r([v for v, _ in decided].count("inconclusive") / len(decided))
        if decided
        else None,
    }


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def run_calibration(
    run_dir: Path,
    *,
    method: Method = "platt",
    inputs: Sequence[str] | None = None,
    prior: float = 0.5,
    target_fpr: float = 0.01,
    target_cleared_fake: float = 0.05,
) -> CalibrationResult:
    run = json.loads((run_dir / "run.json").read_text())
    base = _pipeline_config(run)
    track_score = base.aggregation.track_score
    gend_method = "mean" if track_score == "temporal" else track_score
    records = load_predictions(run_dir)
    val_all = [r for r in records if r.get("split") == "val"]
    if inputs is None:
        has_temporal = bool(val_all) and all(
            raw_inputs(r, gend_method)["temporal"] is not None
            for r in val_all
            if r.get("score") is not None
        )
        inputs = ("gend", "temporal") if has_temporal else ("gend",)
    inputs = tuple(inputs)
    val = _usable(val_all, inputs, gend_method)
    test = _usable([r for r in records if r.get("split") == "test"], inputs, gend_method)
    if not val:
        raise LabError(f"run {run_dir.name} has no scored val videos with inputs {inputs}")

    roles = {"gend": "frame_detector", "temporal": "temporal"}
    model_sha = tuple(
        model["sha256"]
        for model in run.get("models", [])
        if model.get("role") in {roles[name] for name in inputs}
    )
    if len(model_sha) != len(inputs):
        raise LabError(f"run {run_dir.name} does not record the models behind {inputs}")
    val_labels = _labels(val)
    n_real, n_fake = int((val_labels == 0).sum()), int((val_labels == 1).sum())
    calibration = CalibrationConfig(
        method=method,
        inputs=inputs,
        prior=prior,
        fitted_on=f"run {run['run_id']}, val: {n_real} real / {n_fake} manipulated",
        model_sha256=model_sha,
        **fit_calibration(val, inputs, gend_method, method),
    )

    def calibrated(rows: Sequence[Record]) -> m.Array:
        return np.array([calibrate(raw_inputs(r, gend_method), calibration) or 0.0 for r in rows])

    val_calibrated = [{**r, "score": s} for r, s in zip(val, calibrated(val), strict=True)]
    verdict, threshold_notes = choose_thresholds(
        val_calibrated, target_fpr, target_cleared_fake, base.verdict
    )
    config = base.model_copy(
        update={
            "profile": f"calibrated-{run['run_id']}-{method}",
            "calibration": calibration,
            "verdict": verdict,
        }
    )
    config = PipelineConfig.model_validate(config.model_dump(mode="json"))  # re-validate

    test_raw = np.array([float(r["scores"].get(track_score, r["score"])) for r in test])
    test_cal = calibrated(test)
    report: dict[str, Any] = {
        "run_id": run["run_id"],
        "profile": config.profile,
        "method": method,
        "inputs": list(inputs),
        "prior": prior,
        "fitted_on": calibration.fitted_on,
        "reliable": n_real >= MIN_PER_CLASS and n_fake >= MIN_PER_CLASS,
        "parameters": calibration.model_dump(
            mode="json", include={"intercept", "coefficients", "isotonic_x", "isotonic_y"}
        ),
        "targets": {"fpr": target_fpr, "cleared_fake": target_cleared_fake},
        "thresholds": threshold_notes,
        "test": {
            "raw": _quality_metrics(test, test_raw),
            "calibrated": _quality_metrics(test, test_cal),
            "by_quality": {
                tier: {
                    "raw": _quality_metrics(
                        [r for r in test if r.get("video_quality") == tier],
                        test_raw[[r.get("video_quality") == tier for r in test]],
                    ),
                    "calibrated": _quality_metrics(
                        [r for r in test if r.get("video_quality") == tier],
                        test_cal[[r.get("video_quality") == tier for r in test]],
                    ),
                }
                for tier in TIERS
            },
            "verdicts_before": _verdict_rates(test, test_raw, base.verdict),
            "verdicts_after": _verdict_rates(test, test_cal, verdict),
        },
    }
    out = run_dir / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    profile = out / f"{method}.yaml"
    profile.write_text(
        f"# Generated by `dfbench calibrate` from run {run['run_id']} (val split).\n"
        + yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    (out / f"{method}.report.json").write_text(json.dumps(report, indent=2))
    (out / f"{method}.report.md").write_text(_markdown(report))
    return CalibrationResult(config, profile, report)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def _markdown(report: dict[str, Any]) -> str:
    test = report["test"]
    normal, poor = report["thresholds"]["normal"], report["thresholds"]["poor"]
    lines = [
        f"# Calibration `{report['profile']}`",
        "",
        f"- Method: {report['method']} on {', '.join(report['inputs'])}; prior "
        f"{report['prior']:g}; fitted on {report['fitted_on']}",
        f"- Reliable: {'yes' if report['reliable'] else 'no'} "
        f"(needs at least {MIN_PER_CLASS} val videos per class)",
        f"- Thresholds (val, calibrated scale, quality {normal['quality_tiers']}): "
        f"manipulated >= {normal['manipulated']} "
        f"(target FPR {report['targets']['fpr']:g}, val TPR {_fmt(normal['val_tpr'])}), "
        f"authentic <= {normal['authentic']} "
        f"(target cleared-fake rate {report['targets']['cleared_fake']:g})",
        f"- Poor video: manipulated >= {poor['manipulated']}, authentic <= {poor['authentic']}"
        + (
            " (derived from the normal band: no poor val data)"
            if poor["derived_from_normal_band"]
            else ""
        ),
        "",
        "Metrics are class-balanced, which matches the 50% prior of the fit, and computed on "
        "test, which was not used for fitting.",
        "",
        "| Test | n real | n fake | ECE | Brier | Log loss | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def row(label: str, data: dict[str, Any]) -> str:
        return (
            f"| {label} | {data['n_real']} | {data['n_fake']} | {_fmt(data.get('ece_balanced'))} "
            f"| {_fmt(data.get('brier_balanced'))} | {_fmt(data.get('log_loss_balanced'))} "
            f"| {_fmt(data.get('roc_auc'))} |"
        )

    lines += [row("all, raw", test["raw"]), row("all, calibrated", test["calibrated"])]
    for tier, data in test["by_quality"].items():
        if data["raw"]["n_real"] + data["raw"]["n_fake"]:
            lines += [
                row(f"{tier}, raw", data["raw"]),
                row(f"{tier}, calibrated", data["calibrated"]),
            ]
    lines += [
        "",
        "| Test verdicts | Flagged reals | Cleared fakes | Detected fakes | Abstentions |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("run's thresholds, raw", "verdicts_before"),
        ("calibrated", "verdicts_after"),
    ):
        v = test[key]
        lines.append(
            f"| {label} | {_fmt(v['flagged_real_rate'])} | {_fmt(v['cleared_fake_rate'])} | "
            f"{_fmt(v['detected_fake_rate'])} | {_fmt(v['abstention_rate'])} |"
        )
    return "\n".join(lines) + "\n"
