"""Benchmark report: metrics per slice with bootstrap CIs, plots, threshold recommendations.

The recommended thresholds come from the **val** split. They hold the false-positive rate
on real videos at 1% or 5%, on the raw score. `dfbench calibrate` (step 16) chooses them
on the calibrated scale and writes a complete pipeline profile.
Applying a threshold tuned on test would make the test numbers optimistic, so test is
used only for reporting.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from deepfake_bench import metrics as m
from deepfake_bench.run import load_predictions
from deepfake_bench.slices import Record, build_slices
from deepfake_core.config import QualityConfig

FPR_TARGETS = (0.01, 0.05)
AGGREGATORS = ("mean", "median", "window_max", "temporal")


def _arrays(
    records: Sequence[Record], score_key: str | None = None
) -> tuple[m.Array, m.Array, list[str]]:
    rows = [
        (1.0 if r["label"] == "manipulated" else 0.0, _score(r, score_key), str(r["pair_id"]))
        for r in records
        if _score(r, score_key) is not None
    ]
    labels = np.array([row[0] for row in rows], dtype=np.float64)
    scores = np.array([row[1] for row in rows], dtype=np.float64)
    return labels, scores, [row[2] for row in rows]


def _score(record: Record, key: str | None) -> float | None:
    value = record.get("score") if key is None else (record.get("scores") or {}).get(key)
    return None if value is None else float(value)


def score_pairs(records: Sequence[Record]) -> list[tuple[float, list[float]]]:
    reals: dict[tuple[str, str], float] = {}
    fakes: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        if r.get("score") is None:
            continue
        key = (str(r["pair_id"]), str(r["transformation"]))
        if r["label"] == "real":
            reals[key] = float(r["score"])
        else:
            fakes[key].append(float(r["score"]))
    return [(reals[k], v) for k, v in fakes.items() if k in reals]


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def evaluate(
    records: Sequence[Record], operating_threshold: float, *, bootstrap: int = 1000, seed: int = 0
) -> dict[str, Any]:
    labels, scores, groups = _arrays(records)
    n_fake, n_real = int(labels.sum()), int((labels == 0).sum())
    result: dict[str, Any] = {
        "n_real": n_real,
        "n_fake": n_fake,
        "n_failed": sum(1 for r in records if r.get("score") is None),
    }
    if labels.size == 0:
        return result
    conf = m.confusion(labels, scores, operating_threshold)
    at1 = m.tpr_at_fpr(labels, scores, 0.01)
    at5 = m.tpr_at_fpr(labels, scores, 0.05)
    verdicts = [r.get("verdict") for r in records if r.get("score") is not None]
    real_verdicts = [v for v, y in zip(verdicts, labels, strict=True) if y == 0]
    fake_verdicts = [v for v, y in zip(verdicts, labels, strict=True) if y == 1]
    decided = [(v, y) for v, y in zip(verdicts, labels, strict=True) if v != "inconclusive"]
    result.update(
        {
            "roc_auc": _round(m.roc_auc(labels, scores)),
            "pr_auc": _round(m.average_precision(labels, scores)),
            "tpr_at_fpr_1pct": _round(at1[0]) if at1 else None,
            "tpr_at_fpr_5pct": _round(at5[0]) if at5 else None,
            "operating_threshold": operating_threshold,
            "precision": _round(conf.precision),
            "recall": _round(conf.recall),
            "f1": _round(conf.f1),
            "fpr": _round(conf.fpr),
            "fnr": _round(conf.fnr),
            "ece": _round(m.expected_calibration_error(labels, scores)),
            "brier": _round(m.brier(labels, scores)),
            "abstention_rate": _round(verdicts.count("inconclusive") / len(verdicts)),
            "flagged_real_rate": _round(real_verdicts.count("manipulated_likely") / n_real)
            if n_real
            else None,
            "cleared_fake_rate": _round(fake_verdicts.count("authentic_likely") / n_fake)
            if n_fake
            else None,
            "decided_accuracy": _round(
                sum(1 for v, y in decided if (v == "manipulated_likely") == (y == 1)) / len(decided)
            )
            if decided
            else None,
            "paired_win_rate": _round(m.paired_win_rate(score_pairs(records))),
        }
    )
    if bootstrap and len(set(groups)) >= 2:

        def stat(fn: Any) -> Any:
            return lambda idx: fn(labels[idx], scores[idx])

        result["ci95"] = {
            "roc_auc": m.bootstrap_ci(groups, stat(m.roc_auc), iterations=bootstrap, seed=seed),
            "tpr_at_fpr_5pct": m.bootstrap_ci(
                groups,
                stat(lambda y, s: (r := m.tpr_at_fpr(y, s, 0.05)) and r[0]),
                iterations=bootstrap,
                seed=seed,
            ),
            "fpr": m.bootstrap_ci(
                groups,
                stat(lambda y, s: m.confusion(y, s, operating_threshold).fpr),
                iterations=bootstrap,
                seed=seed,
            ),
        }
    return result


THRESHOLD_GROUPS = (
    ("manipulated_threshold", ("good", "fair")),
    ("poor_manipulated_threshold", ("poor",)),
)


def recommend_thresholds(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Per verdict band, the threshold that keeps FPR on val reals at each target.

    The bands depend on quality, so each is tuned on val videos of its own quality tiers.
    """
    out = []
    for setting, tiers in THRESHOLD_GROUPS:
        subset = [r for r in records if r["split"] == "val" and r.get("video_quality") in tiers]
        labels, scores, _ = _arrays(subset)
        n_real = int((labels == 0).sum())
        for target in FPR_TARGETS:
            found = m.tpr_at_fpr(labels, scores, target) if labels.size else None
            threshold = None if found is None or not np.isfinite(found[1]) else round(found[1], 4)
            out.append(
                {
                    "setting": setting,
                    "quality_tiers": list(tiers),
                    "target_fpr": target,
                    "threshold": threshold,
                    "val_tpr": _round(found[0]) if found else None,
                    "n_real": n_real,
                    "reliable": n_real >= round(1 / target),
                }
            )
    return out


def compare_aggregators(records: Sequence[Record]) -> dict[str, Any]:
    out = {}
    for name in AGGREGATORS:
        labels, scores, _ = _arrays(records, name)
        if labels.size:
            at5 = m.tpr_at_fpr(labels, scores, 0.05)
            out[name] = {
                "roc_auc": _round(m.roc_auc(labels, scores)),
                "tpr_at_fpr_5pct": _round(at5[0]) if at5 else None,
                "paired_win_rate": _round(
                    m.paired_win_rate(
                        score_pairs([{**r, "score": _score(r, name)} for r in records])
                    )
                ),
            }
    return out


# --------------------------------------------------------------------------- outputs


def _fmt(value: Any, ci: Sequence[float] | None = None) -> str:
    if value is None:
        return "n/a"
    text = f"{value:.3f}" if isinstance(value, float) else str(value)
    return f"{text} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else text


def _markdown(report: Mapping[str, Any]) -> str:
    run = report["run"]
    lines = [
        f"# Benchmark report: {run['run_id']}",
        "",
        f"Pipeline {run['pipeline_version']}, config `{run['config_profile']}` "
        f"({run['config_hash']}). Operating threshold {run['operating_threshold']}. Models: "
        + ", ".join(f"{x['name']} ({x['revision'][:10]})" for x in run["models"])
        + ".",
        "",
        "Scores are uncalibrated detector probabilities. Confidence intervals are 95% "
        "cluster-bootstrap intervals, resampling source videos.",
        "",
        "Columns: **FPR@op** / **FNR@op** compare the raw score with the operating threshold. "
        "**flagged reals** is the share of real videos whose *verdict* was `manipulated_likely` "
        "(verdicts also apply the quality-dependent bands). **abstain** is the share of "
        "`inconclusive` verdicts. **paired win** is how often a fake scores above the real "
        "video it was made from.",
        "",
        "## Slices",
        "",
        "| Slice | reals | fakes | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@5%FPR | FPR@op | FNR@op "
        "| flagged reals | abstain | ECE | paired win |",
        "|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["slices"].items():
        ci = row.get("ci95", {})
        lines.append(
            f"| {name} | {row['n_real']} | {row['n_fake']} "
            f"| {_fmt(row.get('roc_auc'), ci.get('roc_auc'))} "
            f"| {_fmt(row.get('pr_auc'))} | {_fmt(row.get('tpr_at_fpr_1pct'))} "
            f"| {_fmt(row.get('tpr_at_fpr_5pct'), ci.get('tpr_at_fpr_5pct'))} "
            f"| {_fmt(row.get('fpr'), ci.get('fpr'))} | {_fmt(row.get('fnr'))} "
            f"| {_fmt(row.get('flagged_real_rate'))} | {_fmt(row.get('abstention_rate'))} "
            f"| {_fmt(row.get('ece'))} | {_fmt(row.get('paired_win_rate'))} |"
        )
    lines += [
        "",
        "## Aggregators (all scored lab samples)",
        "",
        "| Aggregator | ROC-AUC | TPR@5%FPR | paired win |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["aggregators"].items():
        lines.append(
            f"| {name} | {_fmt(row['roc_auc'])} | {_fmt(row['tpr_at_fpr_5pct'])} "
            f"| {_fmt(row['paired_win_rate'])} |"
        )
    lines += [
        "",
        "## Threshold recommendation (val split)",
        "",
        "| Setting | Quality tiers | Target FPR | Threshold | val TPR | val reals | Reliable |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    recs = report["recommendations"]
    for rec in recs:
        lines.append(
            f"| `{rec['setting']}` | {', '.join(rec['quality_tiers'])} | {rec['target_fpr']:.0%} "
            f"| {_fmt(rec['threshold'])} | {_fmt(rec['val_tpr'])} | {rec['n_real']} "
            f"| {'yes' if rec['reliable'] else 'no: too few reals'} |"
        )
    chosen = {
        r["setting"]: r
        for r in recs
        if r["target_fpr"] == FPR_TARGETS[0] and r["threshold"] is not None
    }
    if chosen:
        lines += [
            "",
            f"Config for the {FPR_TARGETS[0]:.0%} target (PipelineConfig overrides, `--config`):",
            "",
            "```yaml",
            "verdict:",
        ]
        lines += [f"  {setting}: {rec['threshold']}" for setting, rec in chosen.items()]
        lines.append("```")
    else:
        lines.append("")
        lines.append("Not enough val samples with both classes for a recommendation.")
    lines += _robustness_markdown(report)
    lines += _quality_markdown(report)
    lines += [
        "",
        "## Plots",
        "",
        "![ROC](roc.png)",
        "",
        "![Reliability](reliability.png)",
        "",
        "![Scores](scores.png)",
        "",
    ]
    if (report.get("robustness") or {}).get("profiles"):
        lines += ["![False positives by profile](fpr_by_profile.png)", ""]
    return "\n".join(lines)


def _robustness_markdown(report: Mapping[str, Any]) -> list[str]:
    profiles = (report.get("robustness") or {}).get("profiles") or {}
    if not profiles:
        return []
    lines = [
        "",
        "## Robustness (lab, per encoding profile)",
        "",
        "False positives on real videos are the key HR risk: a candidate on a poor connection "
        "must not look manipulated.",
        "",
        "| Profile | reals | fakes | ROC-AUC | Δ vs pristine | FPR on reals (score) "
        "| flagged reals (verdict) | TPR@5%FPR | abstain | quality tiers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in profiles.items():
        tiers = ", ".join(f"{k} {v}" for k, v in sorted(row["quality_tiers"].items()))
        lines.append(
            f"| {name} | {row['n_real']} | {row['n_fake']} | {_fmt(row['roc_auc'])} "
            f"| {_fmt(row['delta_roc_auc'])} | {_fmt(row['fpr_on_reals'])} "
            f"| {_fmt(row['flagged_real_rate'])} | {_fmt(row['tpr_at_fpr_5pct'])} "
            f"| {_fmt(row['abstention_rate'])} | {tiers} |"
        )
    return lines


def _quality_markdown(report: Mapping[str, Any]) -> list[str]:
    quality = report.get("quality_thresholds") or {}
    metrics = quality.get("metrics") or {}
    lines = ["", "## Quality thresholds (data-driven)", ""]
    if not any(v.get("status") == "ok" for v in metrics.values()):
        lines.append(
            f"Not enough real videos per quality bin ({quality.get('n_real', 0)} reals in total; "
            "each of 4 bins needs at least 20). The quality thresholds stay `uncalibrated-v0`."
        )
        return lines
    lines += [
        f"Overall FPR on lab reals: {_fmt(quality.get('baseline_fpr'))}. FPR on reals per "
        "quartile of each metric, lowest quality first:",
        "",
        "| Metric | quartile → FPR on reals | suggested POOR below |",
        "|---|---|---:|",
    ]
    tuned: dict[str, float] = {}
    for metric, row in metrics.items():
        if row["status"] != "ok":
            lines.append(f"| {metric} | insufficient data ({row['n_real']} reals) | n/a |")
            continue
        cells = "; ".join(
            f"[{_fmt(b['low'])}, {_fmt(b['high'])}] → {_fmt(b['fpr'])}" for b in row["bins"]
        )
        lines.append(f"| {metric} | {cells} | {_fmt(row['suggested_poor_below'])} |")
        if row["suggested_poor_below"] is not None:
            tuned[metric] = float(row["suggested_poor_below"])
    if not tuned:
        lines += ["", "No metric shows FPR clearly rising at low quality: thresholds unchanged."]
        return lines
    defaults = QualityConfig()
    lines += [
        "",
        "Candidate profile `quality-v1` (review before adopting):",
        "",
        "```yaml",
        "profile: quality-v1",
        "quality:",
    ]
    for metric, poor in tuned.items():
        band = getattr(defaults, metric)
        fair = max(band.fair, poor)
        good = max(band.good, fair)
        lines.append(f"  {metric}: {{good: {good:g}, fair: {fair:g}, poor: {poor:g}}}")
    lines.append("```")
    return lines


def _plots(records: Sequence[Record], out: Path, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, scores, _ = _arrays(records)
    fig, ax = plt.subplots(figsize=(5, 5))
    by_transformation: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_transformation[str(r["transformation"])].append(r)
    curves = [("all", records), *sorted(by_transformation.items())[:8]]
    for name, subset in curves:
        y, s, _ = _arrays(subset)
        if (y == 1).any() and (y == 0).any():
            fpr, tpr, _ = m.roc_points(y, s)
            ax.plot(
                fpr,
                tpr,
                label=f"{name} (AUC {m.roc_auc(y, s):.2f})",
                linewidth=2 if name == "all" else 1,
            )
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="ROC",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, 11)
    centres, rates = [], []
    for low, high in itertools.pairwise(edges):
        inside = (scores >= low) & (scores < high if high < 1 else scores <= high)
        if inside.any():
            centres.append(float(scores[inside].mean()))
            rates.append(float(labels[inside].mean()))
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax.plot(centres, rates, marker="o")
    ax.set(
        xlabel="Mean score in bin",
        ylabel="Fraction manipulated",
        title="Reliability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=120)
    plt.close(fig)

    names = sorted(by_transformation)
    fig, axes = plt.subplots(
        len(names), 1, figsize=(6, 1.6 * len(names) + 0.6), squeeze=False, sharex=True
    )
    for ax, name in zip(axes[:, 0], names, strict=True):
        y, s, _ = _arrays(by_transformation[name])
        bins = np.linspace(0, 1, 21)
        ax.hist(s[y == 0], bins=bins, alpha=0.6, label="real")
        ax.hist(s[y == 1], bins=bins, alpha=0.6, label="manipulated", hatch="//")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(name, rotation=0, ha="right", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("Manipulation score")
    fig.tight_layout()
    fig.savefig(out / "scores.png", dpi=120)
    plt.close(fig)


QUALITY_TUNABLE = (
    "resolution_short_side",
    "effective_fps",
    "bits_per_pixel",
    "face_crop_size",
    "face_sharpness",
)


def quality_analysis(
    records: Sequence[Record], threshold: float, *, bins: int = 4, min_reals_per_bin: int = 20
) -> dict[str, Any]:
    """Where does the false-positive rate on real videos rise as quality drops?

    Real lab videos (pristine and variants) are binned by quantiles of each quality
    metric. A POOR boundary is suggested only when every bin holds `min_reals_per_bin`
    reals and the lowest bins' FPR clearly exceeds the overall FPR (twice it, and at
    least 5 points above). Otherwise the metric keeps its `uncalibrated-v0` band.
    """
    reals = [
        r
        for r in records
        if r.get("dataset", "lab") == "lab" and r["label"] == "real" and r.get("score") is not None
    ]
    flagged = [float(r["score"]) >= threshold for r in reals]
    baseline = float(np.mean(flagged)) if reals else None
    out: dict[str, Any] = {"baseline_fpr": _round(baseline), "n_real": len(reals), "metrics": {}}
    for metric in QUALITY_TUNABLE:
        pairs = [
            (float(v), f)
            for r, f in zip(reals, flagged, strict=True)
            if (v := (r.get("quality_metrics") or {}).get(metric)) is not None
        ]
        if len(pairs) < bins * min_reals_per_bin or baseline is None:
            out["metrics"][metric] = {"status": "insufficient data", "n_real": len(pairs)}
            continue
        values = np.array([v for v, _ in pairs])
        hits = np.array([f for _, f in pairs], dtype=float)
        edges = np.quantile(values, np.linspace(0, 1, bins + 1))
        rows = []
        suggestion = None
        rising = True
        for i, (low, high) in enumerate(itertools.pairwise(edges)):
            inside = (values >= low) & ((values < high) if i < bins - 1 else (values <= high))
            fpr = float(hits[inside].mean()) if inside.any() else 0.0
            rows.append(
                {
                    "low": _round(float(low)),
                    "high": _round(float(high)),
                    "n_real": int(inside.sum()),
                    "fpr": _round(fpr),
                }
            )
            if rising and fpr > max(2 * baseline, baseline + 0.05):
                suggestion = float(high)
            else:
                rising = False
        out["metrics"][metric] = {
            "status": "ok",
            "bins": rows,
            "suggested_poor_below": _round(suggestion),
        }
    return out


def build_report(run_dir: Path, *, bootstrap: int = 1000, seed: int = 0) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    records = load_predictions(run_dir)
    threshold = float(run["operating_threshold"])
    slices = build_slices(records, run.get("held_out_swappers", []))
    report: dict[str, Any] = {
        "run": run,
        "counts": {
            "samples": len(records),
            "failed": sum(1 for r in records if r.get("score") is None),
        },
        "slices": {
            s.name: {
                "description": s.description,
                **evaluate(s.select(records), threshold, bootstrap=bootstrap, seed=seed),
            }
            for s in slices
        },
        "aggregators": compare_aggregators(
            [r for r in records if r.get("dataset", "lab") == "lab"]
        ),
        "recommendations": recommend_thresholds(records),
    }
    report["robustness"] = robustness(records, threshold)
    report["quality_thresholds"] = quality_analysis(records, threshold)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    scored = [r for r in records if r.get("score") is not None]
    if scored:
        _plots(scored, run_dir, threshold)
        if report["robustness"]["profiles"]:
            _fpr_plot(report["robustness"], run_dir)
    (run_dir / "report.md").write_text(_markdown(report))
    return report


def robustness(records: Sequence[Record], threshold: float) -> dict[str, Any]:
    """Per transformation: metric deltas vs pristine, FPR on reals, quality-tier mix."""
    lab = [r for r in records if r.get("dataset", "lab") == "lab" and r.get("score") is not None]
    by_profile: dict[str, list[Record]] = defaultdict(list)
    for r in lab:
        by_profile[str(r["transformation"])].append(r)
    if "pristine" not in by_profile or len(by_profile) < 2:
        return {"profiles": {}}
    base = evaluate(by_profile["pristine"], threshold, bootstrap=0)
    profiles = {}
    for name, subset in sorted(by_profile.items()):
        row = evaluate(subset, threshold, bootstrap=0)
        tiers: dict[str, int] = defaultdict(int)
        for r in subset:
            tiers[str(r.get("video_quality"))] += 1
        profiles[name] = {
            "n_real": row["n_real"],
            "n_fake": row["n_fake"],
            "roc_auc": row.get("roc_auc"),
            "delta_roc_auc": None
            if row.get("roc_auc") is None or base.get("roc_auc") is None
            else round(row["roc_auc"] - base["roc_auc"], 4),
            "fpr_on_reals": row.get("fpr"),
            "flagged_real_rate": row.get("flagged_real_rate"),
            "tpr_at_fpr_5pct": row.get("tpr_at_fpr_5pct"),
            "abstention_rate": row.get("abstention_rate"),
            "quality_tiers": dict(tiers),
        }
    return {"baseline": "pristine", "profiles": profiles}


def _fpr_plot(robust: Mapping[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(robust["profiles"])
    values = [robust["profiles"][n]["fpr_on_reals"] or 0.0 for n in names]
    flagged = [robust["profiles"][n]["flagged_real_rate"] or 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.2))
    positions = np.arange(len(names))
    ax.barh(positions - 0.2, values, height=0.4, label="score >= threshold")
    ax.barh(positions + 0.2, flagged, height=0.4, label="verdict manipulated_likely", hatch="//")
    ax.set_yticks(positions, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rate on real videos")
    ax.set_title("False positives on real videos per profile")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fpr_by_profile.png", dpi=120)
    plt.close(fig)
