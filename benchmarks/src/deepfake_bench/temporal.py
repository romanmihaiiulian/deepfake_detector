"""Roadmap step 14: train and evaluate the temporal model on benchmark traces.

The input is one benchmark run with traces over all three splits:

    dfbench run --run-id traces --split train --split val --split test --save-traces

- `train` fits the network on the train split only. The val split picks the epoch:
  early stopping on video-level AUC, with epoch 0 (the untrained network, which equals
  the GenD mean-logit baseline) as a candidate. A checkpoint is therefore never worse on
  val than the detector it extends.
- `evaluate` compares the checkpoint with the mean, median and window_max baselines on
  every split except train. It computes them all from the same traces and the same
  eligible tracks, and gives a pair-level bootstrap CI for the AUC difference.

Lab splits are by identity, so no person is in two splits. Each checkpoint has a
training record (`<name>.json`) listing the samples and identities it was fitted on.
Withdrawing an identity's consent deletes every checkpoint trained on it
(`lab.delete_identity`), and evaluation excludes any sample that shares an identity with
the training record.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from deepfake_bench import metrics as m
from deepfake_bench.lab import LabError
from deepfake_bench.report import score_pairs
from deepfake_bench.run import load_predictions
from deepfake_core.config import AggregationConfig
from deepfake_core.pipeline import TraceTrack
from deepfake_core.quality.gate import Tier, median_tier
from deepfake_core.scoring.aggregate import window_max_mean
from deepfake_core.temporal import (
    CheckpointMeta,
    TemporalHParams,
    TemporalNet,
    TemporalScorer,
    make_windows,
    save_checkpoint,
    score_sequence,
    sequence_features,
)
from deepfake_core.temporal.model import Architecture

Log = Callable[[str], None]
BASELINES = ("mean", "median", "window_max")
_TRACE_KEY = re.compile(r"track(\d+)_(times|p_fake|tiers|embeddings)$")


# --------------------------------------------------------------------------- data


def load_trace(path: Path) -> list[TraceTrack]:
    """The tracks of a trace written by `AnalysisTrace.save_npz`."""
    parts: dict[int, dict[str, Any]] = defaultdict(dict)
    with np.load(path) as data:
        for key in data.files:
            if match := _TRACE_KEY.match(key):
                parts[int(match[1])][match[2]] = data[key]
    return [
        TraceTrack(
            track_id=track_id,
            times=p["times"],
            p_fake=p["p_fake"],
            tiers=p["tiers"],
            embeddings=p["embeddings"],
        )
        for track_id, p in sorted(parts.items())
    ]


def is_eligible(track: TraceTrack, config: AggregationConfig) -> bool:
    """The pipeline's rule: enough scored frames and median face quality at least POOR."""
    scored = int(np.isfinite(track.p_fake).sum())
    tier = median_tier(Tier(int(t)) for t in track.tiers)
    return scored >= config.min_scored_frames and tier >= Tier.POOR


@dataclass(frozen=True, slots=True)
class Example:
    sample_id: str
    key: str
    pair_id: str
    label: int
    split: str
    transformation: str
    identities: frozenset[str]
    tracks: tuple[TraceTrack, ...]
    """Eligible tracks only (at least one)."""

    @property
    def main_track(self) -> TraceTrack:
        """The track with the most scored frames: the interviewee in lab footage."""
        return max(self.tracks, key=lambda t: (int(np.isfinite(t.p_fake).sum()), -t.track_id))


def load_examples(
    run_dir: Path, aggregation: AggregationConfig | None = None
) -> tuple[list[Example], int]:
    """Examples with at least one eligible track, and the number of records skipped
    (failed, no trace, or no eligible track)."""
    aggregation = aggregation or AggregationConfig()
    examples, skipped = [], 0
    for record in load_predictions(run_dir):
        trace = run_dir / "traces" / f"{record['sample_id']}.npz"
        if record.get("error") or not trace.is_file():
            skipped += 1
            continue
        tracks = tuple(t for t in load_trace(trace) if is_eligible(t, aggregation))
        if not tracks:
            skipped += 1
            continue
        identities = {record.get("source_identity"), record.get("target_identity")} - {None}
        examples.append(
            Example(
                sample_id=str(record["sample_id"]),
                key=str(record.get("key", "")),
                pair_id=str(record["pair_id"]),
                label=1 if record["label"] == "manipulated" else 0,
                split=str(record["split"]),
                transformation=str(record.get("transformation", "pristine")),
                identities=frozenset(str(i) for i in identities),
                tracks=tracks,
            )
        )
    return examples, skipped


def frame_model_sha256(run_dir: Path) -> str:
    run = json.loads((run_dir / "run.json").read_text())
    for model in run.get("models", []):
        if model.get("role") == "frame_detector":
            return str(model["sha256"])
    raise LabError(f"run {run_dir.name} does not record its frame model")


# --------------------------------------------------------------------------- training


@dataclass(frozen=True, slots=True)
class TrainConfig:
    arch: Architecture = "tcn"
    window: int = 16
    """Samples per window: 4 s at the default 4 Hz."""
    stride: int = 8
    hidden: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 40
    patience: int = 8
    batch_size: int = 64
    seed: int = 0
    device: str = "cpu"


@dataclass(slots=True)
class TrainResult:
    name: str
    checkpoint: Path
    record: Path
    best_epoch: int
    val_auc: float | None
    baseline_val_auc: float | None
    history: list[dict[str, float | None]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Windows:
    x: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    weight: torch.Tensor


def _windows(examples: Sequence[Example], window: int, stride: int) -> _Windows:
    """Windows of every example's main track. Each example weighs 1 in total, whatever
    its length, and both classes weigh the same."""
    xs, masks, ys, ws = [], [], [], []
    for ex in examples:
        track = ex.main_track
        _, features = sequence_features(track.times, track.p_fake, track.embeddings)
        x, mask, _ = make_windows(features, window, stride)
        xs.append(x)
        masks.append(mask)
        ys.append(np.full(len(x), ex.label, np.float32))
        ws.append(np.full(len(x), 1.0 / len(x), np.float32))
    y, w = np.concatenate(ys), np.concatenate(ws)
    for label in (0.0, 1.0):
        total = w[y == label].sum()
        if total > 0:
            w[y == label] *= 0.5 / total
    return _Windows(
        torch.from_numpy(np.concatenate(xs)),
        torch.from_numpy(np.concatenate(masks)),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )


def _weighted_bce(net: TemporalNet, data: _Windows, device: torch.device) -> float:
    with torch.inference_mode():
        logits = net(data.x.to(device), data.mask.to(device)).cpu()
    losses = F.binary_cross_entropy_with_logits(logits, data.y, reduction="none")
    return float((losses * data.weight).sum() / data.weight.sum())


def video_scores(
    examples: Sequence[Example], score: Callable[[TraceTrack], float | None]
) -> npt.NDArray[np.float64]:
    """Max over eligible tracks, as the pipeline does; NaN when no track has a score."""
    out = []
    for ex in examples:
        values = [v for t in ex.tracks if (v := score(t)) is not None]
        out.append(max(values) if values else np.nan)
    return np.array(out, np.float64)


def _auc(examples: Sequence[Example], scores: npt.NDArray[np.float64]) -> float | None:
    labels = np.array([ex.label for ex in examples], np.float64)
    keep = np.isfinite(scores)
    return m.roc_auc(labels[keep], scores[keep]) if keep.any() else None


def _training_set_sha256(examples: Sequence[Example]) -> str:
    material = "\n".join(sorted(f"{ex.sample_id}:{ex.key}" for ex in examples))
    return hashlib.sha256(material.encode()).hexdigest()


def train(
    run_dir: Path,
    out_dir: Path,
    config: TrainConfig | None = None,
    *,
    name: str | None = None,
    log: Log = print,
) -> TrainResult:
    config = config or TrainConfig()
    examples, skipped = load_examples(run_dir)
    train_set = [ex for ex in examples if ex.split == "train"]
    val_set = [ex for ex in examples if ex.split == "val"]
    hint = "run `dfbench run --split train --split val --split test --save-traces` first"
    if {ex.label for ex in train_set} != {0, 1}:
        raise LabError(f"run {run_dir.name} has no train-split traces of both classes: {hint}")
    if not val_set:
        raise LabError(f"run {run_dir.name} has no val-split traces: {hint}")
    embedding_dim = int(train_set[0].main_track.embeddings.shape[1])
    hparams = TemporalHParams(
        embedding_dim=embedding_dim,
        arch=config.arch,
        hidden=config.hidden,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        max_len=max(64, config.window),
    )
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    net = TemporalNet(hparams).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_windows = _windows(train_set, config.window, config.stride)
    val_windows = _windows(val_set, config.window, config.stride)
    log(
        f"train: {len(train_set)} videos, {len(train_windows.y)} windows; "
        f"val: {len(val_set)} videos; skipped {skipped} records"
    )

    def validate() -> tuple[float | None, float]:
        net.eval()
        scores = video_scores(val_set, lambda t: _track_score(net, config, t))
        return _auc(val_set, scores), _weighted_bce(net, val_windows, device)

    history: list[dict[str, float | None]] = []
    val_auc, val_loss = validate()
    history.append({"epoch": 0, "train_loss": None, "val_loss": val_loss, "val_auc": val_auc})
    baseline_auc = val_auc

    def key(auc: float | None, loss: float) -> tuple[float, float]:
        return (auc if auc is not None else -1.0, -loss)

    best_key, best_epoch = key(val_auc, val_loss), 0
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    stale = 0
    n = len(train_windows.y)
    for epoch in range(1, config.epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n))
        total = 0.0
        for start in range(0, n, config.batch_size):
            batch = order[start : start + config.batch_size]
            weight = train_windows.weight[batch].to(device)
            logits = net(train_windows.x[batch].to(device), train_windows.mask[batch].to(device))
            losses = F.binary_cross_entropy_with_logits(
                logits, train_windows.y[batch].to(device), reduction="none"
            )
            loss = (losses * weight).sum() / weight.sum()
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float((losses.detach() * weight).sum())
        train_loss = total / float(train_windows.weight.sum())
        val_auc, val_loss = validate()
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}
        )
        log(f"epoch {epoch:3}: train {train_loss:.4f}  val {val_loss:.4f}  val AUC {val_auc}")
        if key(val_auc, val_loss) > best_key:
            best_key, best_epoch, stale = key(val_auc, val_loss), epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    net.load_state_dict(best_state)

    run = json.loads((run_dir / "run.json").read_text())
    name = name or f"{config.arch}-{run['run_id']}-s{config.seed}"
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    meta = CheckpointMeta(
        hparams=hparams,
        window=config.window,
        stride=config.stride,
        embedding_model_sha256=frame_model_sha256(run_dir),
        name=name,
        trained_at=trained_at,
        training_set_sha256=_training_set_sha256(train_set),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{name}.safetensors"
    checkpoint_sha = save_checkpoint(net.cpu(), meta, checkpoint)
    record = out_dir / f"{name}.json"
    record.write_text(
        json.dumps(
            {
                "name": name,
                "created_at": trained_at,
                "checkpoint_sha256": checkpoint_sha,
                "run_id": run["run_id"],
                "pipeline_version": run.get("pipeline_version"),
                "config_hash": run.get("config_hash"),
                "embedding_model_sha256": meta.embedding_model_sha256,
                "train_config": asdict(config),
                "hparams": hparams.to_dict(),
                "train_samples": sorted(ex.sample_id for ex in train_set),
                "val_samples": sorted(ex.sample_id for ex in val_set),
                "identities": sorted({i for ex in train_set + val_set for i in ex.identities}),
                "best_epoch": best_epoch,
                "val_auc": history[best_epoch]["val_auc"],
                "baseline_val_auc": baseline_auc,
                "history": history,
            },
            indent=2,
        )
    )
    return TrainResult(
        name=name,
        checkpoint=checkpoint,
        record=record,
        best_epoch=best_epoch,
        val_auc=history[best_epoch]["val_auc"],
        baseline_val_auc=baseline_auc,
        history=history,
    )


def _track_score(net: TemporalNet, config: TrainConfig, track: TraceTrack) -> float | None:
    scored = score_sequence(
        net, config.window, config.stride, track.times, track.p_fake, track.embeddings
    )
    return scored.score if scored is not None else None


# --------------------------------------------------------------------------- evaluation


def baseline_score(track: TraceTrack, method: str, window_s: float = 4.0) -> float | None:
    scored = np.isfinite(track.p_fake)
    values = track.p_fake[scored].astype(np.float64)
    if values.size == 0:
        return None
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(np.median(values))
    times = track.times[scored].astype(np.float64)
    return window_max_mean(times.tolist(), values.tolist(), window_s)


def _metrics(rows: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    kept = [r for r in rows if np.isfinite(r["scores"][method])]
    labels = np.array([r["label"] for r in kept], np.float64)
    scores = np.array([r["scores"][method] for r in kept], np.float64)
    at1 = m.tpr_at_fpr(labels, scores, 0.01) if kept else None
    at5 = m.tpr_at_fpr(labels, scores, 0.05) if kept else None
    pairs = score_pairs(
        [
            {
                "pair_id": r["pair_id"],
                "transformation": r["transformation"],
                "label": "manipulated" if r["label"] else "real",
                "score": r["scores"][method],
            }
            for r in kept
        ]
    )
    return {
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "roc_auc": _round(m.roc_auc(labels, scores)) if kept else None,
        "tpr_at_fpr_1pct": _round(at1[0]) if at1 else None,
        "tpr_at_fpr_5pct": _round(at5[0]) if at5 else None,
        "paired_win_rate": _round(m.paired_win_rate(pairs)),
    }


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _auc_delta_ci(
    rows: Sequence[dict[str, Any]], bootstrap: int, seed: int
) -> tuple[float | None, tuple[float, float] | None]:
    """AUC(temporal) - AUC(mean) on the rows both score, with a pair-level bootstrap CI."""
    kept = [
        r for r in rows if np.isfinite(r["scores"]["temporal"]) and np.isfinite(r["scores"]["mean"])
    ]
    labels = np.array([r["label"] for r in kept], np.float64)
    temporal = np.array([r["scores"]["temporal"] for r in kept], np.float64)
    mean = np.array([r["scores"]["mean"] for r in kept], np.float64)

    def delta(idx: npt.NDArray[np.int64]) -> float | None:
        a, b = m.roc_auc(labels[idx], temporal[idx]), m.roc_auc(labels[idx], mean[idx])
        return None if a is None or b is None else a - b

    point = delta(np.arange(len(kept))) if kept else None
    ci = (
        m.bootstrap_ci([r["pair_id"] for r in kept], delta, iterations=bootstrap, seed=seed)
        if bootstrap and kept
        else None
    )
    return point, ci


def evaluate(
    run_dir: Path,
    checkpoint: Path,
    *,
    bootstrap: int = 1000,
    seed: int = 0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Temporal model vs the frame-score baselines on every split except train."""
    scorer = TemporalScorer(checkpoint)
    if scorer.embedding_model_sha256 != frame_model_sha256(run_dir):
        raise LabError("the checkpoint was trained on embeddings of a different frame model")
    record_path = checkpoint.with_suffix(".json")
    training = json.loads(record_path.read_text()) if record_path.is_file() else {}
    seen_samples = set(training.get("train_samples", [])) | set(training.get("val_samples", []))
    seen_identities = set(training.get("identities", []))

    examples, skipped = load_examples(run_dir)
    rows: list[dict[str, Any]] = []
    excluded = 0
    for ex in examples:
        if ex.split == "train":
            continue
        if ex.split != "val" and (ex.sample_id in seen_samples or ex.identities & seen_identities):
            excluded += 1
            continue
        scores = {
            name: video_scores([ex], partial(baseline_score, method=name))[0] for name in BASELINES
        }

        def temporal_score(track: TraceTrack) -> float | None:
            scored = scorer.score(track.times, track.p_fake, track.embeddings)
            return scored.score if scored is not None else None

        scores["temporal"] = video_scores([ex], temporal_score)[0]
        rows.append(
            {
                "sample_id": ex.sample_id,
                "pair_id": ex.pair_id,
                "label": ex.label,
                "split": ex.split,
                "transformation": ex.transformation,
                "scores": scores,
            }
        )

    methods = (*BASELINES, "temporal")
    splits: dict[str, Any] = {}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        point, ci = _auc_delta_ci(subset, bootstrap, seed)
        by_transformation = {}
        for transformation in sorted({r["transformation"] for r in subset}):
            part = [r for r in subset if r["transformation"] == transformation]
            by_transformation[transformation] = {
                name: _metrics(part, name)["roc_auc"] for name in ("mean", "temporal")
            }
        splits[split] = {
            "aggregators": {name: _metrics(subset, name) for name in methods},
            "auc_delta_vs_mean": _round(point),
            "auc_delta_ci95": [round(v, 4) for v in ci] if ci else None,
            "auc_by_transformation": by_transformation,
        }

    report: dict[str, Any] = {
        "checkpoint": {"name": scorer.meta.name, "sha256": scorer.sha256, "path": str(checkpoint)},
        "hparams": scorer.meta.hparams.to_dict(),
        "window": scorer.meta.window,
        "stride": scorer.meta.stride,
        "run_id": run_dir.name,
        "skipped_records": skipped,
        "excluded_for_overlap": excluded,
        "training": {
            k: training.get(k) for k in ("run_id", "best_epoch", "val_auc", "baseline_val_auc")
        },
        "splits": splits,
    }
    out = out_dir or run_dir / "temporal" / scorer.meta.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(_markdown(report))
    return report


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Temporal model `{report['checkpoint']['name']}` on run `{report['run_id']}`",
        "",
        f"- Architecture: {report['hparams']['arch']}, window {report['window']} samples, "
        f"stride {report['stride']}",
        f"- Checkpoint sha256: `{report['checkpoint']['sha256'][:16]}…`",
        f"- Training: best epoch {report['training'].get('best_epoch')}, val AUC "
        f"{_fmt(report['training'].get('val_auc'))} (untrained baseline "
        f"{_fmt(report['training'].get('baseline_val_auc'))})",
        f"- Records skipped (failed, no trace or no eligible track): "
        f"{report['skipped_records']}; excluded for overlap with the training set: "
        f"{report['excluded_for_overlap']}",
        "",
        "The val split chose the training epoch, so val numbers are optimistic for the "
        "temporal model. Use test for the comparison. All scores are uncalibrated (step 16).",
    ]
    for split, data in report["splits"].items():
        lines += [
            "",
            f"## {split}",
            "",
            "| Aggregator | n real | n fake | ROC-AUC | TPR@1%FPR | TPR@5%FPR | Paired win |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in data["aggregators"].items():
            lines.append(
                f"| {name} | {row['n_real']} | {row['n_fake']} | {_fmt(row['roc_auc'])} | "
                f"{_fmt(row['tpr_at_fpr_1pct'])} | {_fmt(row['tpr_at_fpr_5pct'])} | "
                f"{_fmt(row['paired_win_rate'])} |"
            )
        ci = data["auc_delta_ci95"]
        ci_text = f" (95% CI {ci[0]:+.3f} to {ci[1]:+.3f})" if ci else " (no CI)"
        delta = data["auc_delta_vs_mean"]
        delta_text = "n/a" if delta is None else f"{delta:+.3f}"
        lines += ["", f"AUC(temporal) - AUC(mean): {delta_text}{ci_text}", ""]
        lines += ["| Transformation | AUC mean | AUC temporal |", "|---|---:|---:|"]
        for transformation, aucs in data["auc_by_transformation"].items():
            lines.append(f"| {transformation} | {_fmt(aucs['mean'])} | {_fmt(aucs['temporal'])} |")
    return "\n".join(lines) + "\n"
