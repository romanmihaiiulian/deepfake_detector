"""Run the analyzer over a selection of lab samples; one JSON line per sample.

Predictions are cached across runs by (sample sha256, pipeline version, config hash,
model sha256s). Adding samples or re-running a report therefore never re-analyses
anything that is unchanged. With `save_traces`, per-frame scores and embeddings are kept
as .npz: the training input of the temporal model (roadmap step 14).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from deepfake_bench.manifest import Registry, Sample
from deepfake_core import __version__ as core_version
from deepfake_core.errors import DeepfakeCoreError
from deepfake_core.pipeline import AnalysisOutput
from deepfake_core.schemas import ModelInfo
from deepfake_core.scoring.aggregate import window_max_mean

Log = Callable[[str], None]


class AnalyzerLike(Protocol):
    models: list[ModelInfo]

    def analyze(self, path: Path, *, analysis_id: str | None = None) -> AnalysisOutput: ...


@dataclass(frozen=True, slots=True)
class Selection:
    splits: tuple[str, ...] = ("val", "test")
    transformations: tuple[str, ...] | None = None
    """None means every transformation (pristine and all variants)."""

    def matches(self, sample: Sample) -> bool:
        return sample.split in self.splits and (
            self.transformations is None or sample.transformation in self.transformations
        )


@dataclass(slots=True)
class RunSummary:
    run_dir: Path
    analysed: int = 0
    cached: int = 0
    failed: list[str] = field(default_factory=list)


def pipeline_identity(analyzer: Any) -> dict[str, Any]:
    config = getattr(analyzer, "config", None)
    verdict = getattr(config, "verdict", None)
    return {
        "pipeline_version": core_version,
        "config_hash": config.config_hash() if config is not None else "unknown",
        "config_profile": getattr(config, "profile", "unknown"),
        "operating_threshold": getattr(verdict, "manipulated_threshold", 0.8),
        "models": [m.model_dump() for m in analyzer.models],
    }


def cache_key(sample: Sample, identity: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "sample": sample.sha256,
            "pipeline": identity["pipeline_version"],
            "config": identity["config_hash"],
            "models": sorted(m["sha256"] for m in identity["models"]),
        },
        sort_keys=True,
    )
    return sha256(material.encode()).hexdigest()[:24]


def aggregates(output: AnalysisOutput, window_s: float = 4.0) -> dict[str, float | None]:
    """Video-level score under each candidate aggregator (max over eligible tracks)."""
    eligible = {t.track_id for t in output.result.tracks if t.eligible}
    best: dict[str, float | None] = {"mean": None, "median": None, "window_max": None}
    for track in output.trace.tracks:
        if track.track_id not in eligible:
            continue
        scored = ~np.isnan(track.p_fake)
        values, times = track.p_fake[scored].astype(float), track.times[scored].astype(float)
        if values.size == 0:
            continue
        candidates = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "window_max": window_max_mean(times.tolist(), values.tolist(), window_s),
        }
        for name, value in candidates.items():
            current = best[name]
            if value is not None and (current is None or value > current):
                best[name] = value
    return best


def prediction_record(
    sample: Sample, key: str, output: AnalysisOutput, seconds: float
) -> dict[str, Any]:
    result = output.result
    return {
        "sample_id": sample.sample_id,
        "key": key,
        "label": sample.label,
        "split": sample.split,
        "pair_id": sample.pair_id,
        "generator": sample.generator,
        "swapper_model": sample.swapper_model,
        "transformation": sample.transformation,
        "resolution": sample.resolution,
        "source_identity": sample.source_identity,
        "target_identity": sample.target_identity,
        "dataset": dataset_of(sample),
        "score": result.manipulation_probability,
        "scores": aggregates(output),
        "verdict": result.verdict,
        "confidence": result.confidence,
        "video_quality": result.video_quality,
        "face_tracks": result.face_tracks,
        "frames_scored": result.processing.frames_scored,
        "seconds": round(seconds, 2),
        "error": None,
    }


def run_benchmark(
    registry: Registry,
    analyzer: AnalyzerLike,
    run_id: str,
    selection: Selection,
    *,
    held_out_swappers: Sequence[str] = (),
    save_traces: bool = False,
    log: Log = print,
) -> RunSummary:
    run_dir = registry.paths.runs / run_id
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    cache_dir = registry.paths.runs.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = pipeline_identity(analyzer)
    samples = sorted(
        (s for s in registry.ok_samples() if selection.matches(s)), key=lambda s: s.sample_id
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "selection": {
                    "splits": list(selection.splits),
                    "transformations": list(selection.transformations or []),
                },
                "held_out_swappers": list(held_out_swappers),
                "samples": len(samples),
                **identity,
            },
            indent=2,
        )
    )
    summary = RunSummary(run_dir)
    records = []
    for index, sample in enumerate(samples, 1):
        key = cache_key(sample, identity)
        cached = cache_dir / f"{key}.json"
        trace = run_dir / "traces" / f"{sample.sample_id}.npz"
        if cached.is_file() and (not save_traces or trace.is_file()):
            records.append(
                {
                    **json.loads(cached.read_text()),
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                }
            )
            summary.cached += 1
            continue
        log(f"[{index}/{len(samples)}] {sample.sample_id} {sample.label} {sample.transformation}")
        start = time.perf_counter()
        try:
            output = analyzer.analyze(
                registry.paths.absolute(sample.path), analysis_id=sample.sample_id
            )
        except DeepfakeCoreError as error:
            records.append(_failure(sample, key, error))
            summary.failed.append(sample.sample_id)
            continue
        record = prediction_record(sample, key, output, time.perf_counter() - start)
        cached.write_text(json.dumps(record))
        if save_traces:
            output.trace.save_npz(trace)
        records.append(record)
        summary.analysed += 1
    _write_lines(run_dir / "predictions.jsonl", records)
    return summary


def dataset_of(sample: Sample) -> str:
    """'lab' for our own samples; the dataset name for imported external samples."""
    return str(sample.generator_settings.get("dataset", "lab"))


def _failure(sample: Sample, key: str, error: Exception) -> dict[str, Any]:
    return {
        "dataset": dataset_of(sample),
        "sample_id": sample.sample_id,
        "key": key,
        "label": sample.label,
        "split": sample.split,
        "pair_id": sample.pair_id,
        "generator": sample.generator,
        "swapper_model": sample.swapper_model,
        "transformation": sample.transformation,
        "resolution": sample.resolution,
        "score": None,
        "scores": {},
        "verdict": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _write_lines(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]
