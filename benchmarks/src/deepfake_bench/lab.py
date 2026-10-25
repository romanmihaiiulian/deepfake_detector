"""Lab operations: identities, sources, splits, generation, external data, consent withdrawal."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from deepfake_bench.coverage import measure_coverage
from deepfake_bench.donors import pick_donor_still
from deepfake_bench.facefusion import FaceFusion, SwapSettings
from deepfake_bench.manifest import Identity, Registry, Sample, SourceVideo, Split, utcnow
from deepfake_bench.media import Encoded, canonicalize, sha256_of
from deepfake_bench.splits import assign_splits, leakage_violations
from deepfake_bench.variants import PROFILES, ROBUSTNESS_PROFILES, apply_profile
from deepfake_core.faces import FaceDetector
from deepfake_core.video import validate

Log = Callable[[str], None]


class LabError(RuntimeError):
    pass


# --------------------------------------------------------------------------- configuration


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SwapperSpec(_Config):
    model: str
    pixel_boost: str | None = None
    weight: float | None = None
    face_mask_types: list[str] = Field(default_factory=lambda: ["box"])
    face_enhancer_model: str | None = None

    def settings(self) -> SwapSettings:
        return SwapSettings(
            swapper_model=self.model,
            pixel_boost=self.pixel_boost,
            weight=self.weight,
            face_mask_types=tuple(self.face_mask_types),
            face_enhancer_model=self.face_enhancer_model,
        )


class CanonicalConfig(_Config):
    max_seconds: float = 30.0
    max_fps: float = 30.0
    max_height: int = 1080


def _default_ratios() -> dict[Split, float]:
    return {"train": 0.6, "val": 0.2, "test": 0.2}


class SplitConfig(_Config):
    seed: int = 20261024
    ratios: dict[Split, float] = Field(default_factory=_default_ratios)


class CoverageConfig(_Config):
    min_fraction: float = 0.8
    sample_hz: float = 4.0
    diff_threshold: float = 6.0
    max_frame_count_delta: int = 2


class GenerationConfig(_Config):
    canonical: CanonicalConfig = CanonicalConfig()
    splits: SplitConfig = SplitConfig()
    coverage: CoverageConfig = CoverageConfig()
    donors_per_target: int = 2
    swappers: list[SwapperSpec]
    held_out_for_test: list[str] = Field(
        default_factory=list, description="Swapper models used only for the test split"
    )

    @classmethod
    def from_file(cls, path: str | Path) -> GenerationConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


# --------------------------------------------------------------------------- registration


def add_identity(
    registry: Registry, identity_id: str, consent_ref: str, notes: str = ""
) -> Identity:
    if identity_id in registry.identities:
        raise LabError(f"identity {identity_id} already exists")
    identity = Identity(identity_id=identity_id, consent_ref=consent_ref, notes=notes)
    registry.identities[identity_id] = identity
    registry.save()
    return identity


def add_source(registry: Registry, path: Path, identity_id: str) -> SourceVideo:
    """Copy a consented video into the lab. The copy is named by content hash."""
    identity = registry.identities.get(identity_id)
    if identity is None:
        raise LabError(f"unknown identity {identity_id}: register it (with consent) first")
    info = validate(path)
    digest = sha256_of(path)
    video_id = digest[:16]
    if video_id in registry.sources:
        return registry.sources[video_id]
    dest = registry.paths.sources / f"{video_id}{path.suffix.lower()}"
    shutil.copyfile(path, dest)
    source = SourceVideo(
        video_id=video_id,
        identity_id=identity_id,
        path=registry.paths.relative(dest),
        sha256=digest,
        consent_ref=identity.consent_ref,
        width=info.display_width,
        height=info.display_height,
        fps=info.fps,
        duration=info.duration,
    )
    registry.sources[video_id] = source
    registry.save()
    return source


def assign_identity_splits(
    registry: Registry, config: SplitConfig, *, reassign: bool = False
) -> dict[str, Split]:
    existing = {} if reassign else {i: v.split for i, v in registry.identities.items() if v.split}
    assignment = assign_splits(registry.identities, config.ratios, config.seed, existing)
    for identity_id, split in assignment.items():
        registry.identities[identity_id] = registry.identities[identity_id].model_copy(
            update={"split": split}
        )
    registry.save()
    return assignment


# --------------------------------------------------------------------------- generation


def make_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Job:
    sample_id: str
    split: Split
    source: SourceVideo
    donor: str | None = None
    swapper: SwapperSpec | None = None

    def describe(self) -> str:
        if self.donor is None:
            return f"{self.sample_id} real   video={self.source.video_id} split={self.split}"
        assert self.swapper is not None
        return (
            f"{self.sample_id} fake   video={self.source.video_id} donor={self.donor} "
            f"swapper={self.swapper.model} split={self.split}"
        )


@dataclass(slots=True)
class GenerationReport:
    created: list[str] = field(default_factory=list)
    skipped: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)


def plan(registry: Registry, config: GenerationConfig) -> list[Job]:
    """Every real and fake sample implied by the registry and config, deterministically."""
    jobs: list[Job] = []
    by_split: dict[Split, list[str]] = {}
    for identity in sorted(registry.identities.values(), key=lambda i: i.identity_id):
        if identity.split is None:
            raise LabError("identities have no split yet: run `dfbench split` first")
        by_split.setdefault(identity.split, []).append(identity.identity_id)
    for source in sorted(registry.sources.values(), key=lambda s: s.video_id):
        split = registry.identities[source.identity_id].split
        assert split is not None
        jobs.append(Job(make_id("real", source.video_id, "pristine"), split, source))
        donors = [
            i for i in by_split[split] if i != source.identity_id and _has_source(registry, i)
        ]
        donors.sort(key=lambda d: make_id(str(config.splits.seed), source.video_id, d))
        for donor in donors[: config.donors_per_target]:
            for swapper in config.swappers:
                if split != "test" and swapper.model in config.held_out_for_test:
                    continue
                settings = json.dumps(swapper.model_dump(), sort_keys=True)
                sample_id = make_id("fake", source.video_id, donor, settings, "pristine")
                jobs.append(Job(sample_id, split, source, donor, swapper))
    return jobs


def _has_source(registry: Registry, identity_id: str) -> bool:
    return any(s.identity_id == identity_id for s in registry.sources.values())


def _canonical(registry: Registry, source: SourceVideo, config: CanonicalConfig) -> Path:
    dest = registry.paths.canon / f"{source.video_id}.mp4"
    if not dest.is_file():
        canonicalize(
            registry.paths.absolute(source.path),
            dest,
            max_seconds=config.max_seconds,
            max_fps=config.max_fps,
            max_height=config.max_height,
        )
    return dest


def _donor_still(
    registry: Registry, identity_id: str, config: GenerationConfig, detector: FaceDetector
) -> Path:
    dest = registry.paths.donors / f"{identity_id}.png"
    if not dest.is_file():
        source = min(
            (s for s in registry.sources.values() if s.identity_id == identity_id),
            key=lambda s: s.video_id,
        )
        pick_donor_still(_canonical(registry, source, config.canonical), dest, detector)
    return dest


def _sample(job: Job, registry: Registry, encoded: Encoded, **fields: object) -> Sample:
    target = job.source.identity_id
    return Sample(
        sample_id=job.sample_id,
        pair_id=job.source.video_id,
        split=job.split,
        path=registry.paths.relative(encoded.path),
        sha256=encoded.sha256,
        source_video=job.source.video_id,
        target_identity=target,
        resolution=f"{encoded.width}x{encoded.height}",
        fps=encoded.fps,
        duration=encoded.duration,
        transformation="pristine",
        transformation_argv=_portable(encoded.argv),
        ffmpeg_version=encoded.ffmpeg_version,
        **fields,
    )


def _portable(argv: list[str]) -> list[str]:
    return [Path(a).name if a.startswith("/") else a for a in argv]


def generate(
    registry: Registry,
    config: GenerationConfig,
    facefusion: FaceFusion,
    detector: FaceDetector,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    log: Log = print,
) -> GenerationReport:
    report = GenerationReport()
    for job in plan(registry, config):
        if job.sample_id in registry.samples:
            report.skipped += 1
            continue
        if dry_run:
            report.planned.append(job.describe())
            continue
        if limit is not None and len(report.created) + len(report.rejected) >= limit:
            break
        log(f"generate {job.describe()}")
        canon = _canonical(registry, job.source, config.canonical)
        out = registry.paths.samples / job.sample_id / "pristine.mp4"
        if job.donor is None:
            encoded = apply_profile(PROFILES["pristine"], canon, out)
            sample = _sample(job, registry, encoded, label="real", generator="none")
        else:
            assert job.swapper is not None
            settings = job.swapper.settings()
            raw = registry.paths.raw_fakes / f"{job.sample_id}.mp4"
            donor = _donor_still(registry, job.donor, config, detector)
            run = facefusion.swap(donor, canon, raw, settings)
            coverage = measure_coverage(
                canon,
                raw,
                detector,
                sample_hz=config.coverage.sample_hz,
                diff_threshold=config.coverage.diff_threshold,
            )
            common = {
                "label": "manipulated",
                "generator": "facefusion",
                "generator_version": run.version,
                "swapper_model": settings.swapper_model,
                "swapper_license": settings.license,
                "generator_settings": settings.as_dict(),
                "generator_argv": run.argv,
                "source_identity": job.donor,
                "manipulation_coverage": round(coverage.fraction, 4),
            }
            reason = None
            if abs(coverage.frame_count_delta) > config.coverage.max_frame_count_delta:
                reason = f"frame count differs from the source by {coverage.frame_count_delta}"
            elif coverage.fraction < config.coverage.min_fraction:
                reason = (
                    f"face swapped in {coverage.fraction:.0%} of sampled frames "
                    f"(minimum {config.coverage.min_fraction:.0%})"
                )
            if reason:
                rejected = Sample(
                    sample_id=job.sample_id,
                    pair_id=job.source.video_id,
                    split=job.split,
                    path=registry.paths.relative(raw),
                    sha256=sha256_of(raw),
                    source_video=job.source.video_id,
                    target_identity=job.source.identity_id,
                    resolution="",
                    fps=None,
                    duration=None,
                    status="rejected",
                    status_reason=reason,
                    **common,
                )
                registry.samples[job.sample_id] = rejected
                registry.save()
                report.rejected.append((job.sample_id, reason))
                log(f"rejected {job.sample_id}: {reason}")
                continue
            encoded = apply_profile(PROFILES["pristine"], raw, out)
            sample = _sample(job, registry, encoded, **common)
        registry.samples[job.sample_id] = sample
        registry.save()
        report.created.append(job.sample_id)
    violations = leakage_violations(registry.ok_samples())
    if violations:
        raise LabError("split leakage detected: " + "; ".join(violations))
    return report


# --------------------------------------------------------------------------- robustness variants


@dataclass(slots=True)
class VariantReport:
    created: list[str] = field(default_factory=list)
    skipped: int = 0
    missing_intermediate: list[str] = field(default_factory=list)


def intermediate_for(registry: Registry, sample: Sample) -> Path:
    """The near-lossless file a sample's pristine encode was made from."""
    if sample.label == "real":
        return registry.paths.canon / f"{sample.source_video}.mp4"
    return registry.paths.raw_fakes / f"{sample.sample_id}.mp4"


def create_variants(
    registry: Registry,
    profiles: Sequence[str] = ROBUSTNESS_PROFILES,
    *,
    splits: Sequence[Split] = ("val", "test"),
    log: Log = print,
) -> VariantReport:
    """Re-encode every pristine lab sample with each robustness profile.

    Both members of a pair start from their near-lossless intermediates and receive
    identical recipes, so the degradation never becomes a label shortcut.
    """
    unknown = [name for name in profiles if name not in PROFILES]
    if unknown:
        raise LabError(f"unknown profiles {unknown}; choose from {sorted(PROFILES)}")
    report = VariantReport()
    parents = sorted(
        (
            s
            for s in registry.ok_samples()
            if s.transformation == "pristine" and s.parent_sample_id is None and s.split in splits
        ),
        key=lambda s: s.sample_id,
    )
    for parent in parents:
        source = intermediate_for(registry, parent)
        if not source.is_file():
            report.missing_intermediate.append(parent.sample_id)
            log(f"skip {parent.sample_id}: intermediate {source.name} is missing")
            continue
        for name in profiles:
            sample_id = make_id("variant", parent.sample_id, name)
            if sample_id in registry.samples:
                report.skipped += 1
                continue
            log(f"variant {name:16} {parent.sample_id} ({parent.label})")
            encoded = apply_profile(
                PROFILES[name], source, registry.paths.samples / parent.sample_id / f"{name}.mp4"
            )
            registry.samples[sample_id] = parent.model_copy(
                update={
                    "sample_id": sample_id,
                    "path": registry.paths.relative(encoded.path),
                    "sha256": encoded.sha256,
                    "resolution": f"{encoded.width}x{encoded.height}",
                    "fps": encoded.fps,
                    "duration": encoded.duration,
                    "transformation": name,
                    "transformation_argv": _portable(encoded.argv),
                    "ffmpeg_version": encoded.ffmpeg_version,
                    "parent_sample_id": parent.sample_id,
                    "created_at": utcnow(),
                }
            )
            registry.save()
            report.created.append(sample_id)
    return report


# --------------------------------------------------------------------------- external data


def import_external(registry: Registry, csv_path: Path, dataset: str) -> list[str]:
    """Evaluation-only samples from a dataset you obtained yourself (e.g. Celeb-DF v2).

    CSV columns: path, label (real|manipulated), and optionally video_id and identity.
    Files stay where they are. Samples are marked research-only, split 'external'.
    """
    added = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"]).expanduser()
            if not path.is_absolute():
                path = (csv_path.parent / path).resolve()
            label = row["label"].strip().lower()
            if label not in {"real", "manipulated"}:
                raise LabError(f"{path.name}: label must be real or manipulated, got {label!r}")
            info = validate(path)
            digest = sha256_of(path)
            sample_id = make_id("external", dataset, digest)
            if sample_id in registry.samples:
                continue
            video_id = row.get("video_id") or f"{dataset}:{path.stem}"
            registry.samples[sample_id] = Sample(
                sample_id=sample_id,
                pair_id=video_id,
                label=label,
                split="external",
                path=str(path),
                sha256=digest,
                generator=dataset if label == "manipulated" else "none",
                source_video=video_id,
                target_identity=row.get("identity") or None,
                generator_settings={"dataset": dataset},
                resolution=f"{info.display_width}x{info.display_height}",
                fps=info.fps,
                duration=info.duration,
                research_only=True,
            )
            added.append(sample_id)
    registry.save()
    return added


# --------------------------------------------------------------------------- consent withdrawal


@dataclass(slots=True)
class PurgeReport:
    sources: int = 0
    samples: int = 0
    files: int = 0
    predictions: int = 0
    temporal_models: int = 0


def delete_identity(registry: Registry, identity_id: str) -> PurgeReport:
    """Remove an identity and everything derived from it: sources, intermediates, samples
    and their variants, donor stills, benchmark predictions and traces, and temporal
    models trained or validated on it."""
    if identity_id not in registry.identities:
        raise LabError(f"unknown identity {identity_id}")
    report = PurgeReport()
    paths = registry.paths
    videos = {v for v, s in registry.sources.items() if s.identity_id == identity_id}
    doomed = {
        sid
        for sid, s in registry.samples.items()
        if identity_id in (s.source_identity, s.target_identity) or s.source_video in videos
    }

    def remove(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
            report.files += 1
        elif path.exists():
            path.unlink()
            report.files += 1

    for video_id in videos:
        remove(paths.absolute(registry.sources[video_id].path))
        remove(paths.canon / f"{video_id}.mp4")
        del registry.sources[video_id]
        report.sources += 1
    remove(paths.donors / f"{identity_id}.png")
    for sample_id in doomed:
        remove(paths.samples / sample_id)
        remove(paths.raw_fakes / f"{sample_id}.mp4")
        del registry.samples[sample_id]
        report.samples += 1
    for run in paths.runs.glob("*"):
        predictions = run / "predictions.jsonl"
        if predictions.is_file():
            lines = predictions.read_text().splitlines()
            kept = [line for line in lines if json.loads(line).get("sample_id") not in doomed]
            report.predictions += len(lines) - len(kept)
            predictions.write_text("".join(f"{line}\n" for line in kept))
        for sample_id in doomed:
            remove(run / "traces" / f"{sample_id}.npz")
    for record in paths.temporal.glob("*.json"):
        if identity_id in json.loads(record.read_text()).get("identities", []):
            remove(record.with_suffix(".safetensors"))
            remove(record)
            report.temporal_models += 1
    del registry.identities[identity_id]
    registry.save()
    return report
