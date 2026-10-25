"""`dfbench`: the attack lab and benchmark command line.

dfbench identity add ID --consent-ref REF     register a consenting person (pseudonymous)
dfbench identity list | delete ID --yes       list, or withdraw consent (deletes everything)
dfbench source add VIDEO --identity ID        register a consented source video
dfbench split                                 assign identities to train/val/test
dfbench generate [--fake-facefusion]          real + FaceFusion samples (pristine)
dfbench import CSV --dataset NAME             evaluation-only external dataset
dfbench variants | run | report               robustness variants, benchmark runs, reports
dfbench temporal train RUN | eval RUN --checkpoint NAME   temporal model (step 14)
dfbench calibrate RUN [--method platt|isotonic]           calibration profile (step 16)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from deepfake_bench.facefusion import FaceFusion, FaceFusionError
from deepfake_bench.lab import (
    GenerationConfig,
    LabError,
    add_identity,
    add_source,
    assign_identity_splits,
    create_variants,
    delete_identity,
    generate,
    import_external,
    plan,
)
from deepfake_bench.manifest import Registry
from deepfake_bench.paths import DATA_ROOT_ENV, LabPaths

if TYPE_CHECKING:
    from deepfake_core.faces import FaceDetector

REPO = Path(__file__).resolve().parents[3]
DEFAULT_GENERATION = REPO / "benchmarks" / "configs" / "generation.yaml"


def _registry(args: argparse.Namespace) -> Registry:
    return Registry(LabPaths.resolve(args.data_root))


def _detector() -> FaceDetector:
    from deepfake_core.faces import YuNetDetector

    return YuNetDetector()


def cmd_identity(args: argparse.Namespace) -> int:
    registry = _registry(args)
    if args.action == "add":
        identity = add_identity(registry, args.identity_id, args.consent_ref, args.notes or "")
        print(f"registered {identity.identity_id} (consent {identity.consent_ref})")
    elif args.action == "list":
        for identity in sorted(registry.identities.values(), key=lambda i: i.identity_id):
            videos = sum(
                1 for s in registry.sources.values() if s.identity_id == identity.identity_id
            )
            split = identity.split or "-"
            print(
                f"{identity.identity_id:24} split={split:5} videos={videos} "
                f"consent={identity.consent_ref}"
            )
    else:
        if not args.yes:
            print("refusing to delete without --yes", file=sys.stderr)
            return 2
        report = delete_identity(registry, args.identity_id)
        print(
            f"deleted {args.identity_id}: {report.sources} sources, {report.samples} samples, "
            f"{report.files} files, {report.predictions} predictions, "
            f"{report.temporal_models} temporal models"
        )
    return 0


def cmd_source(args: argparse.Namespace) -> int:
    registry = _registry(args)
    if args.action == "add":
        source = add_source(registry, Path(args.video), args.identity)
        print(f"registered source {source.video_id} for {source.identity_id}")
    else:
        for source in sorted(registry.sources.values(), key=lambda s: s.video_id):
            size = f"{source.width}x{source.height}"
            print(f"{source.video_id} {source.identity_id:24} {size} {source.duration or 0:.1f}s")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    registry = _registry(args)
    config = GenerationConfig.from_file(args.config)
    assignment = assign_identity_splits(registry, config.splits, reassign=args.reassign)
    for split in ("train", "val", "test"):
        members = sorted(i for i, s in assignment.items() if s == split)
        print(f"{split:5} {len(members):3}  {' '.join(members)}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    registry = _registry(args)
    config = GenerationConfig.from_file(args.config)
    if args.dry_run:
        jobs = [job for job in plan(registry, config) if job.sample_id not in registry.samples]
        for job in jobs:
            print(f"planned  {job.describe()}")
        print(f"planned {len(jobs)}, already present {len(registry.samples)}")
        return 0
    runner = (
        FaceFusion.fake() if args.fake_facefusion else FaceFusion.from_home(args.facefusion_home)
    )
    print(f"FaceFusion: {runner.version()}")
    report = generate(registry, config, runner, _detector(), limit=args.limit)
    print(
        f"created {len(report.created)}, rejected {len(report.rejected)}, "
        f"already present {report.skipped}"
    )
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    added = import_external(_registry(args), Path(args.csv), args.dataset)
    print(f"imported {len(added)} samples from {args.dataset} (research-only, split 'external')")
    return 0


def cmd_variants(args: argparse.Namespace) -> int:
    from deepfake_bench.variants import PROFILES, ROBUSTNESS_PROFILES

    if args.list:
        for name, profile in PROFILES.items():
            print(f"{name:17} {profile.description}")
        return 0
    profiles = tuple(args.profile) if args.profile else ROBUSTNESS_PROFILES
    splits = tuple(args.split or ["val", "test"])
    report = create_variants(_registry(args), profiles, splits=splits)
    print(
        f"created {len(report.created)}, already present {report.skipped}, "
        f"missing intermediates {len(report.missing_intermediate)}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from deepfake_bench.run import Selection, run_benchmark
    from deepfake_core.config import PipelineConfig
    from deepfake_core.pipeline import Analyzer

    registry = _registry(args)
    generation = GenerationConfig.from_file(args.config)
    pipeline = PipelineConfig.from_file(args.pipeline_config) if args.pipeline_config else None
    splits = tuple(args.split or ["val", "test"]) + (("external",) if args.include_external else ())
    selection = Selection(splits, tuple(args.transformation) if args.transformation else None)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = run_benchmark(
        registry,
        Analyzer(pipeline, device=args.device),
        run_id,
        selection,
        held_out_swappers=generation.held_out_for_test,
        save_traces=args.save_traces,
    )
    print(
        f"run {run_id}: analysed {summary.analysed}, cached {summary.cached}, "
        f"failed {len(summary.failed)} -> {summary.run_dir}"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from deepfake_bench.report import build_report

    run_dir = LabPaths.resolve(args.data_root).runs / args.run_id
    if not (run_dir / "predictions.jsonl").is_file():
        raise LabError(f"no predictions in {run_dir}: run `dfbench run` first")
    build_report(run_dir, bootstrap=args.bootstrap)
    print(f"wrote {run_dir / 'report.md'} and {run_dir / 'report.json'}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from deepfake_bench.calibrate import run_calibration

    run_dir = LabPaths.resolve(args.data_root).runs / args.run_id
    if not (run_dir / "predictions.jsonl").is_file():
        raise LabError(f"no predictions in {run_dir}: run `dfbench run` first")
    result = run_calibration(
        run_dir,
        method=args.method,
        inputs=tuple(args.input) if args.input else None,
        prior=args.prior,
        target_fpr=args.target_fpr,
        target_cleared_fake=args.target_cleared_fake,
    )
    reliable = "" if result.report["reliable"] else " (unreliable: too few val videos)"
    print(f"wrote profile {result.config.profile} -> {result.profile}{reliable}")
    return 0


def cmd_temporal(args: argparse.Namespace) -> int:
    from deepfake_bench import temporal

    paths = LabPaths.resolve(args.data_root)
    run_dir = paths.runs / args.run_id
    if not (run_dir / "predictions.jsonl").is_file():
        raise LabError(f"no predictions in {run_dir}: run `dfbench run --save-traces` first")
    if args.action == "train":
        config = temporal.TrainConfig(
            arch=args.arch,
            window=args.window,
            stride=args.stride or max(1, args.window // 2),
            epochs=args.epochs,
            seed=args.seed,
            device=args.device or "cpu",
        )
        result = temporal.train(run_dir, paths.temporal, config, name=args.name)
        print(
            f"best epoch {result.best_epoch}: val AUC {result.val_auc} "
            f"(untrained baseline {result.baseline_val_auc}) -> {result.checkpoint}"
        )
        return 0
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        checkpoint = paths.temporal / f"{args.checkpoint}.safetensors"
    if not checkpoint.is_file():
        raise LabError(f"no temporal checkpoint {args.checkpoint}")
    report = temporal.evaluate(run_dir, checkpoint, bootstrap=args.bootstrap)
    out = run_dir / "temporal" / report["checkpoint"]["name"]
    print(f"wrote {out / 'report.md'} and {out / 'report.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root", help=f"research data root (default ${DATA_ROOT_ENV} or ./data)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    identity = commands.add_parser("identity", help="manage consenting identities")
    identity.add_argument("action", choices=["add", "list", "delete"])
    identity.add_argument("identity_id", nargs="?")
    identity.add_argument("--consent-ref", default="")
    identity.add_argument("--notes")
    identity.add_argument("--yes", action="store_true")
    identity.set_defaults(handler=cmd_identity)

    source = commands.add_parser("source", help="register consented source videos")
    source.add_argument("action", choices=["add", "list"])
    source.add_argument("video", nargs="?")
    source.add_argument("--identity")
    source.set_defaults(handler=cmd_source)

    split = commands.add_parser("split", help="assign identities to splits")
    split.add_argument("--config", default=str(DEFAULT_GENERATION))
    split.add_argument("--reassign", action="store_true")
    split.set_defaults(handler=cmd_split)

    gen = commands.add_parser("generate", help="generate real and FaceFusion samples")
    gen.add_argument("--config", default=str(DEFAULT_GENERATION))
    gen.add_argument("--facefusion-home", help="FaceFusion install ($FACEFUSION_HOME)")
    gen.add_argument("--fake-facefusion", action="store_true", help="use the test double")
    gen.add_argument("--limit", type=int)
    gen.add_argument("--dry-run", action="store_true")
    gen.set_defaults(handler=cmd_generate)

    imp = commands.add_parser("import", help="import an external evaluation dataset")
    imp.add_argument("csv")
    imp.add_argument("--dataset", required=True)
    imp.set_defaults(handler=cmd_import)

    variants = commands.add_parser("variants", help="create robustness variants of lab samples")
    variants.add_argument("--profile", action="append", help="default: every robustness profile")
    variants.add_argument("--split", action="append", choices=["train", "val", "test"])
    variants.add_argument("--list", action="store_true", help="list the profiles and exit")
    variants.set_defaults(handler=cmd_variants)

    run = commands.add_parser("run", help="analyse samples and save predictions")
    run.add_argument("--run-id")
    run.add_argument("--split", action="append", choices=["train", "val", "test"])
    run.add_argument("--transformation", action="append", help="default: all")
    run.add_argument("--include-external", action="store_true")
    run.add_argument("--save-traces", action="store_true", help="keep per-frame traces (step 14)")
    run.add_argument("--pipeline-config", help="PipelineConfig overrides (YAML)")
    run.add_argument("--device")
    run.add_argument("--config", default=str(DEFAULT_GENERATION))
    run.set_defaults(handler=cmd_run)

    report = commands.add_parser("report", help="build the report for a run")
    report.add_argument("run_id")
    report.add_argument("--bootstrap", type=int, default=1000)
    report.set_defaults(handler=cmd_report)

    calibrate = commands.add_parser("calibrate", help="fit calibration and thresholds on val")
    calibrate.add_argument("run_id")
    calibrate.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    calibrate.add_argument(
        "--input", action="append", choices=["gend", "temporal"], help="default: all in the run"
    )
    calibrate.add_argument("--prior", type=float, default=0.5, help="assumed P(manipulated)")
    calibrate.add_argument("--target-fpr", type=float, default=0.01)
    calibrate.add_argument("--target-cleared-fake", type=float, default=0.05)
    calibrate.set_defaults(handler=cmd_calibrate)

    temporal = commands.add_parser("temporal", help="train or evaluate the temporal model")
    temporal.add_argument("action", choices=["train", "eval"])
    temporal.add_argument("run_id", help="a run made with --save-traces over train/val/test")
    temporal.add_argument("--name", help="checkpoint name (default ARCH-RUN-sSEED)")
    temporal.add_argument("--arch", choices=["tcn", "transformer"], default="tcn")
    temporal.add_argument("--window", type=int, default=16, help="samples per window")
    temporal.add_argument("--stride", type=int, help="default: half the window")
    temporal.add_argument("--epochs", type=int, default=40)
    temporal.add_argument("--seed", type=int, default=0)
    temporal.add_argument("--device")
    temporal.add_argument("--checkpoint", help="eval: checkpoint name or .safetensors path")
    temporal.add_argument("--bootstrap", type=int, default=1000)
    temporal.set_defaults(handler=cmd_temporal)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "identity" and args.action in {"add", "delete"} and not args.identity_id:
        parser.error("identity add/delete needs an identity ID")
    if args.command == "source" and args.action == "add" and not (args.video and args.identity):
        parser.error("source add needs VIDEO and --identity")
    if args.command == "temporal" and args.action == "eval" and not args.checkpoint:
        parser.error("temporal eval needs --checkpoint")
    try:
        result: int = args.handler(args)
        return result
    except (LabError, FaceFusionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
