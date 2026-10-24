"""`dfbench`: the attack lab and benchmark command line.

dfbench identity add ID --consent-ref REF     register a consenting person (pseudonymous)
dfbench identity list | delete ID --yes       list, or withdraw consent (deletes everything)
dfbench source add VIDEO --identity ID        register a consented source video
dfbench split                                 assign identities to train/val/test
dfbench generate [--fake-facefusion]          real + FaceFusion samples (pristine)
dfbench import CSV --dataset NAME             evaluation-only external dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deepfake_bench.facefusion import FaceFusion, FaceFusionError
from deepfake_bench.lab import (
    GenerationConfig,
    LabError,
    add_identity,
    add_source,
    assign_identity_splits,
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
            f"{report.files} files, {report.predictions} predictions"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "identity" and args.action in {"add", "delete"} and not args.identity_id:
        parser.error("identity add/delete needs an identity ID")
    if args.command == "source" and args.action == "add" and not (args.video and args.identity):
        parser.error("source add needs VIDEO and --identity")
    try:
        result: int = args.handler(args)
        return result
    except (LabError, FaceFusionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
