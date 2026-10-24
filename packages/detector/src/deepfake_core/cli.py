"""Command line: `deepfake-core analyze VIDEO` and `deepfake-core models fetch|verify|list`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deepfake_core import __version__, models
from deepfake_core.errors import ModelIntegrityError, ModelMissingError, VideoValidationError

EXIT_INVALID_VIDEO = 2
EXIT_MODELS = 3


def _progress(quiet: bool) -> object:
    last: list[tuple[str, int]] = [("", -1)]

    def report(stage: str, fraction: float) -> None:
        percent = int(fraction * 100)
        if quiet or (stage, percent // 5) == last[0]:
            return
        last[0] = (stage, percent // 5)
        print(f"\r{stage:<11} {percent:3d}%", end="", file=sys.stderr, flush=True)

    return report


def _summary(result: object) -> str:
    from deepfake_core.schemas import AnalysisResult

    assert isinstance(result, AnalysisResult)
    probability = (
        "n/a"
        if result.manipulation_probability is None
        else f"{result.manipulation_probability:.3f}"
    )
    lines = [
        f"verdict:          {result.verdict} (confidence {result.confidence})",
        f"manipulation:     {probability}  [{', '.join(result.verdict_reasons)}]",
        f"video quality:    {result.video_quality}",
    ]
    lines += [f"                  - {reason.message}" for reason in result.quality.reasons]
    lines.append(f"face tracks:      {result.face_tracks}")
    for track in result.tracks:
        score = "n/a" if track.score is None else f"{track.score:.3f}"
        lines.append(
            f"  track {track.track_id}: {track.start:6.2f}-{track.end:6.2f} s  score {score}  "
            f"quality {track.quality}  frames {track.scored_frames}/{track.detections}"
            + ("" if track.eligible else "  (not eligible)")
        )
    for interval in result.suspicious_intervals:
        lines.append(
            f"suspicious:       {interval.start:.1f}-{interval.end:.1f} s on track "
            f"{interval.track_id} (score {interval.score:.2f})"
        )
    p = result.processing
    lines.append(
        f"processed:        {p.frames_sampled} samples, {p.frames_scored} faces scored, "
        f"{p.duration_ms / 1000:.1f} s on {p.device}"
    )
    lines.append("")
    lines.append(result.disclaimer)
    return "\n".join(lines)


def _analyze(args: argparse.Namespace) -> int:
    from deepfake_core.config import PipelineConfig
    from deepfake_core.pipeline import Analyzer

    config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
    try:
        analyzer = Analyzer(config, models_dir=args.models_dir, device=args.device)
        output = analyzer.analyze(args.video, progress=_progress(args.quiet))  # type: ignore[arg-type]
    except VideoValidationError as error:
        print(f"\ninvalid video ({error.code}): {error.message}", file=sys.stderr)
        return EXIT_INVALID_VIDEO
    except (ModelMissingError, ModelIntegrityError) as error:
        print(f"\n{error}", file=sys.stderr)
        return EXIT_MODELS
    if not args.quiet:
        print(file=sys.stderr)

    result_json = output.result.model_dump_json(indent=2)
    if args.json is not None and str(args.json) == "-":
        print(result_json)
    else:
        if args.json is not None:
            args.json.write_text(result_json + "\n")
        print(_summary(output.result))
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        for image in output.evidence:
            (args.evidence_dir / f"{image.evidence_id}.jpg").write_bytes(image.jpeg)
    if args.trace is not None:
        output.trace.save_npz(args.trace)
    return 0


def _models(args: argparse.Namespace) -> int:
    root = models.models_dir(args.dir)
    failed = False
    for spec in models.MODELS:
        try:
            if args.action == "list":
                target = root / spec.path
                print(f"{spec.key:6} {spec.name} {spec.version} ({spec.license}) -> {target}")
                continue
            if args.action == "fetch":
                models.fetch(spec, root)
            path = models.verify(spec, root)
            print(f"ok      {spec.key:6} {path}")
        except (ModelMissingError, ModelIntegrityError, OSError) as error:
            failed = True
            print(f"failed  {spec.key:6} {error}", file=sys.stderr)
    return EXIT_MODELS if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepfake-core", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="analyze a video file")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--json", type=Path, help="write the result JSON here ('-': stdout)")
    analyze.add_argument("--evidence-dir", type=Path, help="write evidence face crops here")
    analyze.add_argument("--trace", type=Path, help="save the per-frame research trace (.npz)")
    analyze.add_argument("--config", type=Path, help="YAML/JSON overrides of PipelineConfig")
    analyze.add_argument("--device", help="auto (default), cuda, mps or cpu")
    analyze.add_argument("--models-dir", type=Path)
    analyze.add_argument("--quiet", action="store_true")

    model_commands = commands.add_parser("models", help="fetch, verify or list model weights")
    model_commands.add_argument("action", choices=["fetch", "verify", "list"])
    model_commands.add_argument(
        "--dir", help=f"models directory (default ${models.MODELS_DIR_ENV})"
    )

    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    return _models(args)


if __name__ == "__main__":
    raise SystemExit(main())
