"""Run the upstream GenD implementation once, to pin our implementation's fidelity.

The tool builds an isolated virtual environment with GenD's own pinned dependencies
(torch 2.8.0, transformers 4.56.2). It loads upstream `modeling_gend.py` from the pinned
Hugging Face revision against our local copy of the weights, scores GenD's sample crops,
and writes tests/core/data/gend_reference.json. That file holds numbers only, never images.

Usage:  uv run python -m tools.gend_reference_check --workdir /some/scratch/dir

Upstream's constructor downloads the full openai/clip-vit-large-patch14 checkpoint
(about 1.7 GB) to build the architecture. Everything goes under --workdir (HF_HOME is
redirected there), nothing goes into the user's global caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

from deepfake_core.models import GEND, models_dir
from tools.fetch_fixtures import FIXTURES_DIR, GEND_SAMPLES, is_fake_sample

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "tests" / "core" / "data" / "gend_reference.json"
UPSTREAM_PINS = ["torch==2.8.0", "transformers==4.56.2", "pillow", "safetensors", "numpy<2.4"]
HF_FILES = ("modeling_gend.py", "config.json")

RUNNER = textwrap.dedent(
    """
    import hashlib, importlib.util, json, sys
    from pathlib import Path

    import torch
    import transformers
    from PIL import Image

    upstream, model_dir, samples_json = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = importlib.util.spec_from_file_location("modeling_gend", upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.GenD.from_pretrained(model_dir).eval()
    processor = model.feature_extractor._preprocess
    results = {}
    with torch.inference_mode():
        for name, path in json.loads(samples_json).items():
            image = Image.open(path)
            pixels = model.feature_extractor.preprocess(image).unsqueeze(0)
            probs = model(pixels).softmax(dim=-1)[0]
            results[name] = float(probs[1])
    print(json.dumps({
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "image_processor": type(processor.image_processor).__name__,
        "p_fake": results,
    }))
    """
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(argv), file=sys.stderr)
    return subprocess.run(argv, check=True, text=True, **kwargs)  # type: ignore[call-overload,no-any-return]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    work = args.workdir.resolve()
    venv, upstream_dir, model_dir, runner_dir = (
        work / "venv",
        work / "upstream",
        work / "model",
        work / "runner",
    )
    for directory in (upstream_dir, model_dir, runner_dir):
        directory.mkdir(parents=True, exist_ok=True)

    weights = models_dir() / GEND.path
    if not weights.is_file():
        print("GenD weights missing: run `make models` first", file=sys.stderr)
        return 1
    missing = [f.path for f in GEND_SAMPLES if not (FIXTURES_DIR / f.path).is_file()]
    if missing:
        print("GenD sample crops missing: run `make fixtures` first", file=sys.stderr)
        return 1

    # Upstream code and config at the pinned weights revision. The downloaded files live
    # in their own directories, apart from the runner script we write.
    base = f"https://huggingface.co/yermandy/GenD_CLIP_L_14/resolve/{GEND.revision}/"
    for name in HF_FILES:
        target = (upstream_dir if name.endswith(".py") else model_dir) / name
        if not target.is_file():
            urllib.request.urlretrieve(base + name, target)
    link = model_dir / "model.safetensors"
    if not link.exists():
        link.symlink_to(weights.resolve())

    python = venv / "bin" / "python"
    if not python.is_file():
        run(["uv", "venv", "--python", "3.12", str(venv)])
        run(["uv", "pip", "install", "--python", str(python), *UPSTREAM_PINS])

    runner = runner_dir / "run_reference.py"
    runner.write_text(RUNNER)
    samples = {
        f.path.removeprefix("gend_samples/"): str(FIXTURES_DIR / f.path) for f in GEND_SAMPLES
    }
    env = {**os.environ, "HF_HOME": str(work / "hf"), "PYTHONNOUSERSITE": "1"}
    completed = run(
        [
            str(python),
            "-I",
            str(runner),
            str(upstream_dir / "modeling_gend.py"),
            str(model_dir),
            json.dumps(samples),
        ],
        capture_output=True,
        env=env,
        cwd=runner_dir,
    )
    reference = json.loads(completed.stdout.strip().splitlines()[-1])

    output = {
        "description": "Upstream GenD probabilities on GenD's sample crops (numbers only).",
        "model": {"repo": "yermandy/GenD_CLIP_L_14", "revision": GEND.revision},
        "upstream_modeling_gend_sha256": sha256_of(upstream_dir / "modeling_gend.py"),
        "torch": reference["torch"],
        "transformers": reference["transformers"],
        "image_processor": reference["image_processor"],
        "samples": {
            fixture.path.removeprefix("gend_samples/"): {
                "sha256": fixture.sha256,
                "label": "fake" if is_fake_sample(fixture) else "real",
                "p_fake": round(reference["p_fake"][fixture.path.removeprefix("gend_samples/")], 6),
            }
            for fixture in GEND_SAMPLES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
