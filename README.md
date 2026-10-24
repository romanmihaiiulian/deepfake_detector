# Deepfake Detector

An open-source (Apache-2.0) detector for face manipulation in video, focused first on face
swaps made with tools such as FaceFusion. V1 is a web application: you upload a video and
get a probabilistic risk assessment. The same engine is designed to later run in near real
time during remote interviews (HR and recruitment security).

> **Not proof.** Results are probabilistic assessments with stated confidence and video
> quality. They must not be used as the sole basis for any decision about a person.
> Verdicts are `authentic_likely`, `inconclusive` and `manipulated_likely`.

## Repository

```
packages/detector   deepfake-core: video decoding, quality gate, face tracking, GenD scoring
apps/api            deepfake-api: FastAPI service
apps/web            deepfake-web: React + Vite upload and results UI
benchmarks          deepfake-bench: FaceFusion attack lab and evaluation (internal)
tools               model fetcher, license checker, fixture fetcher
docs                architecture, ADRs, privacy, licensing, lab procedures, roadmap
tests               pytest suites
dependencies        licenses.yml, the machine-readable license inventory
```

See [docs/architecture.md](docs/architecture.md) for the detection cascade and module
boundaries, and [docs/roadmap.md](docs/roadmap.md) for the 22-step plan.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 (uv installs it), and
Node.js 20 or later for the web app. On Apple Silicon the engine uses the MPS GPU. Linux
uses CUDA when available, otherwise the CPU.

```sh
make setup    # Python workspace (+ web dependencies)
make models   # GenD (1.2 GB) and YuNet weights, pinned and sha256-verified, into ./models
make check    # lint, types, tests, license inventory
```

`make help` lists all targets. Tests that need model weights or downloaded fixtures
(`make fixtures`) skip when those are absent.

## Analyze a video

```sh
uv run deepfake-core analyze interview.mp4 --json result.json --evidence-dir evidence/
```

```python
from deepfake_core import analyze_video

result = analyze_video("interview.mp4")
print(result.verdict, result.manipulation_probability, result.confidence, result.video_quality)
```

## Run the web app

```sh
make build-web && uv run deepfake-api        # API + UI on http://127.0.0.1:8000
```

For development, run `make dev-api` and `make dev-web` together, then open
http://localhost:5173. Vite proxies `/api` to the API. Set `DFD_ANALYZER=fake` to work on
the UI without model weights: results are then canned and clearly marked as demo results.
The API is documented at `/api/docs`. Uploaded videos are deleted after analysis (see
[docs/privacy.md](docs/privacy.md)).

Read [docs/model-card.md](docs/model-card.md) before you rely on a result. Scores are
uncalibrated, and GenD has known false positives on some real footage.

## Attack lab and benchmark

The internal lab generates paired real/face-swapped videos from **consented** recordings
with FaceFusion (run as an external tool). It then measures the detector, including under
compression, screen-recording and video-conferencing degradation:

```sh
tools/install_facefusion.sh && export FACEFUSION_HOME=~/.local/share/deepfake-lab/facefusion
uv run dfbench identity add p0001 --consent-ref CONSENT-2026-001
uv run dfbench source add recording.mp4 --identity p0001     # ... more people
uv run dfbench split && uv run dfbench generate && uv run dfbench variants
uv run dfbench run --run-id baseline --save-traces && uv run dfbench report baseline
```

See [docs/lab/data-collection.md](docs/lab/data-collection.md) (consent rules) and
[docs/lab/benchmark.md](docs/lab/benchmark.md) (design, metrics, slices).

## Licensing

All code is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The model weights
are downloaded separately and have their own terms. Both GenD and YuNet are MIT-licensed,
but they were trained on research-only datasets, so commercial deployment needs legal
review. See [docs/licensing.md](docs/licensing.md) and
[`dependencies/licenses.yml`](dependencies/licenses.yml).

## Known issue

On macOS, importing the engine prints `objc[…]: Class AVFFrameReceiver is implemented in
both …`. PyAV and OpenCV each bundle FFmpeg's `libavdevice`. The warning is harmless
because the engine never opens capture devices (see ADR 0004).
