# Attack lab and benchmark

The lab builds our own face-swap benchmark from consented recordings
([data-collection.md](data-collection.md)). The benchmark measures the detector on it,
with particular attention to the false-positive rate under conferencing conditions.

## Paired samples

Each consented source video A yields:

- one **real** sample: A itself;
- **fakes derived from A**: the face of a donor identity from the same split is swapped
  into A with FaceFusion, once per configured swapper.

All of these share `pair_id = A`. A detector that tells them apart cannot rely on
identity, background, camera or content. The only difference is the manipulation, so the
benchmark reports the **paired win rate**: how often a fake scores above its own source.

### Encoding parity

```
source A ──canonicalise──▶ A_canon (x264 crf 0, CFR ≤ 30 fps, ≤ 1080p, no audio)
                              ├──────────────── encode "pristine" ─▶ real sample
                              └─ FaceFusion (crf 0 output) ─▶ F_raw ─ encode "pristine" ─▶ fake sample
```

Both members of a pair go through the same final encode, and in M8 through the same
robustness variants. If reals kept their camera encoding while fakes were re-encoded, a
detector could learn codec history instead of manipulation.

### Coverage check

Swappers sometimes miss faces (profiles, small or fast-moving faces). For every fake, the
lab compares `F_raw` with `A_canon` frame by frame at 4 Hz. A frame counts as swapped when
the face region changed clearly more than the background. Fakes swapped in less than 80%
of sampled frames, or whose frame count differs from the source, are recorded as
`status: rejected` with a reason, so they are never mislabelled.

## Splits

Identities are split **before** any fake is made. Donors and targets are paired only
within a split, so no identity or source video can appear in two splits: no identity
leakage and no source-video leakage. A union-find check over identities and videos runs
after every generation. Splits open in priority order test, val, train, each only once it
can hold at least two identities. Held-out swappers (`held_out_for_test`) appear only in
test: the "unseen FaceFusion settings" slice. External datasets (`dfbench import`) form a
separate `external` split: the "unseen tools" slice.

## Provenance (`lab/registry/samples.jsonl`)

| Field | Meaning |
|---|---|
| `generator`, `generator_version` | `facefusion` and the installed version, `none` for reals, or the external dataset name |
| `swapper_model`, `swapper_license`, `generator_settings`, `generator_argv` | Exactly how the fake was made. Paths in `generator_argv` are reduced to file names |
| `source_video` | The video the sample derives from (also `pair_id`) |
| `source_identity` | The donor whose face was inserted |
| `target_identity` | The person in the source video, whose face was replaced |
| `resolution`, `fps`, `duration` | Of the final file |
| `transformation`, `transformation_argv`, `ffmpeg_version` | The encode or degradation profile, recorded exactly |
| `manipulation_coverage` | Fraction of sampled frames in which the face was swapped |
| `split`, `research_only`, `status`, `status_reason` | |

Identities are pseudonymous and carry only a `consent_ref`.

## Running the benchmark

```sh
dfbench run --run-id 2026-10-24-baseline --save-traces   # val + test by default
dfbench report 2026-10-24-baseline                        # data/bench/runs/<id>/report.md
```

`run` analyses each selected sample with the production pipeline (`Analyzer`). It writes
`predictions.jsonl` and, with `--save-traces`, `traces/<sample_id>.npz`: per-frame
timestamps, scores, quality tiers and GenD embeddings for each track, which is the input
of the temporal model (step 14). Predictions are cached in `data/bench/cache/`, keyed by
sample sha256, pipeline version, config hash and model checksums, so re-runs and new
reports never redo work. `--pipeline-config` evaluates a candidate configuration, for
example new thresholds.

## Metrics (video level)

| Metric | Notes |
|---|---|
| ROC-AUC, PR-AUC | Threshold-free ranking quality |
| Precision, recall, F1, FPR, FNR | At the operating threshold of the evaluated config |
| TPR at 1% and 5% FPR | The HR-relevant operating points |
| ECE (15 bins), Brier | Calibration of the score as reported; see `dfbench calibrate` below for class-balanced versions (step 16) |
| Flagged-real rate | Real videos whose verdict was `manipulated_likely`. This is the false accusation rate |
| Cleared-fake rate, abstention rate, decided accuracy | Verdict-level behaviour, including `inconclusive` |
| Paired win rate | Share of fakes that score above their own source video |

95% confidence intervals come from a cluster bootstrap (1000 resamples of source videos).
Expect wide intervals with few identities: they are reported precisely so that small
benchmarks are not over-read.

**Slices:** all, lab, each split, `identities:unseen` (test), each transformation and
transformation group, each swapper, `settings:seen` and `settings:unseen` (held-out
swappers, test only), `tool:<dataset>` (external datasets, i.e. unseen tools), predicted
quality tier, and resolution bucket.

**Threshold recommendation (raw scale):** for each verdict band (`manipulated_threshold` on
good/fair video, `poor_manipulated_threshold` on poor video), the report lists the
threshold that keeps the FPR on **val** reals at 1% and at 5%. It marks a recommendation
unreliable when there are too few val reals to resolve that rate. Test is never used to
choose thresholds.

**Aggregators:** mean (GenD's protocol), median and maximum 4 s window mean, compared on
the same predictions. The window statistic exists to catch partial manipulations. When
the evaluated config has a temporal model, `temporal` is compared as well.

## Temporal model (`dfbench temporal`, step 14)

The temporal model needs traces of all three splits. Robustness variants of train are
optional, and they teach the model degraded footage:

```sh
dfbench variants --split train --split val --split test          # optional
dfbench run --run-id traces --split train --split val --split test --save-traces
dfbench temporal train traces --arch tcn          # or --arch transformer; --seed N
dfbench temporal eval traces --checkpoint tcn-traces-s0
```

- `train` fits on train and early-stops on val video AUC. The untrained network, which
  equals the GenD mean, is a candidate, so the result is never worse on val. It writes
  `data/bench/temporal/<name>.safetensors` and a training record `<name>.json` with the
  samples, identities and per-epoch history.
- `eval` writes `data/bench/runs/<run>/temporal/<name>/report.md`. The report has
  temporal vs mean, median and window_max on val and test (AUC, TPR at 1% and 5% FPR,
  paired win rate), AUC by transformation, and AUC(temporal) − AUC(mean) with a
  pair-level bootstrap CI. Read test, not val: val chose the epoch.
- To run the product pipeline with the model, set `temporal.checkpoint` in a pipeline
  config. Add `aggregation.track_score: temporal` to let it decide the verdict. Then
  `dfbench run --pipeline-config` benchmarks the whole pipeline with it.

Design and rationale: [ADR 0007](../adr/0007-temporal-model.md).

## Calibration (`dfbench calibrate`, step 16)

```sh
dfbench calibrate traces                       # Platt; fuses temporal when the run has it
dfbench calibrate traces --method isotonic --input gend
dfbench calibrate traces --prior 0.01 --target-fpr 0.01 --target-cleared-fake 0.05
dfbench run --run-id calibrated --pipeline-config data/bench/runs/traces/calibration/platt.yaml
```

- **Fit:** on val, with classes weighted equally, so the result is P(manipulated) at a
  50% prior. `--prior` shifts it to the base rate you expect.
- **Thresholds:** chosen on the calibrated val scores. `manipulated` holds the FPR at
  the target. `authentic` clears at most the target share of fakes. The poor band uses
  poor val videos, or is derived from the normal band when there are none.
- **Report:** `calibration/<method>.report.md` shows class-balanced ECE, Brier, log loss
  and AUC on test, raw vs calibrated, overall and per quality tier, plus verdict rates
  before and after.
- **Profile:** `calibration/<method>.yaml` is a complete pipeline config, bound to the
  scoring models' checksums. Use it with `deepfake-core analyze --config`, the API's
  config file or `dfbench run --pipeline-config`.
- **Reliability:** with fewer than 20 val videos per class, the fit is marked unreliable.

The run must record its pipeline config, which `dfbench run` does from this version
on. Design: [ADR 0008](../adr/0008-calibration-and-fusion.md).

## Robustness variants (`dfbench variants`)

Every pristine val and test sample is re-encoded with each profile. Each profile starts
from the sample's own near-lossless intermediate, so both members of a pair get exactly
the same recipe. The argv and the ffmpeg version are recorded, and the profiles involve
no randomness.

| Profile | What it simulates |
|---|---|
| `h264_crf23`, `h264_crf28`, `h264_crf35`, `h265_crf28` | Recompression (re-uploads, messaging apps) |
| `res_1080p`, `res_720p`, `res_480p` | Downscaling, never upscaling |
| `fps_30`, `fps_15` | Frame-rate conversion |
| `crop_10`, `letterbox` | Reframing |
| `screen_recording` | The call in a window on a 1080p desktop, captured by a screen recorder |
| `conf_low`, `conf_medium`, `conf_high` | WebRTC-like call (realtime VP8 at 300/800/1500 kbps, 360p15/540p24/720p30), then re-encoded as the meeting recording |

`dfbench variants --list` prints the profiles. The report adds a robustness table: AUC
change vs pristine, **FPR on real videos** (by score and by verdict) and the quality-tier
mix, per profile. A data-driven quality-threshold section bins real videos by each
quality metric. It proposes a `quality-v1` POOR boundary only where every bin has at
least 20 reals and false positives clearly rise at low quality. Otherwise the thresholds
stay `uncalibrated-v0`, and the report says so.
