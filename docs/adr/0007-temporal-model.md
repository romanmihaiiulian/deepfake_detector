# ADR 0007: Temporal model over GenD embeddings

- Status: accepted (code). No checkpoint has been trained yet, because training needs
  consented lab data.
- Date: 2026-10-25

## Context

GenD scores frames independently, and the video score is a mean over frames. Face swaps
also leave temporal traces, such as flicker in identity, texture and blending from one
frame to the next. A per-frame mean cannot see those traces. Step 14 adds a model that
sees each track as a sequence.

Constraints:
- Training data is our own consented lab footage, so it is small: tens of identities,
  not thousands.
- The input already exists. Benchmark traces (`dfbench run --save-traces`) store, per
  track, the timestamps, GenD probabilities, face-quality tiers and L2-normalised
  1024-d GenD embeddings at about 4 samples/s.
- The product's first objective is a low false-positive rate, and scores are still
  uncalibrated (step 16).
- Everything we ship stays Apache-2.0, with no third-party detector code.

## Decision

**Input.** One row per scored frame. A row is the embedding plus four scalars: the GenD
logit, the gap since the previous scored frame, the cosine distance to the previous
embedding, and the absolute GenD-logit change (`temporal/features.py`). Unscored frames
are dropped, and the gap feature exposes the holes they leave.

The two step features are deliberate. On synthetic traces where fakes differ only by
frame-to-frame flicker, both architectures memorised the 48 training videos from
absolute embeddings alone (val AUC about 0.5). With the step features they generalised
(val AUC 1.0). Lab data will be just as small.

**Windows.** 16 samples (4 s at 4 Hz) with stride 8. The last window is aligned to the
track end, so every scored frame is covered. The track score is the mean window
probability, and the video score is the maximum over eligible tracks, as for every other
aggregator.

**Network** (`temporal/model.py`, pure PyTorch). There are two encoders behind one
interface:
- `tcn`: 3 residual blocks of dilated 1-D convolutions, with dilations 1, 2 and 4.
- `transformer`: 3 layers with learned positions.

Both have 128 hidden units, a masked mean pool and a linear head. The output is
`base_weight * mean(GenD logit) + base_bias + head(pooled)`, with the head initialised to
zero. An untrained network therefore equals the GenD mean-logit baseline, and training
learns a correction to it. Padding is exact, and a test checks it.

**Training** (`dfbench temporal train`). Rules:
- The train split only, using the main (longest) eligible track of each video.
- Every video weighs the same and both classes weigh the same.
- AdamW and weighted binary cross-entropy.
- Early stopping on val video AUC. Epoch 0, the untrained baseline, is a candidate, so a
  checkpoint is never worse on val than GenD's mean.
- Deterministic for a given seed on CPU.

**Evaluation** (`dfbench temporal eval`). Compares the checkpoint with the mean, median
and window_max baselines on val and test. All four use the same traces and the same
eligible tracks. The report gives AUC, TPR at 1% and 5% FPR, the paired win rate, AUC by
transformation, and AUC(temporal) − AUC(mean) with a pair-level bootstrap CI. Test
samples that share an identity with the training record are excluded and counted.

**Checkpoints.** A checkpoint is one safetensors file whose metadata holds the
hyper-parameters, window, stride, a training-set hash, and the sha256 of the frame model
whose embeddings it was trained on. A sidecar `<name>.json` lists the training and
validation samples and identities. Checkpoints live under
`$DFD_DATA_ROOT/bench/temporal/`, never in the repository. Withdrawing an identity's
consent deletes every checkpoint whose sidecar lists that identity.

**Integration.** `temporal.checkpoint` in the pipeline config enables the model:
- Its score is reported as `detector_scores.temporal`, and the model appears in
  `models` with role `temporal`.
- It decides the verdict only when `aggregation.track_score: temporal` is set.
- The config rejects `track_score: temporal` without a checkpoint.
- Loading a checkpoint next to a different frame model raises `ModelIntegrityError`.

The benchmark cache key includes model checksums, so runs with and without a temporal
model never mix.

## Alternatives considered

- **Raw-frame video models** (3D CNNs, video transformers). These need far more data
  and compute, and they would discard GenD's generalisation.
- **Hand-crafted temporal statistics only** (variance and autocorrelation of scores).
  These are cheap, but they cannot use the embeddings. The step features keep the
  useful part of this idea inside a learned model.
- **A recurrent network.** This is a valid option, but a TCN and a small Transformer
  cover short- and long-range context with less tuning, and both are benchmarked.

## Consequences

- Nothing changes by default. Without a checkpoint, results are identical to before
  (`detector_scores` holds only `gend`).
- The real question is whether lab traces beat the mean baseline on test. That waits
  for consented data, at least 6 identities and preferably 30 or more. Until then, the
  synthetic tests show only that the pipeline can learn temporal cues the baselines
  cannot see.
- A trained checkpoint is derived from lab data of consenting people. It is internal
  and is never distributed. A model trained on a commercially clean subset (ADR 0006
  notes, `licenses.yml`) is a separate, later decision.
- Fusing GenD and temporal scores, and calibrating them, is step 16.
