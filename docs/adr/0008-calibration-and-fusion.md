# ADR 0008: Calibration and score fusion

- Status: accepted (code). No profile has been fitted on lab data yet.
- Date: 2026-10-25

## Context

`manipulation_probability` has been a raw GenD probability. GenD is over-confident
outside its training distribution: real Celeb-DF frames score 0.8–0.98. The verdict
bands (`uncalibrated-v0`) were therefore set by hand, and M7 could only recommend
thresholds on the raw scale. Step 14 added a second score, the temporal model, with no
rule yet for combining it with GenD. Step 16 makes the probability mean what it says,
and fuses the two detectors.

Constraints:
- Fitting data is lab val data. It is small, and its class ratio is arbitrary: one real
  video yields several fakes (swappers × variants).
- The product's first objective is a low false-positive rate.
- Results must stay reproducible. Every result already carries `config_profile` and
  `config_hash`.

## Decision

**Methods** (`deepfake_core/scoring/calibration.py`, numpy only):
- `platt`: logistic regression on the logits of the inputs. With the inputs `gend` and
  `temporal`, this also fuses the two detectors, and it is the default.
- `isotonic`: a monotone, piecewise-linear map of one input. It is more flexible, but it
  needs more data and it cannot fuse.

**Prior.** Fitting weighs classes equally, so the fitted output is P(manipulated) at a
50% prior. `calibration.prior` shifts the log-odds to the base rate an operator expects.
For example, 0.01 means "one interview in a hundred is a swap". The lab's class ratio is
never baked in. Every result states the prior in `calibration.prior` and in its note.

**Where it applies.** Calibration runs per track on the track's raw inputs. The video
probability is the maximum over eligible tracks, and the decisive track is the one with
the highest calibrated score.
- Calibrated: `manipulation_probability`, the track `score` and the verdict bands.
- Raw: frame scores, suspicious intervals, evidence and `detector_scores`.
- `calibration.raw_probability` keeps the decisive raw score.

The fit uses video-level scores, the maximum per input over eligible tracks. That
equals per-track application for single-face videos, which is what the lab produces.

**Thresholds** are chosen on the calibrated scale by `dfbench calibrate`, from val:
- `manipulated_threshold` holds val FPR at `--target-fpr` (default 1%).
- `authentic_threshold` clears at most `--target-cleared-fake` of val fakes (default 5%).
- The poor band uses poor val videos. When there are none, it is derived: the
  manipulated threshold moves halfway to 1 and the authentic threshold halves.
- Poor thresholds are clamped so that poor video never decides more readily than
  normal video. This keeps the quality-monotonicity property of `verdict.py`.

**Profiles.** `dfbench calibrate` writes a complete pipeline config: the run's recorded
config plus `calibration` and `verdict`, under profile `calibrated-<run>-<method>`. The
coefficients are inline, so `config_hash` covers them.

`calibration.model_sha256` binds the profile to the scoring models: the frame model,
and the temporal checkpoint when it is an input. The analyzer refuses a calibration
whose models are not loaded. The config rejects a `temporal` input without a temporal
checkpoint.

**Reporting.** Test is never used for fitting. The report shows class-balanced ECE
(15 bins), Brier score, log loss and AUC, raw vs calibrated, overall and per quality
tier. It also shows verdict rates on test under the run's thresholds and under the new
ones. A fit with fewer than 20 val videos per class is marked unreliable.

## Alternatives considered

- **Temperature scaling only.** This has one parameter, so it cannot fuse two detectors
  or correct a shifted intercept.
- **Beta calibration and splines.** These are reasonable, but there is too little lab
  data to justify more parameters than Platt's.
- **Fitting at the lab's natural class ratio.** This would encode an arbitrary
  generation choice as the probability of fraud.
- **Calibrating per quality tier.** This triples the parameters on tiny data. Quality
  already gates the verdict through the bands, and the report shows calibration per tier
  so that miscalibration on poor video is visible.

## Consequences

- Nothing changes by default. Profile `uncalibrated-v0` reports `calibration.method:
  "none"` and the same numbers as before.
- `AnalysisResult.calibration` gains `inputs`, `prior`, `fitted_on` and
  `raw_probability`, and `method` may be `platt` or `isotonic`. The change is additive,
  so the schema version stays 1.0. The web UI shows the calibrated probability next to
  the raw detector scores.
- Val now serves several purposes: temporal early stopping, calibration and thresholds.
  Val numbers are optimistic, so the reports direct readers to test.
- A calibrated probability is a statement about lab-like video and the configured
  prior. It is not a statement about a person, and the result note says so.
