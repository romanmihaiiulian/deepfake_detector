# ADR 0006: VLAForge provenance review, so do not integrate it

- Status: accepted (do not integrate). Step 15 is deferred (option D, chosen
  2026-10-25).
- Date: 2026-10-25

## Context

Step 15 planned to use VLAForge (Zhu et al., CVPR 2026) to confirm suspicious windows.
The code is [mala-lab/VLAForge](https://github.com/mala-lab/VLAForge), and the repository
carries an MIT `LICENSE` (copyright 2026 MaLA). `licenses.yml` recorded three concerns:
the dataset and trainer layout mirrors DeepfakeBench, the weights have no licence, and
the model was trained on FF++. Checking those concerns was the first task of step 15.

Method: we fetched every non-vendored Python file from VLAForge commit
`5ec3b2b4333a8b4140b1d5de18f80faae0789813` into a scratch directory, outside this
repository. We compared each one with its counterpart in two other repositories:
- DeepfakeBench (`SCLBD/DeepfakeBench` commit `f188b1c105465e2e5377eb536a95022ae0e4522d`)
- Forensics Adapter (`OUC-VAS/ForensicsAdapter` commit
  `a4cc94fa4255d2baa85cb26989fb884b098010e3`, Cui et al., CVPR 2025)

The metric counts identical lines, ignoring blank lines and comments, using `difflib`
matching blocks. No code from any of the three repositories was copied into this
repository.

## Findings

**1. The model code is Forensics Adapter's, which is CC BY-NC 4.0.**
Forensics Adapter's `LICENSE` is the CC BY-NC 4.0 legal code, and its README says the
repository is licensed CC BY-NC 4.0. VLAForge keeps a hard-coded weights path from
Forensics Adapter's first author (`download_root='/data/cuixinjie/weights'`).

| VLAForge file | Lines | Identical to Forensics Adapter |
|---|---:|---|
| `model/layer.py` | 143 | 141 (99%), `model/layer.py` |
| `model/attn.py` (`RecAttnClip`) | 141 | 132 (94%), `model/attn.py` |
| `model/ForgePerceiver.py` | 141 | 130 (92%), `model/adapters/adapter.py` |
| `model/VLAForge.py` | 198 | 114 (58%), `model/ds.py` |
| `model/clip/clip.py` | 179 | 179 (100%), `model/clip/clip.py` |
| `model/clip/model.py` | 383 | 365 (95%), `model/clip/model.py` |

The `model/clip` files are originally OpenAI CLIP, which is MIT. Forensics Adapter
modified them, and VLAForge then added an identity embedding to `encode_text`.

**2. The training, dataset and metrics code is DeepfakeBench's, which is CC BY-NC 4.0.**
DeepfakeBench's files carry `author: Zhiyuan Yan` headers. VLAForge's copies have the
headers removed, and its README does not acknowledge DeepfakeBench.

| VLAForge file | Lines | Identical to DeepfakeBench |
|---|---:|---|
| `dataset/albu.py`, `logger.py`, `trainer/base_trainer.py`, `trainer/metrics/base_metrics_class.py`, `trainer/metrics/registry.py` | 309 | 309 (100%) |
| `trainer/metrics/utils.py` | 71 | 70 (99%) |
| `trainer/trainer.py` | 352 | 292 (83%) |
| `test.py` | 130 | 101 (78%) |
| `dataset/abstract_dataset.py` | 383 | 267 (70%) |
| `train.py` | 244 | 161 (66%) |

**3. The MIT label cannot relicense this material.** A licensor can only grant rights
it holds. CC BY-NC 4.0 §2(a)(5)(B) forbids imposing different terms on the licensed
material, and Adapted Material must keep the NonCommercial restriction. The only parts
of VLAForge we could use under MIT are its own additions: the identity-aware text
prompting and the VLA score in `model/VLAForge.py`, the loss terms, and the configs.
Those additions are not separable from the adapter they extend.

**4. The weights have no usable licence.** The pretrained checkpoints are a Google Drive
link in the README, with no licence statement. They contain a trained Forensics Adapter
side network, and they were trained on FaceForensics++, whose terms of use are
research-only. DF40 subsets, also non-commercial, are used for evaluation.

**5. Other dependencies are clean.** The vendored `open_clip` is MIT, and the frozen
backbone is OpenAI CLIP ViT-L/14 (MIT, with the model-card caveat already recorded for
GenD). The identity embedding comes from CLIP's image features rather than an ArcFace
model, so InsightFace's non-commercial weights are not part of inference. They do appear
in `environment.yml`.

## Decision

- Do not vendor, import or pip-install VLAForge code in this repository or in anything
  we distribute.
- Do not fetch or ship VLAForge or Forensics Adapter weights through `make models`.
- In `licenses.yml`, record VLAForge as `commercial_use: 'no'` with license
  `CC-BY-NC-4.0`, since that is the effective licence of most of its code. Add Forensics
  Adapter as a research reference.
- The papers stay usable as research references, as DeepfakeBench and SBI already are.

## Options for step 15

Chosen on 2026-10-25: **D, defer**. Steps 14 and 16 come first, and escalation is
revisited once the benchmark shows where GenD fails.

| Option | What it takes | Risk |
|---|---|---|
| A. Ask for licences | Write to MaLA (VLAForge), OUC-VAS (Forensics Adapter) and CUHK-SZ (DeepfakeBench) asking for a commercial licence of the code and weights. | Three licensors, and FF++ training data remains. Slow, but it is the only path to using their checkpoint. |
| B. Clean-room reimplementation | Reimplement from the two papers only (frozen CLIP, a side adapter with learned queries, identity-conditioned prompts), written without looking at either codebase, and train on lab data plus licensed real footage. | Engineering and training cost. We need our own data at scale, which is the same gap as GenD's long-term clean path (ADR 0003). Patents are unchecked. |
| C. A different second opinion | A model with permissive code and weights, or more GenD-family capacity: a second GenD backbone, or test-time augmentation over the suspicious window. | Less diversity of errors than a different architecture. |
| D. Defer step 15 | Go to step 14 (temporal model) and step 16 (calibration). Revisit escalation once the benchmark shows where GenD fails. | None now. Escalation is an accuracy add-on, not a dependency. |

The recommendation is D now, A in parallel if a commercial product is still the goal, and
B only once the lab produces enough consented data to train it. Even internal evaluation
of the original checkpoint counts as use "primarily intended for commercial advantage"
when it serves a commercial product, so it needs a licence (option A) first.

## Consequences

- The architecture keeps the escalation hook, but no model fills it in V1.
- `docs/model-card.md` and `docs/roadmap.md` no longer name VLAForge as the planned
  second opinion.
- If option B is chosen, the clean-room rule is that implementers must not have read
  the VLAForge or Forensics Adapter code. This review read it, so the reviewer should not
  write that implementation.
