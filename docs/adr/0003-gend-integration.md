# ADR 0003: GenD integration with our own loader

- Status: accepted
- Date: 2026-10-24

## Context

GenD (Yermakov et al., WACV 2026) fine-tunes only the LayerNorm parameters of CLIP
ViT-L/14 and targets cross-dataset generalisation. The reference code is
[yermandy/GenD](https://github.com/yermandy/GenD) (MIT, pinned commit
`387a42266dd385fbe8f7626c5d3aa03eea3bfaab`). The weights are on Hugging Face as
`yermandy/GenD_CLIP_L_14` (MIT tag, pinned revision
`891ce014a0308386c4d7d25b3dcf436a22db5504`).

The upstream loader is a custom `PreTrainedModel` subclass that is loaded with
`trust_remote_code`, pinned to transformers 4.56.2. Facts taken from the weights and the
code:

- `config.json` has `"head": "LinearNorm"`. The 1024-d `pooler_output` is L2-normalised
  before the linear layer.
- The safetensors keys are `feature_extractor.vision_model.*` (exactly the HF
  `CLIPVisionTransformer` layout), `feature_extractor.visual_projection.weight` (not used
  by the forward pass) and `model.linear.{weight,bias}` with shape `[2, 1024]`.
- There are 257 position embeddings, so the input is 224×224 (16×16 patches plus CLS).
- Label index 1 is fake. `src/dataset/dataset.py` sets `label2name = {0: "real", 1: "fake"}`,
  and the demo app reads `p_fake = probs[1]`.
- Preprocessing is `detector.py:align_face`: a similarity transform from 5 landmarks to
  the template `[[.34,.46],[.66,.46],[.5,.64],[.37,.82],[.63,.82]]`, with 1.3× scale
  margin and a native-size square crop. It is followed by `CLIPProcessor`: bicubic resize
  to 224 and CLIP mean/std normalisation.
- Upstream evaluates 32 frames per video and scores the video as the mean of the frame
  probabilities.

## Decision

- Implement the model ourselves: transformers `CLIPVisionModel` (ViT-L/14 config defined
  in code), `F.normalize`, then `Linear(1024, 2)`. Load the safetensors with an explicit,
  strict key remap that drops `visual_projection`. There is no `trust_remote_code`, no
  network access at load time, and it works on current transformers 5.x.
- Port `align_face` (MIT, attribution in the file header and in NOTICE), and reproduce
  `CLIPProcessor`'s resize and normalisation.
- Report `p_fake = softmax(logits)[:, 1]`. The embedding exposed to later stages is the
  L2-normalised `pooler_output`, which is exactly what the head consumes.
- Fidelity is pinned by tests:
  - `tools/gend_reference_check.py` runs the upstream implementation once, in an
    isolated environment with its pinned dependencies, on GenD's own sample crops. It
    records the resulting probabilities in `tests/core/data/gend_reference.json`.
  - `test_gend_fidelity.py` requires our implementation to agree within 0.01.
  - A tiny random checkpoint with the same key layout tests the key mapping and the head
    without the 1.2 GB download.

## Consequences

- The GenD weights are MIT-labelled but were trained on FaceForensics++, whose terms
  restrict it to research. The OpenAI CLIP model card lists deployed use as out of scope.
  Both are recorded as `commercial_use: review_required`. The clean long-term path is to
  retrain GenD's recipe on our lab data plus licensed real footage.
- GenD's video score is the mean over frames. A window-max aggregator is recorded in the
  trace so that the benchmark can decide between them with data.
