# Model card: Deepfake Detector V1 (pipeline 0.1.0, profile `uncalibrated-v0`)

## What it is

A video pipeline that estimates whether the faces in a video were manipulated, mainly by
face swapping:

- **Face detector:** YuNet (`face_detection_yunet_2023mar`, opencv_zoo
  `eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4`).
- **Tracking:** our own geometric tracker. It does no identity recognition.
- **Alignment:** GenD's 5-point template with a 1.3× margin.
- **Frame detector:** GenD CLIP ViT-L/14 with its LinearNorm head (`yermandy/GenD_CLIP_L_14`
  revision `891ce014a0308386c4d7d25b3dcf436a22db5504`). Our loader reproduces upstream
  GenD's probabilities on its sample crops to within 1e-4.
- **Video decision:** the mean frame score per face track, then the maximum over eligible
  tracks, gated by video quality.

## Intended use

- Decision support in remote-hiring and identity-assurance workflows. A
  `manipulated_likely` or `inconclusive` result should trigger additional verification by
  a person.
- Research and benchmarking of face-manipulation detection.

## Out of scope

- Proof that a video or a person is fraudulent, and automated rejection of candidates or
  any automated decision about a person.
- Identity verification or recognition. The system never determines who someone is.
- Audio deepfakes, voice cloning, and attribution to a specific generator.

## Outputs

- `manipulation_probability`: the **raw, uncalibrated** detector score (0–1). It is not the
  probability that the video is fake.
- `verdict`: `authentic_likely`, `inconclusive` or `manipulated_likely`, from the bands in
  `VerdictConfig`. With GOOD or FAIR quality: manipulated at ≥ 0.80, authentic at ≤ 0.30.
  With POOR quality: ≥ 0.90 and ≤ 0.15. INSUFFICIENT quality is always inconclusive.
- `confidence`: high, medium or low. It depends on the margin to the threshold, the
  number of scored frames, and the spread of frame scores. FAIR quality caps it at
  medium; POOR quality forces low.
- `video_quality` with reasons, per-track timelines, suspicious intervals, and up to four
  evidence face crops.

## Training data and provenance

GenD was trained by its authors on FaceForensics++ (c23). The dataset's terms restrict it
to research. The weights are MIT-labelled. Commercial deployment needs legal review (see
`dependencies/licenses.yml`, ADR 0003). YuNet was trained on WIDER FACE.

## Known limitations

These are observations, not benchmark results:

- **False positives on some real footage.** On the 20 sample crops that ship with GenD,
  fake FF++ frames score 0.96–0.99, and real FF++ frames 0.09–0.66. Four real Celeb-DF
  v2 frames (`Celeb-real/id0_0000`) score **0.80–0.98**. Upstream GenD gives the same
  values, so this is model behaviour, not a porting error.
- **A real portrait is flagged.** We made test clips from three public-domain NASA portraits,
  all real photographs (`make fixtures`). As a static clip, `mae_jemison.jpg` scored **0.94**;
  gently animated, it scored 0.71. The other two portraits scored 0.22–0.32. Three images
  prove nothing statistically. But a false positive on the only Black woman in our small
  fixture set is exactly the disparate-impact failure a demographic evaluation must rule
  out before any HR deployment.
- **Out-of-distribution content.** Old or scanned photos, heavy retouching, beauty filters
  and animated stills are not interview-like video. Expect unreliable scores on them.
- **Different face detector.** Our detector is not the one GenD was trained with. YuNet
  landmarks differ slightly from SCRFD's, so crops differ slightly. The benchmark (M7)
  measures the effect.
- **Compression and conferencing degradation have not been measured yet.** M8 measures
  FPR on recompressed, low-resolution, screen-recorded and conference-like video.
- **Frame-level detector by default.** The temporal model (step 14, ADR 0007) is
  implemented, but no checkpoint has been trained on lab data yet, so it is off by
  default. There is no second opinion (step 15, deferred).
- **No demographic evaluation yet.** The detector must be evaluated across skin tones, ages
  and genders before any deployment.

## Thresholds and calibration

All thresholds are profile `uncalibrated-v0`. They are deliberately conservative on the
manipulated side, because the product's first objective is a very low false-positive rate.
The benchmark recommends thresholds at 1% and 5% FPR on the validation split. Proper
calibration is step 16. Every result reports `processing.config_profile` and
`config_hash`.

## Privacy

Uploaded videos are deleted after analysis. Results keep only scores, quality metrics and
up to four face crops, which expire after 24 h. Embeddings and full frames are never
stored. See [privacy.md](privacy.md).
