# ADR 0002: YuNet as face detector instead of InsightFace

- Status: accepted
- Date: 2026-10-24

## Context

GenD was trained on face crops aligned from 5-point landmarks. Its preprocessing
(`src/retinaface.py`) uses InsightFace's `buffalo_l/det_10g.onnx`, an SCRFD detector.
InsightFace's code is MIT, but its README states that the pretrained models "are
available for non-commercial research purposes only". The product must stay commercially
usable, so we cannot ship that model.

Candidates we considered:

| Detector | License | Landmarks | Notes |
|---|---|---|---|
| InsightFace SCRFD (`det_10g`) | models non-commercial | 5 | matches GenD training, cannot ship |
| YuNet (OpenCV Zoo) | MIT | 5, same order | small (233 KB), fast, built into OpenCV as `cv2.FaceDetectorYN` |
| MediaPipe BlazeFace | Apache-2.0 | 6, no mouth corners | does not fit the 5-point template |
| MediaPipe Face Landmarker | Apache-2.0 | 478 | could derive 5 points, but heavier and a different geometry |
| YOLO-face variants | AGPL | 5 | copyleft |

## Decision

- Use YuNet `face_detection_yunet_2023mar.onnx` from opencv_zoo commit
  `eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4` (sha256 `8f2383e4…2fa4`). It runs through
  `cv2.FaceDetectorYN` on opencv-python-headless 4.14 (pinned `<5`).
- YuNet returns its 5 landmarks in image-space order: image-left eye, image-right eye,
  nose tip, image-left mouth corner, image-right mouth corner. That is the order GenD's
  template expects, and a test asserts it on a real face.
- The detector sits behind a `FaceDetector` protocol so it can be swapped.

## Consequences

- Our crops differ slightly from GenD's training crops, because landmark noise differs
  between detectors. The benchmark (M7) measures the effect with the paired win-rate and
  AUC on lab data. If it is significant, the first remedies are landmark smoothing along
  a track, or a 5-point derivation from MediaPipe Face Landmarker.
- YuNet was trained on WIDER FACE, whose terms are research-oriented. The weights are
  MIT-licensed, but their training-data provenance is recorded as `review_required` in
  `dependencies/licenses.yml`.
- OpenCV 5 ships a new DNN engine, and YuNet's `2026may` model targets it. Moving to it
  is a separate, later change.
