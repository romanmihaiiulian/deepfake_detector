# ADR 0004: Video decoding with PyAV, and FFmpeg licensing

- Status: accepted
- Date: 2026-10-24

## Context

We need exact presentation timestamps, so that suspicious intervals and deterministic
sampling hold for variable-frame-rate video such as browser MediaRecorder WebM and phone
recordings. We also need container and codec metadata for validation, and display-matrix
rotation for phone videos. OpenCV's `VideoCapture` gives estimated frame counts and little
container information.

PyAV 19.0.2 (BSD-3-Clause, Python 3.12 or later) exposes all of this. Its binary wheels
bundle FFmpeg 9.0.2 built with x264 and x265, which makes that FFmpeg a **GPL build**.
opencv-python-headless bundles a separate LGPL FFmpeg.

## Decision

- Decode with PyAV, sequentially, with `thread_type="AUTO"`.
- Local development and the internal benchmark use the PyAV wheels as they are. Nothing
  is distributed in this pass.
- **Before any binary distribution** (Docker images, step 19), build PyAV against an LGPL
  FFmpeg configured without `--enable-gpl`. All the decoders we need (H.264, HEVC, VP8,
  VP9 and AV1, via the native decoders or dav1d) are LGPL or BSD. We never encode in the
  product.
- The attack lab uses the Homebrew `ffmpeg` CLI (also a GPL build) as an external tool
  for generating variants. It is never distributed.

## Consequences

- `dependencies/licenses.yml` lists "FFmpeg (bundled in PyAV wheels)" as GPL with
  `distributed: false`, and carries the action item above.
- On macOS, importing both PyAV and OpenCV prints an Objective-C warning: "Class
  AVFFrameReceiver is implemented in both …". Both wheels bundle `libavdevice`. The
  warning is harmless because we never open capture devices.
