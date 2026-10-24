# Roadmap

V1 is the web upload detector. It is followed by V2 (API), V3 (streaming) and V4 (the
Teams/Zoom HR product).

| Step | Item | Status |
|---:|---|---|
| 1 | Repository, architecture, licensing | pass 1, M0 |
| 2 | Video decoding | pass 1, M1 |
| 3 | Deterministic frame sampling | pass 1, M1 |
| 4 | Video quality analysis | pass 1, M2 |
| 5 | Face detection | pass 1, M3 |
| 6 | Persistent face tracking | pass 1, M3 |
| 7 | Face alignment | pass 1, M3 |
| 8 | GenD integration | pass 1, M4 |
| 9 | GenD video-level aggregation | pass 1, M4 |
| 10 | Basic web interface | pass 1, M5 |
| 11 | FaceFusion dataset generator | pass 1, M6 |
| 12 | Benchmark framework | pass 1, M7 |
| 13 | Compression / conferencing benchmark | pass 1, M8 |
| 14 | Temporal detector | later |
| 15 | VLAForge escalation layer | later |
| 16 | Fusion / probability calibration | later |
| 17 | Suspicious timeline / evidence | later (a first version ships in M4/M5) |
| 18 | Production API | later |
| 19 | Docker / GPU deployment | later |
| 20 | Streaming engine | later |
| 21 | Teams / Zoom integration research | later |
| 22 | Enterprise HR pilot | later |

## Notes for later steps

- **14, temporal model.** The input is in the benchmark's `traces/*.npz`: per-track
  L2-normalised GenD embeddings at about 4 samples/s, so 8–16 samples cover 3–5 s.
  Benchmark the mean/median baseline, a TCN and a small Transformer. Train only on the
  `train` split. The lab's identity-level splits prevent leakage.
- **15, VLAForge.** Start with the provenance review recorded in `licenses.yml`: code
  structure derived from DeepfakeBench, weights with no license, FF++ training. Run it
  only on suspicious windows.
- **16, calibration.** Replace the M7 threshold recommendation with calibrated
  probabilities fitted on `val`, reported with ECE on `test`. Set `calibration.method` in
  results.
- **19, Docker.** Build PyAV against an LGPL FFmpeg (ADR 0004). Use CPU and CUDA torch
  indexes for Linux.
- **20, streaming.** The engine already processes frames in streaming order with an
  online tracker. Add a rolling 3–5 s buffer and emit a risk update about once a second.
