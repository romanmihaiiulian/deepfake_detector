# Roadmap

V1 is the web upload detector. It is followed by V2 (API), V3 (streaming) and V4 (the
Teams/Zoom HR product).

| Step | Item | Status |
|---:|---|---|
| 1 | Repository, architecture, licensing | done (pass 1, M0) |
| 2 | Video decoding | done (pass 1, M1) |
| 3 | Deterministic frame sampling | done (pass 1, M1) |
| 4 | Video quality analysis | done (pass 1, M2) |
| 5 | Face detection | done (pass 1, M3) |
| 6 | Persistent face tracking | done (pass 1, M3) |
| 7 | Face alignment | done (pass 1, M3) |
| 8 | GenD integration | done (pass 1, M4) |
| 9 | GenD video-level aggregation | done (pass 1, M4) |
| 10 | Basic web interface | done (pass 1, M5) |
| 11 | FaceFusion dataset generator | done (pass 1, M6) |
| 12 | Benchmark framework | done (pass 1, M7) |
| 13 | Compression / conferencing benchmark | done (pass 1, M8) |
| 14 | Temporal detector | done (M9: code and tests; training waits for consented lab data) |
| 15 | Escalation layer (second opinion) | deferred: VLAForge rejected on provenance ([ADR 0006](adr/0006-vlaforge-provenance.md)) |
| 16 | Fusion / probability calibration | done (M10: code and tests; fitting waits for consented lab data) |
| 17 | Suspicious timeline / evidence | later (a first version ships in M4/M5) |
| 18 | Production API | later |
| 19 | Docker / GPU deployment | later |
| 20 | Streaming engine | later |
| 21 | Teams / Zoom integration research | later |
| 22 | Enterprise HR pilot | later |

## Notes for later steps

- **14, temporal model.** Done in M9 ([ADR 0007](adr/0007-temporal-model.md)). TCN and
  Transformer over GenD embeddings, trained with `dfbench temporal train` on train-split
  traces and compared with the frame-score baselines by `dfbench temporal eval`. What
  is left is the real run on consented lab data.
- **15, escalation.** VLAForge cannot be used. Its code is mostly Forensics Adapter
  and DeepfakeBench, both CC BY-NC 4.0, and its weights have no licence (ADR 0006).
  Deferred until a lab benchmark of steps 14 and 16 shows where GenD fails.
  Whatever fills the hook later runs only on suspicious windows.
- **16, calibration.** Done in M10 ([ADR 0008](adr/0008-calibration-and-fusion.md)).
  `dfbench calibrate` fits Platt or isotonic on val, optionally fusing GenD and temporal.
  It chooses thresholds at a target FPR and reports class-balanced ECE, Brier and log
  loss on test. What is left is a fit on consented lab data.
- **19, Docker.** Build PyAV against an LGPL FFmpeg (ADR 0004). Use CPU and CUDA torch
  indexes for Linux.
- **20, streaming.** The engine already processes frames in streaming order with an
  online tracker. Add a rolling 3–5 s buffer and emit a risk update about once a second.
