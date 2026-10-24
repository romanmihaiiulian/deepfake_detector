# ADR 0005: Asynchronous jobs with a single worker and SQLite

- Status: accepted
- Date: 2026-10-24

## Context

Analysing a clip takes tens of seconds on a laptop GPU (MPS) and longer on a CPU. HTTP
requests must not block that long. The engine holds about 1.3 GB of weights, so we want
exactly one loaded `Analyzer` per process. The V1 deployment is a single process.

## Decision

- `POST /api/v1/analyses` stores the upload, probes and validates it, records a job and
  returns `202` with an `analysis_id`. Clients poll `GET /api/v1/analyses/{id}`.
- A single worker thread owns the one `Analyzer` and processes jobs in order. A bounded
  queue (16 by default) returns `503` when full.
- Job state lives in SQLite (stdlib `sqlite3`, WAL mode) under `$DFD_VAR_DIR`. The states
  are `queued`, `processing` (with stage and progress), `completed` and `failed`. On
  restart, jobs left `queued` or `processing` are marked failed, because their uploads
  have already been swept.
- A janitor thread deletes expired results and evidence (24 h TTL by default).
- `analysis_id` is a random UUID4, which also acts as the capability to read a result.
  V1 has no accounts.

## Consequences

- Throughput is one job at a time. This is correct for one GPU, and the production API
  (step 18) can move to a real queue without changing the engine.
- The design is not multi-process. Running several API workers needs a shared store and
  a model per worker. That is a step 18 concern.
