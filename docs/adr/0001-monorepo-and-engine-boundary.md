# ADR 0001: Monorepo with an engine independent of the web layers

- Status: accepted
- Date: 2026-10-24

## Context

V1 is a web upload detector. The same detection logic must later serve an API, a
benchmark, a near-real-time streaming engine and Teams/Zoom integrations. If the
detection code grows inside a web app, every later product has to untangle it.

## Decision

- One repository, laid out as in the handoff doc: `packages/detector` (engine),
  `apps/api`, `apps/web`, `benchmarks`, `tools`, `docs`, `tests`.
- Python packages form a uv workspace on Python 3.12. The web app is a separate npm
  project.
- The engine (`deepfake-core`) has no dependency on FastAPI, the benchmark, or any
  plotting or ML-evaluation library. An AST-based test enforces this.
- The web UI is a React + Vite single-page app that talks to the API over HTTP only. In
  production FastAPI serves the built files, so everything is one origin. In development
  Vite proxies `/api`.
- The benchmark package is internal and never distributed. It may use research-only
  tools and data that the product must not ship.

## Consequences

- The engine's public surface (`Analyzer`, `analyze_video`, `AnalysisResult`) is the
  contract every front end uses. Changing it requires a `schema_version` bump.
- The web UI components take an `AnalysisResult` and nothing else, so they can be reused
  in later meeting-app panels.
- A single `make check` covers Python and web, lint, types, tests and licenses.
