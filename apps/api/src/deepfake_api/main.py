"""Application factory: `uvicorn deepfake_api.main:create_app --factory`."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from deepfake_api import __version__
from deepfake_api.jobs import AnalyzerLike, Clock, Janitor, JobStore, Worker
from deepfake_api.routes import ApiError, Services, router
from deepfake_api.settings import Settings
from deepfake_api.storage import Storage
from deepfake_core.config import PipelineConfig

logger = logging.getLogger("deepfake_api")

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "img-src 'self' data: blob:",
        "style-src 'self' 'unsafe-inline'",
        "script-src 'self'",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'self'",
    ]
)


def default_analyzer_factory(
    settings: Settings, pipeline: PipelineConfig
) -> Callable[[], AnalyzerLike]:
    def build() -> AnalyzerLike:
        if settings.analyzer == "fake":
            from deepfake_api.fake import FakeAnalyzer

            return FakeAnalyzer(settings.fake_delay_s, pipeline)
        from deepfake_core.pipeline import Analyzer

        return Analyzer(pipeline, models_dir=settings.models_dir, device=settings.device)

    return build


def _mount_spa(app: FastAPI, dist: Path) -> None:
    dist = dist.resolve()
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            raise ApiError(404, "not_found", "Unknown API route.")
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})


def create_app(
    settings: Settings | None = None,
    *,
    analyzer_factory: Callable[[], AnalyzerLike] | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    settings = settings or Settings()
    pipeline = (
        PipelineConfig.from_file(settings.config_file) if settings.config_file else PipelineConfig()
    )
    now: Clock = clock or (lambda: datetime.now(UTC))
    factory = analyzer_factory or default_analyzer_factory(settings, pipeline)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage = Storage(settings.var_dir)
        swept = storage.sweep_uploads()
        store = JobStore(storage.db_path)
        orphans = store.fail_orphans(now(), settings.result_ttl)
        if swept or orphans:
            logger.info(
                "startup: removed %d stale uploads, failed %d orphaned jobs", swept, orphans
            )
        worker = Worker(
            store=store,
            storage=storage,
            analyzer_factory=factory,
            ttl=settings.result_ttl,
            max_queue=settings.max_queue,
            clock=now,
        )
        janitor = Janitor(store, storage, settings.janitor_interval_s, now)
        janitor.sweep()
        worker.start()
        janitor.start()
        app.state.services = Services(settings, pipeline, storage, store, worker, janitor, now)
        try:
            yield
        finally:
            janitor.stop()
            worker.stop()
            store.close()

    app = FastAPI(
        title="Deepfake Detector API",
        version=__version__,
        description=(
            "Upload a video and receive a probabilistic assessment of face manipulation. "
            "Results are not proof and must not be the sole basis for decisions about people."
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": error.code, "message": error.message}},
            status_code=error.status,
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e.get("loc", ())[1:]) for e in error.errors()})
        message = f"Invalid request: {', '.join(fields) or 'malformed body'}."
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": message}}, status_code=422
        )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if not request.url.path.startswith("/api/docs"):  # Swagger UI loads from a CDN
            response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        return response

    app.include_router(router, prefix="/api/v1")
    if settings.web_dist is not None and (settings.web_dist / "index.html").is_file():
        _mount_spa(app, settings.web_dist)
    return app
