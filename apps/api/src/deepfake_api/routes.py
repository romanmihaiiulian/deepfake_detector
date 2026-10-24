"""HTTP routes under /api/v1."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from deepfake_api import __version__
from deepfake_api.jobs import Clock, Janitor, JobStatus, JobStore, QueueFullError, Worker
from deepfake_api.settings import Settings
from deepfake_api.storage import Storage, UploadTooLargeError
from deepfake_core.config import PipelineConfig
from deepfake_core.errors import VideoValidationError
from deepfake_core.models import MODELS
from deepfake_core.pipeline import model_info
from deepfake_core.schemas import AnalysisResult, ModelInfo
from deepfake_core.video import ALLOWED_EXTENSIONS, validate

router = APIRouter()

_ANALYSIS_ID = re.compile(r"^[0-9a-f]{32}$")
_STATUS_FOR_CODE = {
    "unsupported_extension": 415,
    "container_mismatch": 415,
    "empty_file": 400,
}


@dataclass(slots=True)
class Services:
    settings: Settings
    pipeline: PipelineConfig
    storage: Storage
    store: JobStore
    worker: Worker
    janitor: Janitor
    clock: Clock


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class AnalysisCreated(BaseModel):
    analysis_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class AnalysisStatus(BaseModel):
    analysis_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    stage: str | None
    progress: float
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    error: ErrorBody | None = None
    result: AnalysisResult | None = None


class Health(BaseModel):
    status: Literal["ok", "loading", "error"]
    models_loaded: bool
    analyzer: Literal["real", "fake"]
    device: str | None
    queue_depth: int
    version: str
    detail: str | None = None


class ModelsResponse(BaseModel):
    loaded: bool
    models: list[ModelInfo]


ERRORS: dict[int | str, dict[str, object]] = {
    status: {"model": ErrorResponse} for status in (400, 404, 413, 415, 422, 503)
}


def services(request: Request) -> Services:
    value: Services = request.app.state.services
    return value


def _not_found() -> ApiError:
    return ApiError(404, "not_found", "No such analysis. Results are deleted after they expire.")


@router.post(
    "/analyses",
    status_code=202,
    response_model=AnalysisCreated,
    responses=ERRORS,
    summary="Upload a video for analysis",
)
async def create_analysis(
    request: Request, file: Annotated[UploadFile, File(description="MP4, MOV or WEBM video")]
) -> AnalysisCreated:
    svc = services(request)
    max_bytes = svc.settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + 1024 * 1024:
        raise ApiError(413, "too_large", f"Uploads are limited to {svc.settings.max_upload_mb} MB.")

    # Only the extension is taken from the client's filename; the name itself is discarded.
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ApiError(415, "unsupported_extension", "Supported formats are MP4, MOV and WEBM.")
    if svc.worker.queue_depth >= svc.settings.max_queue:
        raise ApiError(503, "busy", "The server is busy. Try again shortly.", {"Retry-After": "30"})

    analysis_id = uuid.uuid4().hex
    upload = svc.storage.upload_path(analysis_id, extension)
    try:
        size = await svc.storage.save_upload(file, upload, max_bytes)
        if size == 0:
            raise ApiError(400, "empty_file", "The file is empty.")
        try:
            await run_in_threadpool(validate, upload, svc.pipeline.validation)
        except VideoValidationError as error:
            raise ApiError(
                _STATUS_FOR_CODE.get(error.code, 422), error.code, error.message
            ) from error
        svc.store.create(analysis_id, svc.clock(), svc.settings.result_ttl)
        try:
            svc.worker.submit(analysis_id, upload)
        except QueueFullError as error:
            svc.store.delete(analysis_id)
            raise ApiError(
                503, "busy", "The server is busy. Try again shortly.", {"Retry-After": "30"}
            ) from error
    except UploadTooLargeError as error:
        svc.storage.delete_upload(upload)
        raise ApiError(
            413, "too_large", f"Uploads are limited to {svc.settings.max_upload_mb} MB."
        ) from error
    except BaseException:
        svc.storage.delete_upload(upload)
        raise
    finally:
        await file.close()
    return AnalysisCreated(analysis_id=analysis_id, status_url=f"/api/v1/analyses/{analysis_id}")


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisStatus,
    responses=ERRORS,
    summary="Status and, once completed, the result",
)
def get_analysis(request: Request, analysis_id: str) -> AnalysisStatus:
    svc = services(request)
    job = svc.store.get(analysis_id) if _ANALYSIS_ID.match(analysis_id) else None
    if job is None:
        raise _not_found()
    return AnalysisStatus(
        analysis_id=job.analysis_id,
        status=job.status.value,
        stage=job.stage,
        progress=job.progress,
        created_at=job.created_at,
        updated_at=job.updated_at,
        expires_at=job.expires_at,
        error=ErrorBody(code=job.error_code, message=job.error_message or "")
        if job.status is JobStatus.FAILED and job.error_code
        else None,
        result=AnalysisResult.model_validate_json(job.result_json) if job.result_json else None,
    )


@router.get(
    "/analyses/{analysis_id}/evidence/{evidence_id}",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse}, 200: {"content": {"image/jpeg": {}}}},
    summary="An evidence face crop (JPEG)",
)
def get_evidence(request: Request, analysis_id: str, evidence_id: str) -> FileResponse:
    svc = services(request)
    path = (
        svc.storage.evidence_file(analysis_id, evidence_id)
        if _ANALYSIS_ID.match(analysis_id)
        else None
    )
    if path is None:
        raise ApiError(404, "not_found", "No such evidence image.")
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"}
    )


@router.delete(
    "/analyses/{analysis_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a result and its evidence now",
)
def delete_analysis(request: Request, analysis_id: str) -> Response:
    svc = services(request)
    if not _ANALYSIS_ID.match(analysis_id) or not svc.store.delete(analysis_id):
        raise _not_found()
    svc.storage.delete_evidence(analysis_id)
    return Response(status_code=204)


@router.get("/health", response_model=Health, summary="Liveness and model status")
def health(request: Request) -> Health:
    svc = services(request)
    worker = svc.worker
    if not worker.ready.is_set():
        status: Literal["ok", "loading", "error"] = "loading"
    elif worker.analyzer is None:
        status = "error"
    else:
        status = "ok"
    return Health(
        status=status,
        models_loaded=worker.analyzer is not None,
        analyzer=svc.settings.analyzer,
        device=worker.analyzer.device if worker.analyzer is not None else None,
        queue_depth=worker.queue_depth,
        version=__version__,
        detail=worker.load_error,
    )


@router.get("/models", response_model=ModelsResponse, summary="Models used for analysis")
def list_models(request: Request) -> ModelsResponse:
    analyzer = services(request).worker.analyzer
    if analyzer is not None:
        return ModelsResponse(loaded=True, models=list(analyzer.models))
    return ModelsResponse(loaded=False, models=[model_info(spec) for spec in MODELS])
