"""Job state (SQLite), the single analysis worker, and the retention janitor (ADR 0005)."""

from __future__ import annotations

import contextlib
import logging
import queue
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from deepfake_api.storage import Storage
from deepfake_core.errors import VideoValidationError
from deepfake_core.pipeline import AnalysisOutput, ProgressCallback
from deepfake_core.schemas import ModelInfo

logger = logging.getLogger("deepfake_api.jobs")

Clock = Callable[[], datetime]


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Job:
    analysis_id: str
    status: JobStatus
    stage: str | None
    progress: float
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    error_code: str | None
    error_message: str | None
    result_json: str | None


class AnalyzerLike(Protocol):
    device: str
    models: list[ModelInfo]

    def warmup(self) -> None: ...

    def analyze(
        self,
        path: Path,
        *,
        analysis_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> AnalysisOutput: ...


class JobStore:
    """Thread-safe SQLite job table. Stores no filenames, no embeddings, no frames."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id   TEXT PRIMARY KEY,
            status        TEXT NOT NULL,
            stage         TEXT,
            progress      REAL NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            expires_at    TEXT NOT NULL,
            error_code    TEXT,
            error_message TEXT,
            result_json   TEXT
        )
    """

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(self._SCHEMA)
        path.chmod(0o600)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, params)

    def create(self, analysis_id: str, now: datetime, ttl: timedelta) -> None:
        self._execute(
            "INSERT INTO analyses (analysis_id, status, progress, created_at, updated_at,"
            " expires_at) VALUES (?, ?, 0, ?, ?, ?)",
            (
                analysis_id,
                JobStatus.QUEUED,
                now.isoformat(),
                now.isoformat(),
                (now + ttl).isoformat(),
            ),
        )

    def get(self, analysis_id: str) -> Job | None:
        row = self._execute(
            "SELECT analysis_id, status, stage, progress, created_at, updated_at, expires_at,"
            " error_code, error_message, result_json FROM analyses WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
        if row is None:
            return None
        return Job(
            analysis_id=row[0],
            status=JobStatus(row[1]),
            stage=row[2],
            progress=row[3],
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5]),
            expires_at=datetime.fromisoformat(row[6]),
            error_code=row[7],
            error_message=row[8],
            result_json=row[9],
        )

    def update_progress(self, analysis_id: str, stage: str, progress: float, now: datetime) -> None:
        self._execute(
            "UPDATE analyses SET status = ?, stage = ?, progress = ?, updated_at = ?"
            " WHERE analysis_id = ? AND status IN (?, ?)",
            (
                JobStatus.PROCESSING,
                stage,
                round(progress, 4),
                now.isoformat(),
                analysis_id,
                JobStatus.QUEUED,
                JobStatus.PROCESSING,
            ),
        )

    def complete(self, analysis_id: str, result_json: str, now: datetime, ttl: timedelta) -> bool:
        cursor = self._execute(
            "UPDATE analyses SET status = ?, stage = 'done', progress = 1, result_json = ?,"
            " updated_at = ?, expires_at = ? WHERE analysis_id = ?",
            (
                JobStatus.COMPLETED,
                result_json,
                now.isoformat(),
                (now + ttl).isoformat(),
                analysis_id,
            ),
        )
        return cursor.rowcount > 0

    def fail(
        self, analysis_id: str, code: str, message: str, now: datetime, ttl: timedelta
    ) -> bool:
        cursor = self._execute(
            "UPDATE analyses SET status = ?, error_code = ?, error_message = ?, updated_at = ?,"
            " expires_at = ? WHERE analysis_id = ?",
            (
                JobStatus.FAILED,
                code,
                message,
                now.isoformat(),
                (now + ttl).isoformat(),
                analysis_id,
            ),
        )
        return cursor.rowcount > 0

    def delete(self, analysis_id: str) -> bool:
        return (
            self._execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,)).rowcount > 0
        )

    def expired(self, now: datetime) -> list[str]:
        rows = self._execute(
            "SELECT analysis_id FROM analyses WHERE expires_at <= ?", (now.isoformat(),)
        ).fetchall()
        return [row[0] for row in rows]

    def fail_orphans(self, now: datetime, ttl: timedelta) -> int:
        """Jobs left queued or processing by a previous process can never finish: their
        uploads were swept. Mark them failed."""
        cursor = self._execute(
            "UPDATE analyses SET status = ?, error_code = 'server_restarted', error_message = ?,"
            " updated_at = ?, expires_at = ? WHERE status IN (?, ?)",
            (
                JobStatus.FAILED,
                "The server restarted before this analysis finished. Please upload again.",
                now.isoformat(),
                (now + ttl).isoformat(),
                JobStatus.QUEUED,
                JobStatus.PROCESSING,
            ),
        )
        return cursor.rowcount


class QueueFullError(Exception):
    pass


class Worker:
    """One thread, one analyzer: loads the models, then processes jobs in order."""

    def __init__(
        self,
        *,
        store: JobStore,
        storage: Storage,
        analyzer_factory: Callable[[], AnalyzerLike],
        ttl: timedelta,
        max_queue: int,
        clock: Clock,
    ) -> None:
        self._store = store
        self._storage = storage
        self._factory = analyzer_factory
        self._ttl = ttl
        self._clock = clock
        self._queue: queue.Queue[tuple[str, Path] | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name="analysis-worker", daemon=True)
        self.analyzer: AnalyzerLike | None = None
        self.load_error: str | None = None
        self.ready = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=timeout)
        self._thread.join(timeout)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def submit(self, analysis_id: str, upload: Path) -> None:
        try:
            self._queue.put_nowait((analysis_id, upload))
        except queue.Full as exc:
            raise QueueFullError from exc

    def _run(self) -> None:
        try:
            self.analyzer = self._factory()
            self.analyzer.warmup()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("analyzer failed to load")
        finally:
            self.ready.set()
        while (item := self._queue.get()) is not None:
            self._process(*item)

    def _process(self, analysis_id: str, upload: Path) -> None:
        try:
            if self._store.get(analysis_id) is None:
                return  # deleted while queued
            if self.analyzer is None:
                self._store.fail(
                    analysis_id,
                    "models_unavailable",
                    "The detection models are not available on this server.",
                    self._clock(),
                    self._ttl,
                )
                return
            last: list[tuple[str, float]] = [("", -1.0)]

            def progress(stage: str, fraction: float) -> None:
                if stage != last[0][0] or fraction - last[0][1] >= 0.02:
                    last[0] = (stage, fraction)
                    self._store.update_progress(analysis_id, stage, fraction, self._clock())

            output = self.analyzer.analyze(upload, analysis_id=analysis_id, progress=progress)
            if self._store.get(analysis_id) is None:
                return  # deleted while processing: keep nothing
            self._storage.write_evidence(analysis_id, output.evidence)
            if not self._store.complete(
                analysis_id, output.result.model_dump_json(), self._clock(), self._ttl
            ):
                self._storage.delete_evidence(analysis_id)
        except VideoValidationError as error:
            self._store.fail(analysis_id, error.code, error.message, self._clock(), self._ttl)
        except Exception:
            logger.exception("analysis %s failed", analysis_id)
            self._store.fail(
                analysis_id,
                "internal_error",
                "The analysis failed unexpectedly.",
                self._clock(),
                self._ttl,
            )
        finally:
            self._storage.delete_upload(upload)  # the original video never outlives its analysis


class Janitor:
    """Deletes results and evidence once they expire."""

    def __init__(self, store: JobStore, storage: Storage, interval_s: float, clock: Clock) -> None:
        self._store = store
        self._storage = storage
        self._interval = interval_s
        self._clock = clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="retention-janitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)

    def sweep(self) -> int:
        expired = self._store.expired(self._clock())
        for analysis_id in expired:
            self._storage.delete_evidence(analysis_id)
            self._store.delete(analysis_id)
        return len(expired)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.sweep()
            except Exception:
                logger.exception("retention sweep failed")
