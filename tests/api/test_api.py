from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from deepfake_api.fake import FakeAnalyzer
from deepfake_api.main import create_app
from deepfake_api.settings import Settings
from deepfake_core.errors import ModelMissingError
from deepfake_core.schemas import AnalysisResult
from tests.conftest import animate, face_patch

MakeVideo = Callable[..., Path]


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {"var_dir": tmp_path / "var", "analyzer": "fake", "web_dist": None}
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings_for(tmp_path))) as test_client:
        yield test_client


def upload(client: TestClient, path: Path, name: str | None = None) -> Any:
    with path.open("rb") as handle:
        return client.post(
            "/api/v1/analyses",
            files={"file": (name or path.name, handle, "application/octet-stream")},
        )


def wait_for(client: TestClient, analysis_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body: dict[str, Any] = client.get(f"/api/v1/analyses/{analysis_id}").json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"analysis {analysis_id} did not finish")


def uploads_dir(tmp_path: Path) -> list[Path]:
    return list((tmp_path / "var" / "uploads").iterdir())


# --------------------------------------------------------------------------- happy path


def test_health_and_models(client: TestClient) -> None:
    deadline = time.monotonic() + 5
    while (
        client.get("/api/v1/health").json()["status"] == "loading" and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    health = client.get("/api/v1/health").json()
    assert health["status"] == "ok" and health["analyzer"] == "fake"
    models = client.get("/api/v1/models").json()
    assert models["loaded"] is True
    assert {m["role"] for m in models["models"]} == {"face_detector", "frame_detector"}


def test_upload_analyse_and_fetch_result(
    client: TestClient, make_video: MakeVideo, tmp_path: Path
) -> None:
    response = upload(client, make_video("clip.mp4", frames=60, fps=20))
    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "queued"
    assert created["status_url"] == f"/api/v1/analyses/{created['analysis_id']}"

    body = wait_for(client, created["analysis_id"])
    assert body["status"] == "completed", body
    result = AnalysisResult.model_validate(body["result"])
    assert result.analysis_id == created["analysis_id"]
    assert result.processing.analyzer == "fake"
    assert uploads_dir(tmp_path) == []  # the original video is gone

    evidence = client.get(
        f"/api/v1/analyses/{result.analysis_id}/evidence/{result.evidence[0].evidence_id}"
    )
    assert evidence.status_code == 200
    assert evidence.headers["content-type"] == "image/jpeg"
    assert evidence.headers["cache-control"] == "private, no-store"
    assert evidence.content[:2] == b"\xff\xd8"


def test_security_headers(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


# --------------------------------------------------------------------------- rejections


@pytest.mark.parametrize(
    ("name", "content", "status", "code"),
    [
        ("clip.avi", b"RIFF....AVI ", 415, "unsupported_extension"),
        ("clip.mp4", b"", 400, "empty_file"),
        ("clip.mp4", b"definitely not a video" * 50, 422, "unreadable"),
    ],
)
def test_bad_uploads_are_rejected_and_nothing_is_kept(
    client: TestClient, tmp_path: Path, name: str, content: bytes, status: int, code: str
) -> None:
    response = client.post(
        "/api/v1/analyses", files={"file": (name, content, "application/octet-stream")}
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert uploads_dir(tmp_path) == []


def test_webm_content_named_mp4_is_unsupported(
    client: TestClient, make_video: MakeVideo, tmp_path: Path
) -> None:
    response = upload(client, make_video("clip.webm", frames=40), name="clip.mp4")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "container_mismatch"


def test_upload_size_limit(tmp_path: Path, make_video: MakeVideo) -> None:
    with TestClient(create_app(settings_for(tmp_path, max_upload_mb=1))) as limited:
        big = tmp_path / "big.mp4"
        big.write_bytes(b"\0" * (3 * 1024 * 1024))
        response = upload(limited, big)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "too_large"
        assert uploads_dir(tmp_path) == []


def test_missing_file_field(client: TestClient) -> None:
    response = client.post("/api/v1/analyses", data={"not_file": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("analysis_id", ["0" * 32, "../../etc/passwd", "ABC"])
def test_unknown_ids_are_404(client: TestClient, analysis_id: str) -> None:
    assert client.get(f"/api/v1/analyses/{analysis_id}").status_code == 404
    assert client.delete(f"/api/v1/analyses/{analysis_id}").status_code == 404
    assert client.get(f"/api/v1/analyses/{analysis_id}/evidence/e0").status_code == 404


# --------------------------------------------------------------------------- privacy & retention


def test_delete_removes_result_and_evidence(
    client: TestClient, make_video: MakeVideo, tmp_path: Path
) -> None:
    analysis_id = upload(client, make_video("clip.mp4", frames=60)).json()["analysis_id"]
    wait_for(client, analysis_id)
    assert (tmp_path / "var" / "evidence" / analysis_id).is_dir()
    assert client.delete(f"/api/v1/analyses/{analysis_id}").status_code == 204
    assert client.get(f"/api/v1/analyses/{analysis_id}").status_code == 404
    assert not (tmp_path / "var" / "evidence" / analysis_id).exists()


def test_results_expire_after_the_ttl(tmp_path: Path, make_video: MakeVideo) -> None:
    clock = MutableClock()
    app = create_app(settings_for(tmp_path, result_ttl_hours=24), clock=clock)
    with TestClient(app) as ttl_client:
        analysis_id = upload(ttl_client, make_video("clip.mp4", frames=60)).json()["analysis_id"]
        wait_for(ttl_client, analysis_id)
        janitor = app.state.services.janitor
        clock.now += timedelta(hours=23)
        assert janitor.sweep() == 0
        clock.now += timedelta(hours=2)
        assert janitor.sweep() == 1
        assert ttl_client.get(f"/api/v1/analyses/{analysis_id}").status_code == 404
        assert not (tmp_path / "var" / "evidence" / analysis_id).exists()


def test_filenames_are_never_stored(
    client: TestClient, make_video: MakeVideo, tmp_path: Path
) -> None:
    path = make_video("clip.mp4", frames=60)
    analysis_id = upload(client, path, name="Jane-Doe interview 2026.mp4").json()["analysis_id"]
    wait_for(client, analysis_id)
    db = sqlite3.connect(tmp_path / "var" / "analyses.sqlite3")
    dump = "\n".join(db.iterdump())
    assert "jane" not in dump.lower()
    names = " ".join(str(p) for p in (tmp_path / "var").rglob("*"))
    assert "jane" not in names.lower()


# --------------------------------------------------------------------------- worker behaviour


class FailingAnalyzer(FakeAnalyzer):
    def analyze(self, path: Path, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


class BlockingAnalyzer(FakeAnalyzer):
    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self.release = release

    def analyze(self, path: Path, **kwargs: Any) -> Any:
        self.release.wait(10)
        return super().analyze(path, **kwargs)


def test_unexpected_failure_is_reported_and_upload_deleted(
    tmp_path: Path, make_video: MakeVideo
) -> None:
    app = create_app(settings_for(tmp_path), analyzer_factory=FailingAnalyzer)
    with TestClient(app) as failing:
        analysis_id = upload(failing, make_video("clip.mp4", frames=60)).json()["analysis_id"]
        body = wait_for(failing, analysis_id)
        assert body["status"] == "failed"
        assert body["error"]["code"] == "internal_error"
        assert "boom" not in body["error"]["message"]
        assert uploads_dir(tmp_path) == []


def test_missing_models_fail_jobs_cleanly(tmp_path: Path, make_video: MakeVideo) -> None:
    def no_models() -> Any:
        raise ModelMissingError("GenD not found")

    with TestClient(create_app(settings_for(tmp_path), analyzer_factory=no_models)) as broken:
        health = broken.get("/api/v1/health").json()
        deadline = time.monotonic() + 5
        while health["status"] == "loading" and time.monotonic() < deadline:
            time.sleep(0.02)
            health = broken.get("/api/v1/health").json()
        assert health["status"] == "error" and "GenD not found" in health["detail"]
        assert broken.get("/api/v1/models").json()["loaded"] is False
        analysis_id = upload(broken, make_video("clip.mp4", frames=60)).json()["analysis_id"]
        assert wait_for(broken, analysis_id)["error"]["code"] == "models_unavailable"


def test_queue_full_returns_503(tmp_path: Path, make_video: MakeVideo) -> None:
    release = threading.Event()
    app = create_app(
        settings_for(tmp_path, max_queue=1), analyzer_factory=lambda: BlockingAnalyzer(release)
    )
    with TestClient(app) as busy:
        video = make_video("clip.mp4", frames=60)
        responses = []
        for _ in range(4):
            responses.append(upload(busy, video))
            time.sleep(0.1)  # let the worker pick up the first job
        # one job running, one waiting in the queue (max_queue=1), the rest turned away
        assert [r.status_code for r in responses] == [202, 202, 503, 503]
        assert all(r.json()["error"]["code"] == "busy" for r in responses[2:])
        assert responses[2].headers["retry-after"] == "30"
        release.set()
        for response in responses[:2]:
            assert wait_for(busy, response.json()["analysis_id"])["status"] == "completed"
        assert uploads_dir(tmp_path) == []


def test_delete_while_processing_keeps_nothing(tmp_path: Path, make_video: MakeVideo) -> None:
    release = threading.Event()
    app = create_app(settings_for(tmp_path), analyzer_factory=lambda: BlockingAnalyzer(release))
    with TestClient(app) as slow:
        analysis_id = upload(slow, make_video("clip.mp4", frames=60)).json()["analysis_id"]
        time.sleep(0.2)
        assert slow.delete(f"/api/v1/analyses/{analysis_id}").status_code == 204
        release.set()
        time.sleep(0.5)
        assert not (tmp_path / "var" / "evidence" / analysis_id).exists()
        assert uploads_dir(tmp_path) == []


def test_orphaned_jobs_fail_after_restart(tmp_path: Path, make_video: MakeVideo) -> None:
    release = threading.Event()
    app = create_app(settings_for(tmp_path), analyzer_factory=lambda: BlockingAnalyzer(release))
    with TestClient(app) as first:
        analysis_id = upload(first, make_video("clip.mp4", frames=60)).json()["analysis_id"]
        stray = tmp_path / "var" / "uploads" / "leftover.mp4"
        stray.write_bytes(b"x")
        release.set()  # let the worker drain so shutdown is quick
        wait_for(first, analysis_id)
        db = sqlite3.connect(tmp_path / "var" / "analyses.sqlite3")
        db.execute("UPDATE analyses SET status = 'processing'")  # simulate a crash mid-job
        db.commit()
        db.close()
    with TestClient(create_app(settings_for(tmp_path))) as second:
        body = second.get(f"/api/v1/analyses/{analysis_id}").json()
        assert body["status"] == "failed"
        assert body["error"]["code"] == "server_restarted"
        assert uploads_dir(tmp_path) == []


# --------------------------------------------------------------------------- SPA serving


def test_spa_fallback_and_api_404(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    with TestClient(create_app(settings_for(tmp_path, web_dist=dist))) as spa:
        assert "<title>app</title>" in spa.get("/").text
        assert "<title>app</title>" in spa.get("/analyses/0123").text
        assert spa.get("/assets/app.js").text == "console.log(1)"
        missing = spa.get("/api/v1/nope")
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
        assert spa.get("/../../etc/passwd").status_code in {200, 404}  # never escapes dist
        assert "root:" not in spa.get("/../../etc/passwd").text


# --------------------------------------------------------------------------- real models


@pytest.mark.models
def test_real_analyzer_through_the_api(
    tmp_path: Path, make_video: MakeVideo, portraits: dict[str, Any], yunet: Any
) -> None:
    patch = face_patch(portraits["mae_jemison"], yunet, side=300)
    video = make_video("face.mp4", frames=animate([(patch, (330, 260), (10, 0))], seconds=3.0))
    try:
        from deepfake_core.detectors.gend import GenDDetector  # noqa: F401
    except ModelMissingError:
        pytest.skip("weights missing")
    with TestClient(create_app(settings_for(tmp_path, analyzer="real"))) as real:
        analysis_id = upload(real, video).json()["analysis_id"]
        body = wait_for(real, analysis_id, timeout=120)
        assert body["status"] == "completed", body
        result = AnalysisResult.model_validate(body["result"])
        assert result.processing.analyzer == "real"
        assert result.face_tracks == 1
        assert uploads_dir(tmp_path) == []
