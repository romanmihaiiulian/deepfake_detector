from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from deepfake_core import models
from deepfake_core.errors import ModelIntegrityError, ModelMissingError


def local_spec(tmp_path: Path, payload: bytes, *, sha256: str | None = None) -> models.ModelSpec:
    source = tmp_path / "upstream.bin"
    source.write_bytes(payload)
    return dataclasses.replace(
        models.YUNET,
        path="fake/model.bin",
        url=source.as_uri(),
        sha256=sha256 or hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def test_fetch_downloads_and_verifies(tmp_path: Path) -> None:
    spec = local_spec(tmp_path, b"weights" * 1000)
    root = tmp_path / "models"
    seen: list[int] = []
    path = models.fetch(spec, root, progress=lambda received, total: seen.append(received))
    assert path.read_bytes() == b"weights" * 1000
    assert seen[-1] == spec.size
    assert models.verify(spec, root) == path


def test_fetch_rejects_checksum_mismatch_and_leaves_nothing_behind(tmp_path: Path) -> None:
    spec = local_spec(tmp_path, b"tampered", sha256="0" * 64)
    root = tmp_path / "models"
    with pytest.raises(ModelIntegrityError):
        models.fetch(spec, root)
    assert list((root / "fake").iterdir()) == []


def test_fetch_replaces_a_corrupt_file(tmp_path: Path) -> None:
    spec = local_spec(tmp_path, b"good bytes")
    root = tmp_path / "models"
    target = models.model_path(spec, root)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad bytes!")  # same size, wrong content
    models.fetch(spec, root)
    assert target.read_bytes() == b"good bytes"


def test_require_reports_missing_and_truncated_files(tmp_path: Path) -> None:
    spec = local_spec(tmp_path, b"0123456789")
    with pytest.raises(ModelMissingError, match="make models"):
        models.require(spec, tmp_path / "empty")
    target = models.model_path(spec, tmp_path / "m")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"0123")
    with pytest.raises(ModelIntegrityError, match="expected 10"):
        models.require(spec, tmp_path / "m")


def test_models_dir_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(models.MODELS_DIR_ENV, raising=False)
    assert models.models_dir() == Path("models")
    monkeypatch.setenv(models.MODELS_DIR_ENV, str(tmp_path))
    assert models.models_dir() == tmp_path
    assert models.models_dir("explicit") == Path("explicit")


def test_manifest_pins_are_well_formed() -> None:
    for spec in models.MODELS:
        assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
        assert len(spec.revision) == 40
        assert spec.revision in spec.url
        assert spec.size > 0
