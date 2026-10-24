"""Provenance records for the attack lab, stored as JSONL under lab/registry/.

Identities are pseudonymous: no names, only an ID and a reference to a consent record
kept outside this repository. Every generated sample records how it was made: the
generator, its version and argv, the swapper and its license, the source video, the
donor and target identities, the resolution and the transformation. Any number can then
be traced back to its inputs.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deepfake_bench.paths import LabPaths

Split = Literal["train", "val", "test", "external"]
Label = Literal["real", "manipulated"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identity(_Record):
    identity_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    consent_ref: str
    """Reference to a signed consent and rights record kept outside the repository."""
    split: Split | None = None
    notes: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("consent_ref")
    @classmethod
    def _consent_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("consent_ref is required: no consent, no data")
        return value.strip()


class SourceVideo(_Record):
    video_id: str
    identity_id: str
    path: str
    """Relative to the data root."""
    sha256: str
    consent_ref: str
    width: int
    height: int
    fps: float | None
    duration: float | None
    created_at: datetime = Field(default_factory=utcnow)


class Sample(_Record):
    sample_id: str
    pair_id: str
    """The source video the sample derives from: a real video and every fake made from it
    share one pair_id."""
    label: Label
    split: Split
    path: str
    sha256: str
    generator: str
    """'none' for real videos, 'facefusion', or the name of an external dataset."""
    generator_version: str | None = None
    swapper_model: str | None = None
    swapper_license: str | None = None
    generator_settings: dict[str, Any] = Field(default_factory=dict)
    generator_argv: list[str] = Field(default_factory=list)
    source_video: str
    source_identity: str | None = None
    """The donor whose face was inserted (None for real videos)."""
    target_identity: str | None = None
    """The person who appears in the source video (whose face was replaced)."""
    resolution: str
    fps: float | None
    duration: float | None
    transformation: str = "pristine"
    transformation_argv: list[str] = Field(default_factory=list)
    ffmpeg_version: str | None = None
    parent_sample_id: str | None = None
    manipulation_coverage: float | None = None
    status: Literal["ok", "rejected"] = "ok"
    status_reason: str | None = None
    research_only: bool = False
    created_at: datetime = Field(default_factory=utcnow)


def _write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w") as out:
        for record in records:
            out.write(record.model_dump_json() + "\n")
    Path(tmp).replace(path)


def _read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        return []
    return [
        model.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line
    ]


class Registry:
    """Identities, sources and samples, loaded fully into memory (lab scale is small)."""

    def __init__(self, paths: LabPaths) -> None:
        self.paths = paths.ensure()
        self.identities = {
            i.identity_id: i for i in _read_jsonl(self._file("identities"), Identity)
        }
        self.sources = {s.video_id: s for s in _read_jsonl(self._file("sources"), SourceVideo)}
        self.samples = {s.sample_id: s for s in _read_jsonl(self._file("samples"), Sample)}

    def _file(self, name: str) -> Path:
        return self.paths.registry / f"{name}.jsonl"

    def save(self) -> None:
        _write_jsonl(self._file("identities"), self.identities.values())
        _write_jsonl(self._file("sources"), self.sources.values())
        _write_jsonl(self._file("samples"), self.samples.values())

    def ok_samples(self) -> list[Sample]:
        return [s for s in self.samples.values() if s.status == "ok"]
