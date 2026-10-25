"""Layout of the research data root ($DFD_DATA_ROOT, default ./data). Never the API's var/."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT_ENV = "DFD_DATA_ROOT"


@dataclass(frozen=True, slots=True)
class LabPaths:
    root: Path

    @classmethod
    def resolve(cls, root: str | os.PathLike[str] | None = None) -> LabPaths:
        return cls(Path(root if root is not None else os.environ.get(DATA_ROOT_ENV, "data")))

    @property
    def lab(self) -> Path:
        return self.root / "lab"

    @property
    def registry(self) -> Path:
        return self.lab / "registry"

    @property
    def sources(self) -> Path:
        return self.lab / "sources"

    @property
    def canon(self) -> Path:
        return self.lab / "canon"

    @property
    def donors(self) -> Path:
        return self.lab / "donors"

    @property
    def raw_fakes(self) -> Path:
        return self.lab / "raw_fakes"

    @property
    def samples(self) -> Path:
        return self.lab / "samples"

    @property
    def runs(self) -> Path:
        return self.root / "bench" / "runs"

    @property
    def temporal(self) -> Path:
        """Temporal-model checkpoints trained on lab traces, with their training records."""
        return self.root / "bench" / "temporal"

    def ensure(self) -> LabPaths:
        for directory in (
            self.registry,
            self.sources,
            self.canon,
            self.donors,
            self.raw_fakes,
            self.samples,
            self.runs,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.lab.chmod(0o700)
        return self

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root.resolve()))

    def absolute(self, relative: str) -> Path:
        return self.root / relative
