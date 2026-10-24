"""Pipeline configuration: every threshold the engine uses, in one versioned place.

Values in the `uncalibrated-v0` profile are starting points. The benchmark (M7/M8)
measures them on paired lab data, and `profile` changes whenever they are retuned.
`config_hash()` goes into every result, so a result can always be traced back to the
exact thresholds that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ValidationConfig(_Section):
    min_duration_s: float = 1.0
    max_duration_s: float = 600.0
    max_dimension_px: int = Field(default=4096, description="Longest coded side accepted")


class SamplingConfig(_Section):
    rate_hz: float = Field(default=4.0, gt=0, description="Target samples per second")
    max_samples: int = Field(
        default=960, gt=0, description="Rate is lowered so a video yields at most this"
    )
    duplicate_max_abs_diff: int = Field(
        default=2,
        ge=0,
        description="Max luma difference (0-255) for consecutive frames to be duplicates",
    )


class PipelineConfig(_Section):
    profile: str = "uncalibrated-v0"
    validation: ValidationConfig = ValidationConfig()
    sampling: SamplingConfig = SamplingConfig()

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        """Load overrides from YAML or JSON; unspecified values keep their defaults."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
