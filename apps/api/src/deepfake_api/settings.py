"""API settings, read from DFD_* environment variables."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/src/deepfake_api/settings.py -> apps/web/dist (present after `make build-web`)
DEFAULT_WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DFD_", extra="ignore")

    var_dir: Path = Field(default=Path("var"), description="Uploads, job database, evidence")
    models_dir: Path | None = Field(default=None, description="Defaults to ./models")
    device: str | None = Field(default=None, description="auto, cuda, mps or cpu")
    analyzer: Literal["real", "fake"] = Field(
        default="real", description="fake: canned results for UI development and tests"
    )
    config_file: Path | None = Field(default=None, description="PipelineConfig overrides")
    max_upload_mb: int = Field(default=500, gt=0)
    result_ttl_hours: float = Field(default=24.0, gt=0)
    max_queue: int = Field(default=16, gt=0)
    janitor_interval_s: float = Field(default=300.0, gt=0)
    web_dist: Path | None = Field(default=DEFAULT_WEB_DIST, description="Built SPA to serve")
    fake_delay_s: float = Field(default=0.0, ge=0, description="Simulated work in fake mode")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def result_ttl(self) -> timedelta:
        return timedelta(hours=self.result_ttl_hours)
