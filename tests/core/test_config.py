from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from deepfake_core.config import PipelineConfig, SamplingConfig


def test_config_hash_is_stable_and_sensitive_to_thresholds() -> None:
    assert PipelineConfig().config_hash() == PipelineConfig().config_hash()
    changed = PipelineConfig(sampling=SamplingConfig(rate_hz=2.0))
    assert changed.config_hash() != PipelineConfig().config_hash()


def test_from_file_merges_overrides_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("profile: test\nsampling:\n  rate_hz: 2.5\n")
    config = PipelineConfig.from_file(path)
    assert config.profile == "test"
    assert config.sampling.rate_hz == 2.5
    assert config.sampling.max_samples == SamplingConfig().max_samples


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("sampling:\n  rate: 2\n")
    with pytest.raises(ValidationError):
        PipelineConfig.from_file(path)
