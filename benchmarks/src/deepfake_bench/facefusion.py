"""FaceFusion as an external tool: built argv, subprocess, recorded provenance.

FaceFusion (OpenRAIL-AS) is installed separately in its own environment
(`tools/install_facefusion.sh`, $FACEFUSION_HOME). It is never imported, so its license
and dependencies stay outside this project. The flags below match FaceFusion 3.9.1
(facefusion/program.py and processors/modules/face_swapper/core.py at tag 3.9.1).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FACEFUSION_HOME_ENV = "FACEFUSION_HOME"
PINNED_VERSION = "3.9.1"

# From FaceFusion 3.9.1 model metadata (processors/modules/face_swapper/core.py).
SWAPPER_LICENSES: dict[str, str] = {
    "alphaface_256": "Non-Commercial",
    "blendswap_256": "Non-Commercial",
    "ghost_1_256": "Apache-2.0",
    "ghost_2_256": "Apache-2.0",
    "ghost_3_256": "Apache-2.0",
    "hififace_unofficial_256": "Unknown",
    "hyperswap_1a_256": "ResearchRAIL",
    "hyperswap_1b_256": "ResearchRAIL",
    "hyperswap_1c_256": "ResearchRAIL",
    "inswapper_128": "Non-Commercial",
    "inswapper_128_fp16": "Non-Commercial",
    "simswap_256": "Non-Commercial",
    "simswap_unofficial_512": "Non-Commercial",
    "uniface_256": "Unknown",
}


class FaceFusionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SwapSettings:
    swapper_model: str
    pixel_boost: str | None = None
    weight: float | None = None
    face_selector_mode: str = "one"
    face_mask_types: tuple[str, ...] = ("box",)
    face_enhancer_model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "swapper_model": self.swapper_model,
            "pixel_boost": self.pixel_boost,
            "weight": self.weight,
            "face_selector_mode": self.face_selector_mode,
            "face_mask_types": list(self.face_mask_types),
            "face_enhancer_model": self.face_enhancer_model,
        }

    @property
    def license(self) -> str:
        return SWAPPER_LICENSES.get(self.swapper_model, "Unknown")


@dataclass(frozen=True, slots=True)
class RunRecord:
    argv: list[str]
    version: str
    seconds: float


def default_execution_providers() -> tuple[str, ...]:
    return ("coreml",) if sys.platform == "darwin" and platform.machine() == "arm64" else ("cpu",)


@dataclass(slots=True)
class FaceFusion:
    command: list[str]
    """How to invoke FaceFusion, e.g. [python, 'facefusion.py']."""
    cwd: Path | None = None
    execution_providers: tuple[str, ...] = field(default_factory=default_execution_providers)
    output_video_encoder: str = "libx264"
    output_video_quality: int = 100
    """100 maps to x264 crf 0 in FaceFusion, i.e. a near-lossless output."""
    _version: str | None = None

    @classmethod
    def from_home(cls, home: str | os.PathLike[str] | None = None) -> FaceFusion:
        root = Path(home or os.environ.get(FACEFUSION_HOME_ENV, ""))
        python = root / ".venv" / "bin" / "python"
        if not (root / "facefusion.py").is_file() or not python.is_file():
            raise FaceFusionError(
                f"FaceFusion not found in {root!s} (set ${FACEFUSION_HOME_ENV}; see "
                "tools/install_facefusion.sh)"
            )
        return cls(command=[str(python), "facefusion.py"], cwd=root)

    @classmethod
    def fake(cls) -> FaceFusion:
        """The test double: same CLI contract, draws an ellipse instead of swapping."""
        return cls(command=[sys.executable, "-m", "deepfake_bench.testing.fake_facefusion"])

    def version(self) -> str:
        if self._version is None:
            completed = subprocess.run(
                [*self.command, "--version"], capture_output=True, text=True, cwd=self.cwd
            )
            text = (completed.stdout or completed.stderr).strip().splitlines()
            self._version = text[-1] if completed.returncode == 0 and text else "unknown"
        return self._version

    def argv(
        self, source_image: Path, target_video: Path, output: Path, settings: SwapSettings
    ) -> list[str]:
        processors = ["face_swapper"] + (["face_enhancer"] if settings.face_enhancer_model else [])
        argv = [
            *self.command,
            "headless-run",
            "--processors", *processors,
            "--face-swapper-model", settings.swapper_model,
        ]  # fmt: skip
        if settings.pixel_boost:
            argv += ["--face-swapper-pixel-boost", settings.pixel_boost]
        if settings.weight is not None:
            argv += ["--face-swapper-weight", f"{settings.weight:g}"]
        if settings.face_enhancer_model:
            argv += ["--face-enhancer-model", settings.face_enhancer_model]
        argv += [
            "--face-selector-mode", settings.face_selector_mode,
            "--face-mask-types", *settings.face_mask_types,
            "--execution-providers", *self.execution_providers,
            "--output-video-encoder", self.output_video_encoder,
            "--output-video-quality", str(self.output_video_quality),
            "--log-level", "error",
            "-s", str(source_image.resolve()),
            "-t", str(target_video.resolve()),
            "-o", str(output.resolve()),
        ]  # fmt: skip
        return argv

    def swap(
        self,
        source_image: Path,
        target_video: Path,
        output: Path,
        settings: SwapSettings,
        timeout_s: float = 3600,
    ) -> RunRecord:
        argv = self.argv(source_image, target_video, output, settings)
        output.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        completed = subprocess.run(
            argv, capture_output=True, text=True, cwd=self.cwd, timeout=timeout_s
        )
        if completed.returncode != 0 or not output.is_file():
            tail = "\n".join((completed.stderr or completed.stdout).strip().splitlines()[-8:])
            raise FaceFusionError(f"FaceFusion failed ({completed.returncode}): {tail}")
        return RunRecord(
            argv=_redact(argv), version=self.version(), seconds=time.perf_counter() - start
        )


def _redact(argv: Sequence[str]) -> list[str]:
    """Absolute paths in the recorded argv are reduced to file names (portable provenance)."""
    return [Path(a).name if Path(a).is_absolute() else a for a in argv]
