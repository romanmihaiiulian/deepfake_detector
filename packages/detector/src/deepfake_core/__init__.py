"""deepfake-core: the detection engine behind the web app, the API and future integrations.

The engine is independent of any web framework. It takes a video file and produces a
probabilistic assessment; it never claims proof of manipulation.

    from deepfake_core import analyze_video
    result = analyze_video("interview.mp4")
    print(result.verdict, result.manipulation_probability, result.video_quality)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from deepfake_core.config import PipelineConfig
    from deepfake_core.pipeline import AnalysisOutput, Analyzer, analyze_video
    from deepfake_core.schemas import AnalysisResult

__all__ = [
    "AnalysisOutput",
    "AnalysisResult",
    "Analyzer",
    "PipelineConfig",
    "__version__",
    "analyze_video",
]

_LAZY = {
    "AnalysisOutput": "deepfake_core.pipeline",
    "AnalysisResult": "deepfake_core.schemas",
    "Analyzer": "deepfake_core.pipeline",
    "PipelineConfig": "deepfake_core.config",
    "analyze_video": "deepfake_core.pipeline",
}


def __getattr__(name: str) -> Any:
    """Heavy modules (OpenCV, PyAV, torch) load on first use, not on `import deepfake_core`."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'deepfake_core' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
