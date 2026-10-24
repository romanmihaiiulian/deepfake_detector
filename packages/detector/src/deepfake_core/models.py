"""Model manifest: where each pretrained weight comes from, pinned by revision and checksum.

The engine never downloads anything while analyzing. `fetch()` is only used by
`tools/fetch_models.py` and `deepfake-core models fetch`; analysis reads local files from
the models directory and raises `ModelMissingError` when one is absent.

Licensing and training-data provenance of every model is recorded in
`dependencies/licenses.yml` (see docs/adr/0002 and 0003).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from deepfake_core.errors import ModelIntegrityError, ModelMissingError

MODELS_DIR_ENV = "DFD_MODELS_DIR"
DEFAULT_MODELS_DIR = "models"

_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    name: str
    role: str
    version: str
    revision: str
    path: str
    """Location relative to the models directory."""
    url: str
    sha256: str
    size: int
    license: str
    source: str


YUNET = ModelSpec(
    key="yunet",
    name="YuNet face detector",
    role="face_detector",
    version="2023mar",
    revision="eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4",
    path="yunet/face_detection_yunet_2023mar.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232_589,
    license="MIT",
    source="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
)

GEND = ModelSpec(
    key="gend",
    name="GenD CLIP ViT-L/14",
    role="frame_detector",
    version="GenD_CLIP_L_14",
    revision="891ce014a0308386c4d7d25b3dcf436a22db5504",
    path="gend/GenD_CLIP_L_14/model.safetensors",
    url=(
        "https://huggingface.co/yermandy/GenD_CLIP_L_14/resolve/"
        "891ce014a0308386c4d7d25b3dcf436a22db5504/model.safetensors"
    ),
    sha256="d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833",
    size=1_215_928_160,
    license="MIT",
    source="https://huggingface.co/yermandy/GenD_CLIP_L_14",
)

MODELS: tuple[ModelSpec, ...] = (YUNET, GEND)


def models_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The models directory: the explicit argument, else $DFD_MODELS_DIR, else ./models."""
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get(MODELS_DIR_ENV, DEFAULT_MODELS_DIR))


def model_path(spec: ModelSpec, root: str | os.PathLike[str] | None = None) -> Path:
    return models_dir(root) / spec.path


def require(spec: ModelSpec, root: str | os.PathLike[str] | None = None) -> Path:
    """Return the local path of a model, checking that it exists and has the pinned size.

    This is the cheap check done at load time; `verify()` also hashes the file.
    """
    path = model_path(spec, root)
    if not path.is_file():
        raise ModelMissingError(
            f"{spec.name} not found at {path}. Run `make models` "
            f"(or set ${MODELS_DIR_ENV} to the directory holding the weights)."
        )
    size = path.stat().st_size
    if size != spec.size:
        raise ModelIntegrityError(
            f"{spec.name} at {path} has {size} bytes, expected {spec.size}. Re-run `make models`."
        )
    return path


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(spec: ModelSpec, root: str | os.PathLike[str] | None = None) -> Path:
    """Check presence, size and sha256 of a model file. Raises on any mismatch."""
    path = require(spec, root)
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ModelIntegrityError(
            f"{spec.name} at {path} has sha256 {actual}, expected {spec.sha256}."
        )
    return path


def fetch(
    spec: ModelSpec,
    root: str | os.PathLike[str] | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    force: bool = False,
) -> Path:
    """Download a model to the models directory, verifying its sha256 before it is used.

    The file is written to a temporary name in the destination directory and only renamed
    into place once the checksum matches, so an interrupted download never leaves a
    truncated model behind.
    """
    dest = model_path(spec, root)
    if not force and dest.is_file():
        try:
            return verify(spec, root)
        except ModelIntegrityError:
            pass  # stale or corrupt: download again

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(spec.url, headers={"User-Agent": "deepfake-detector/0.1"})
    digest = hashlib.sha256()
    received = 0
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request, timeout=60) as response:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if progress is not None:
                    progress(received, spec.size)
        actual = digest.hexdigest()
        if actual != spec.sha256:
            raise ModelIntegrityError(
                f"Downloaded {spec.name} has sha256 {actual}, expected {spec.sha256}."
            )
        tmp.chmod(0o644)  # mkstemp creates 0600; weights are not secret
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest
