"""Checkpoints and track scoring for the temporal model.

A checkpoint is one safetensors file: the network's tensors plus string metadata (the
hyper-parameters, window and stride, and the sha256 of the frame model whose embeddings
it was trained on). The weights are trained locally on lab data (`dfbench temporal
train`); nothing is downloaded. A checkpoint only makes sense with the exact frame model
it was trained on, so loading it next to a different one is an error.

A track's temporal score is the mean window probability over overlapping windows that
cover every scored frame.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from deepfake_core.errors import ModelIntegrityError, ModelMissingError
from deepfake_core.schemas import ModelInfo
from deepfake_core.temporal.features import EXTRA_FEATURES, make_windows, sequence_features
from deepfake_core.temporal.model import TemporalHParams, TemporalNet

FORMAT = "deepfake-temporal"
FORMAT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    hparams: TemporalHParams
    window: int
    stride: int
    embedding_model_sha256: str
    name: str
    trained_at: str
    training_set_sha256: str

    def to_strings(self) -> dict[str, str]:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "hparams": json.dumps(self.hparams.to_dict(), sort_keys=True),
            "window": str(self.window),
            "stride": str(self.stride),
            "embedding_model_sha256": self.embedding_model_sha256,
            "name": self.name,
            "trained_at": self.trained_at,
            "training_set_sha256": self.training_set_sha256,
        }

    @classmethod
    def from_strings(cls, data: dict[str, str]) -> CheckpointMeta:
        if data.get("format") != FORMAT or data.get("format_version") != FORMAT_VERSION:
            raise ModelIntegrityError("not a temporal-model checkpoint of a supported version")
        try:
            return cls(
                hparams=TemporalHParams.from_dict(json.loads(data["hparams"])),
                window=int(data["window"]),
                stride=int(data["stride"]),
                embedding_model_sha256=data["embedding_model_sha256"],
                name=data["name"],
                trained_at=data["trained_at"],
                training_set_sha256=data["training_set_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelIntegrityError(f"invalid temporal checkpoint metadata: {error}") from error


def save_checkpoint(net: TemporalNet, meta: CheckpointMeta, path: str | os.PathLike[str]) -> str:
    """Write the checkpoint and return its sha256."""
    state = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
    save_file(state, os.fspath(path), metadata=meta.to_strings())
    return _sha256(Path(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TemporalScore:
    score: float
    windows: tuple[tuple[float, float, float], ...]
    """(start time, end time, probability) per window."""


class TemporalScorer:
    def __init__(self, path: str | os.PathLike[str], *, device: str = "cpu") -> None:
        path = Path(path)
        if not path.is_file():
            raise ModelMissingError(f"temporal checkpoint not found: {path}")
        with safe_open(os.fspath(path), framework="pt") as handle:
            self.meta = CheckpointMeta.from_strings(handle.metadata() or {})
        self.sha256 = _sha256(path)
        self.device = torch.device(device)
        net = TemporalNet(self.meta.hparams)
        try:
            net.load_state_dict(load_file(os.fspath(path)), strict=True)
        except RuntimeError as error:
            raise ModelIntegrityError(f"temporal checkpoint does not match: {error}") from error
        self.net = net.to(self.device).eval()

    @property
    def embedding_model_sha256(self) -> str:
        return self.meta.embedding_model_sha256

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Temporal {self.meta.hparams.arch} ({self.meta.name})",
            role="temporal",
            version=self.meta.trained_at,
            revision=self.meta.training_set_sha256[:16],
            sha256=self.sha256,
            license="internal: trained on consented lab data",
        )

    def score(
        self,
        times: npt.ArrayLike,
        p_fake: npt.ArrayLike,
        embeddings: npt.ArrayLike,
    ) -> TemporalScore | None:
        """Score one track; None when it has no scored frame."""
        return score_sequence(
            self.net, self.meta.window, self.meta.stride, times, p_fake, embeddings
        )


@torch.inference_mode()
def score_sequence(
    net: TemporalNet,
    window: int,
    stride: int,
    times: npt.ArrayLike,
    p_fake: npt.ArrayLike,
    embeddings: npt.ArrayLike,
) -> TemporalScore | None:
    """Mean window probability of one track (the network must be in eval mode)."""
    t, features = sequence_features(times, p_fake, embeddings)
    if t.size == 0:
        return None
    if features.shape[1] != net.hparams.feature_dim:
        raise ModelIntegrityError(
            f"embeddings have {features.shape[1] - EXTRA_FEATURES} dimensions, the temporal "
            f"model expects {net.hparams.embedding_dim}"
        )
    device = next(net.parameters()).device
    windows, mask, starts = make_windows(features, window, stride)
    logits = net(torch.from_numpy(windows).to(device), torch.from_numpy(mask).to(device))
    probs = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
    spans = tuple(
        (float(t[s]), float(t[min(s + window, t.size) - 1]), float(p))
        for s, p in zip(starts, probs, strict=True)
    )
    return TemporalScore(float(probs.mean()), spans)
