"""GenD (Yermakov et al., WACV 2026) with our own loader. See ADR 0003.

The architecture is CLIP ViT-L/14 vision tower → pooled output (1024-d) → L2
normalisation → Linear(1024, 2). That is GenD's `LinearNorm` head; the checkpoint's
config.json says so. Softmax column 1 is "fake" (upstream label2name {0: real, 1: fake}).

The weights are read from a local safetensors file with a strict, explicit key remap: no
`trust_remote_code`, no network, and every tensor must match.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from deepfake_core.detectors.base import FramePredictions
from deepfake_core.device import select_device
from deepfake_core.errors import ModelIntegrityError
from deepfake_core.faces.align import model_input
from deepfake_core.models import GEND, require

INPUT_SIZE = 224
FAKE_INDEX = 1
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_CHECKPOINT_BACKBONE = "feature_extractor.vision_model."
_CHECKPOINT_HEAD = "model.linear."
_CHECKPOINT_UNUSED = ("feature_extractor.visual_projection.weight",)


def clip_vit_l14_config() -> CLIPVisionConfig:
    """Vision config of openai/clip-vit-large-patch14, the GenD backbone."""
    return CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=INPUT_SIZE,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        projection_dim=768,
    )


class GenDNetwork(nn.Module):
    def __init__(self, config: CLIPVisionConfig | None = None) -> None:
        super().__init__()
        self.vision = CLIPVisionModel(config or clip_vit_l14_config())
        self.head = nn.Linear(self.vision.config.hidden_size, 2)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.vision(pixel_values=pixel_values).pooler_output
        embedding = F.normalize(pooled, p=2, dim=1)
        return self.head(embedding), embedding


def remap_checkpoint(
    checkpoint: dict[str, torch.Tensor], network: GenDNetwork
) -> dict[str, torch.Tensor]:
    """Rename GenD checkpoint tensors to `network` parameter names, strictly."""
    vision_keys = set(network.vision.state_dict())
    # transformers 4.x nests the tower under `vision_model.`; 5.x does not.
    prefix = "vision_model." if any(k.startswith("vision_model.") for k in vision_keys) else ""
    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in checkpoint.items():
        if key.startswith(_CHECKPOINT_BACKBONE):
            mapped["vision." + prefix + key.removeprefix(_CHECKPOINT_BACKBONE)] = tensor
        elif key.startswith(_CHECKPOINT_HEAD):
            mapped["head." + key.removeprefix(_CHECKPOINT_HEAD)] = tensor
        elif key not in _CHECKPOINT_UNUSED:
            raise ModelIntegrityError(f"Unexpected tensor in GenD checkpoint: {key}")
    expected = set(network.state_dict())
    if set(mapped) != expected:
        missing, extra = sorted(expected - set(mapped)), sorted(set(mapped) - expected)
        raise ModelIntegrityError(
            f"GenD checkpoint does not match the network: missing {missing[:5]}, "
            f"unexpected {extra[:5]}"
        )
    return mapped


def preprocess(crops: Sequence[npt.NDArray[np.uint8]]) -> torch.Tensor:
    """CLIPImageProcessor-equivalent: bicubic resize to 224, rescale, CLIP mean/std."""
    batch = np.stack([model_input(crop, INPUT_SIZE) for crop in crops]).astype(np.float32)
    batch = (batch / 255.0 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


class GenDDetector:
    name = "gend"

    def __init__(
        self,
        models_dir: str | os.PathLike[str] | None = None,
        *,
        device: str | torch.device | None = None,
        batch_size: int = 16,
        checkpoint: str | os.PathLike[str] | None = None,
        network_config: CLIPVisionConfig | None = None,
    ) -> None:
        self.device = device if isinstance(device, torch.device) else select_device(device)
        self.batch_size = batch_size
        path = checkpoint if checkpoint is not None else require(GEND, models_dir)
        network = GenDNetwork(network_config)
        network.load_state_dict(remap_checkpoint(load_file(os.fspath(path)), network), strict=True)
        self.network = network.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, crops: Sequence[npt.NDArray[np.uint8]]) -> FramePredictions:
        if not crops:
            dim = self.network.head.in_features
            return FramePredictions(np.zeros(0, np.float32), np.zeros((0, dim), np.float32))
        probabilities, embeddings = [], []
        for start in range(0, len(crops), self.batch_size):
            pixels = preprocess(crops[start : start + self.batch_size]).to(self.device)
            logits, embedding = self.network(pixels)
            probabilities.append(logits.float().softmax(dim=-1)[:, FAKE_INDEX].cpu().numpy())
            embeddings.append(embedding.float().cpu().numpy())
        return FramePredictions(
            np.concatenate(probabilities).astype(np.float32),
            np.concatenate(embeddings).astype(np.float32),
        )

    def warmup(self) -> None:
        self.predict([np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)])
