"""The temporal network: a window of per-frame features → one manipulation logit.

Two encoders share the same input projection and output head:

- `tcn`: residual dilated 1-D convolutions (dilations 1, 2, 4, ...), non-causal.
- `transformer`: a small Transformer encoder with learned positions.

The output is `base_weight * mean(GenD logit) + base_bias + head(pooled)`. The head
starts at zero and the base at (1, 0), so an untrained network reproduces the GenD
mean-logit baseline and training only learns a correction. With little lab data, that
keeps the model from starting worse than the detector it extends.

Padding is exact: masked rows never influence valid ones, so a short track gives the
same output padded or not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from deepfake_core.temporal.features import EXTRA_FEATURES

Architecture = Literal["tcn", "transformer"]


@dataclass(frozen=True, slots=True)
class TemporalHParams:
    embedding_dim: int
    arch: Architecture = "tcn"
    hidden: int = 128
    layers: int = 3
    heads: int = 4
    kernel: int = 3
    dropout: float = 0.1
    max_len: int = 64

    @property
    def feature_dim(self) -> int:
        return self.embedding_dim + EXTRA_FEATURES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalHParams:
        return cls(**data)


class _ConvBlock(nn.Module):
    def __init__(self, hidden: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(hidden, hidden, kernel, dilation=dilation, padding=padding)
        self.mix = nn.Conv1d(hidden, hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, L, H). Padded rows are zeroed before each convolution, which makes them
        # equivalent to the convolution's own zero padding at the sequence end.
        x = (h * mask.unsqueeze(-1)).transpose(1, 2)
        x = self.mix(F.gelu(self.conv(x))).transpose(1, 2)
        out: torch.Tensor = self.norm(h + self.dropout(x))
        return out


class TemporalNet(nn.Module):
    def __init__(self, hparams: TemporalHParams) -> None:
        super().__init__()
        if hparams.kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.hparams = hparams
        h = hparams.hidden
        self.proj = nn.Linear(hparams.feature_dim, h)
        self.norm = nn.LayerNorm(h)
        if hparams.arch == "tcn":
            self.blocks = nn.ModuleList(
                _ConvBlock(h, hparams.kernel, 2**i, hparams.dropout) for i in range(hparams.layers)
            )
        elif hparams.arch == "transformer":
            self.position = nn.Parameter(torch.zeros(hparams.max_len, h))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                h, hparams.heads, 2 * h, hparams.dropout, activation="gelu", batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, hparams.layers, enable_nested_tensor=False)
        else:
            raise ValueError(f"unknown architecture {hparams.arch!r}")
        self.head = nn.Linear(h, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.base_weight = nn.Parameter(torch.ones(()))
        self.base_bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) features, mask: (B, L) True for valid rows → (B,) logits."""
        if x.shape[1] > self.hparams.max_len:
            raise ValueError(f"window of {x.shape[1]} exceeds max_len {self.hparams.max_len}")
        weights = mask.to(x.dtype)
        count = weights.sum(dim=1).clamp(min=1.0)
        h = self.norm(self.proj(x))
        if self.hparams.arch == "tcn":
            for block in self.blocks:
                h = block(h, weights)
        else:
            h = self.encoder(h + self.position[: x.shape[1]], src_key_padding_mask=~mask)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
        gend_logit = x[..., self.hparams.embedding_dim]
        base = (gend_logit * weights).sum(dim=1) / count
        logit: torch.Tensor = self.base_weight * base + self.base_bias
        correction: torch.Tensor = self.head(pooled).squeeze(-1)
        return logit + correction
