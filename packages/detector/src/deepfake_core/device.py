"""Torch device selection: CUDA, then Apple MPS, then CPU."""

from __future__ import annotations

import os

import torch

DEVICE_ENV = "DFD_DEVICE"


def select_device(preferred: str | None = None) -> torch.device:
    """`preferred` (or $DFD_DEVICE) may be 'auto', 'cuda', 'cuda:N', 'mps' or 'cpu'."""
    choice = (preferred or os.environ.get(DEVICE_ENV) or "auto").lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(choice)
