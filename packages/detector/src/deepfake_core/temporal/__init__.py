"""Temporal model (roadmap step 14): scores a face track from its sequence of GenD
embeddings and frame scores. See ADR 0007."""

from deepfake_core.temporal.features import make_windows, sequence_features, window_starts
from deepfake_core.temporal.model import TemporalHParams, TemporalNet
from deepfake_core.temporal.scorer import (
    CheckpointMeta,
    TemporalScore,
    TemporalScorer,
    save_checkpoint,
    score_sequence,
)

__all__ = [
    "CheckpointMeta",
    "TemporalHParams",
    "TemporalNet",
    "TemporalScore",
    "TemporalScorer",
    "make_windows",
    "save_checkpoint",
    "score_sequence",
    "sequence_features",
    "window_starts",
]
