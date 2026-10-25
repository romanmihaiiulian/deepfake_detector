"""Per-track input of the temporal model: one feature row per scored frame, then windows.

A row is the frame's L2-normalised GenD embedding followed by four scalars: its GenD
logit, the time since the previous scored frame, the cosine distance between its
embedding and the previous one, and the absolute change in GenD logit. Unscored frames
are dropped, so the time gap is what tells the model about holes in the track.

The two step features are explicit on purpose. Face swaps tend to flicker from frame to
frame, and with lab-sized training sets a network given only absolute embeddings learns
to memorise videos rather than to compare neighbouring frames.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

EXTRA_FEATURES = 4
"""Appended after the embedding: GenD logit, gap to the previous scored frame (s),
embedding cosine distance to it, absolute GenD-logit change since it."""
MAX_GAP_S = 2.0
_EPS = 1e-4

Floats = npt.NDArray[np.float32]


def sequence_features(
    times: npt.ArrayLike, p_fake: npt.ArrayLike, embeddings: npt.ArrayLike
) -> tuple[Floats, Floats]:
    """(times, features) of the scored frames, in time order. `p_fake` is NaN where a
    frame was not scored."""
    t = np.asarray(times, np.float32)
    p = np.asarray(p_fake, np.float32)
    e = np.asarray(embeddings, np.float32)
    order = np.argsort(t, kind="stable")
    t, p, e = t[order], p[order], e[order]
    scored = np.isfinite(p)
    t, p, e = t[scored], np.clip(p[scored], _EPS, 1 - _EPS), e[scored]
    logit = np.log(p) - np.log1p(-p)
    gap = np.clip(np.diff(t, prepend=t[:1]), 0.0, MAX_GAP_S)
    unit = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-6)
    previous = np.concatenate([unit[:1], unit[:-1]])
    step = 1.0 - (unit * previous).sum(axis=1)
    logit_step = np.abs(np.diff(logit, prepend=logit[:1]))
    extra = np.stack([logit, gap, step, logit_step], axis=1)
    return t, np.concatenate([e, extra], axis=1).astype(np.float32)


def window_starts(n: int, length: int, stride: int) -> list[int]:
    """Start indices of windows of `length` rows every `stride` rows. The last window is
    aligned to the end of the sequence, so every row is covered; a sequence shorter than
    one window gives a single (padded) window."""
    if n <= length:
        return [0]
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    return starts


def make_windows(
    features: Floats, length: int, stride: int
) -> tuple[Floats, npt.NDArray[np.bool_], list[int]]:
    """(windows (W, length, F), valid mask (W, length), start indices). Short sequences
    are right-padded with zeros and masked out."""
    n, dim = features.shape
    starts = window_starts(n, length, stride)
    windows = np.zeros((len(starts), length, dim), np.float32)
    mask = np.zeros((len(starts), length), np.bool_)
    for i, start in enumerate(starts):
        chunk = features[start : start + length]
        windows[i, : len(chunk)] = chunk
        mask[i, : len(chunk)] = True
    return windows, mask, starts
