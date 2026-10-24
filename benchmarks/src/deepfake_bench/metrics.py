"""Video-level evaluation metrics, with cluster-bootstrap confidence intervals.

Labels: 1 = manipulated, 0 = real. Scores: the pipeline's manipulation probability
(uncalibrated). AUC and TPR@FPR are implemented with ranks in numpy, so a 1000-sample
bootstrap over many slices stays cheap. scikit-learn provides the reference values the
tests compare against.

Confidence intervals resample whole source videos (pair IDs), not individual samples: a
real video and the fakes derived from it are not independent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

Array = npt.NDArray[np.float64]


def roc_auc(labels: Array, scores: Array) -> float | None:
    """Mann-Whitney AUC with average ranks for ties; None when one class is missing."""
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: Array, scores: Array) -> float | None:
    """PR-AUC as average precision (same definition as sklearn)."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    # evaluate only at the last index of each tied score block
    distinct = np.r_[np.where(np.diff(sorted_scores))[0], sorted_labels.size - 1]
    tp = np.cumsum(sorted_labels)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))


def roc_points(labels: Array, scores: Array) -> tuple[Array, Array, Array]:
    """(fpr, tpr, thresholds) at every distinct score, predicting 'manipulated' when
    score >= threshold, highest threshold first (plus the all-negative point)."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    distinct = np.r_[np.where(np.diff(sorted_scores))[0], sorted_labels.size - 1]
    tp = np.cumsum(sorted_labels)[distinct]
    fp = (distinct + 1) - tp
    n_pos, n_neg = max(1, int((labels == 1).sum())), max(1, int((labels == 0).sum()))
    fpr = np.r_[0.0, fp / n_neg]
    tpr = np.r_[0.0, tp / n_pos]
    thresholds = np.r_[np.inf, sorted_scores[distinct]]
    return fpr, tpr, thresholds


def tpr_at_fpr(labels: Array, scores: Array, max_fpr: float) -> tuple[float, float] | None:
    """Highest TPR whose FPR <= max_fpr, and the threshold that achieves it."""
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return None
    fpr, tpr, thresholds = roc_points(labels, scores)
    allowed = fpr <= max_fpr + 1e-12
    best = int(np.argmax(np.where(allowed, tpr, -1.0)))
    return float(tpr[best]), float(thresholds[best])


@dataclass(frozen=True, slots=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None

    @property
    def fpr(self) -> float | None:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else None

    @property
    def fnr(self) -> float | None:
        return self.fn / (self.fn + self.tp) if self.fn + self.tp else None


def confusion(labels: Array, scores: Array, threshold: float) -> Confusion:
    predicted = scores >= threshold
    positive = labels == 1
    return Confusion(
        tp=int((predicted & positive).sum()),
        fp=int((predicted & ~positive).sum()),
        tn=int((~predicted & ~positive).sum()),
        fn=int((~predicted & positive).sum()),
    )


def expected_calibration_error(labels: Array, scores: Array, bins: int = 15) -> float | None:
    """ECE of score-as-P(manipulated): weighted |accuracy - confidence| over equal-width bins."""
    if labels.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            error += members.mean() * abs(labels[members].mean() - scores[members].mean())
    return float(error)


def brier(labels: Array, scores: Array) -> float | None:
    return float(np.mean((scores - labels) ** 2)) if labels.size else None


def paired_win_rate(pairs: Sequence[tuple[float, Sequence[float]]]) -> float | None:
    """For each (real score, fake scores derived from it): how often fake > real (ties ½)."""
    wins = total = 0.0
    for real, fakes in pairs:
        for fake in fakes:
            wins += 1.0 if fake > real else 0.5 if fake == real else 0.0
            total += 1
    return wins / total if total else None


def bootstrap_ci(
    groups: Sequence[str],
    statistic: Callable[[npt.NDArray[np.int64]], float | None],
    *,
    iterations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile CI of `statistic(indices)` when resampling whole groups with replacement."""
    unique = sorted(set(groups))
    if len(unique) < 2:
        return None
    members = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        chosen = rng.choice(len(unique), size=len(unique), replace=True)
        indices = np.concatenate([members[unique[c]] for c in chosen])
        value = statistic(indices)
        if value is not None:
            values.append(value)
    if len(values) < iterations * 0.5:
        return None
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)
