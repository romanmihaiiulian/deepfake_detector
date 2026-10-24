"""Representative frames: the highest-scoring face crops, spread out in time.

Only aligned face crops are kept, never full frames. They are the minimum needed to show
why a video was flagged.
"""

from __future__ import annotations

import heapq
import io
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PIL import Image

from deepfake_core.scoring.intervals import Interval

JPEG_QUALITY = 90


@dataclass(order=True, slots=True)
class Candidate:
    p_fake: float
    order: int
    """Tie-breaker: observation order, so selection is deterministic."""
    track_id: int = field(compare=False)
    time: float = field(compare=False)
    crop: npt.NDArray[np.uint8] = field(compare=False, repr=False)


class CandidatePool:
    """Bounded pool of the highest-scoring crops seen so far (crops of raw track IDs)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._heap: list[Candidate] = []
        self._order = 0

    def offer(self, p_fake: float, track_id: int, time: float, crop: npt.NDArray[np.uint8]) -> None:
        if self.capacity <= 0:
            return
        candidate = Candidate(p_fake, -self._order, track_id, time, crop)
        self._order += 1
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, candidate)
        elif candidate > self._heap[0]:
            heapq.heapreplace(self._heap, candidate)

    def candidates(self) -> list[Candidate]:
        return sorted(self._heap, reverse=True)


@dataclass(frozen=True, slots=True)
class EvidencePick:
    evidence_id: str
    track_id: int
    time: float
    p_fake: float
    jpeg: bytes


def encode_jpeg(crop: npt.NDArray[np.uint8]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(crop).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def select_evidence(
    candidates: Sequence[Candidate],
    intervals: Sequence[Interval],
    *,
    count: int,
    min_separation_s: float,
) -> list[EvidencePick]:
    """Prefer frames inside suspicious intervals, then the highest scores; keep picks of the
    same track at least `min_separation_s` apart."""

    def inside(candidate: Candidate) -> bool:
        return any(
            iv.track_id == candidate.track_id and iv.start <= candidate.time <= iv.end
            for iv in intervals
        )

    ranked = sorted(candidates, key=lambda c: (inside(c), c.p_fake, c.order), reverse=True)
    picks: list[EvidencePick] = []
    for candidate in ranked:
        if len(picks) >= count:
            break
        if any(
            p.track_id == candidate.track_id and abs(p.time - candidate.time) < min_separation_s
            for p in picks
        ):
            continue
        picks.append(
            EvidencePick(
                evidence_id=f"e{len(picks)}",
                track_id=candidate.track_id,
                time=candidate.time,
                p_fake=candidate.p_fake,
                jpeg=encode_jpeg(candidate.crop),
            )
        )
    return picks
