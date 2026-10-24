"""Deterministic frame sampling on a fixed timestamp grid.

Targets are `t_k = k / rate` seconds after the first frame. Each target takes the decoded
frame whose timestamp is nearest to it, and ties go to the earlier frame. Because the
grid is defined in time rather than in frame indices, variable-frame-rate video samples
correctly.

Timestamps are exact rationals (`pts * time_base`), so tie-breaking is exact and the same
file always yields the same frames on every platform. Ties are common: at 25 fps on a
4 Hz grid every second target falls exactly between two frames.
"""

from __future__ import annotations

from fractions import Fraction

from deepfake_core.config import SamplingConfig

_RATE_DENOMINATOR = 1_000_000


def plan_sample_rate(duration: float | None, fps: float | None, config: SamplingConfig) -> Fraction:
    """Samples per second: the configured rate, lowered so the video yields at most
    `max_samples`, and never above the video's own frame rate."""
    rate = config.rate_hz
    if duration and duration > 0:
        rate = min(rate, config.max_samples / duration)
    if fps and fps > 0:
        rate = min(rate, fps)
    return Fraction(rate).limit_denominator(_RATE_DENOMINATOR)


class SamplingGrid:
    """Online nearest-frame selection over frame timestamps in presentation order.

    `push(index, time)` returns the frame indices newly selected by that frame. The list
    is empty, the previous frame, the current one, or both, in that order. A frame that
    is nearest to several targets is returned only once. `finish()` assigns trailing
    targets (up to `end_tolerance` after the last frame) to the last frame.

    Coverage: a target is "far" when its nearest frame is more than half a sampling
    interval away, which only happens when frames are missing.
    """

    def __init__(self, rate: Fraction, end_tolerance: Fraction = Fraction(0)) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.end_tolerance = end_tolerance
        self.targets = 0
        self.far_targets = 0
        self._far = 1 / (2 * rate)
        self._k = 0
        self._prev: tuple[int, Fraction] | None = None
        self._last_selected: int | None = None

    def _target(self) -> Fraction:
        return self._k / self.rate

    def _assign(self, index: int, time: Fraction, target: Fraction, selected: list[int]) -> None:
        self.targets += 1
        if abs(time - target) > self._far:
            self.far_targets += 1
        if index != self._last_selected:
            selected.append(index)
            self._last_selected = index
        self._k += 1

    def push(self, index: int, time: Fraction) -> list[int]:
        selected: list[int] = []
        while self._target() <= time:
            target = self._target()
            prev = self._prev
            if prev is not None and target - prev[1] <= time - target:
                self._assign(prev[0], prev[1], target, selected)
            else:
                self._assign(index, time, target, selected)
        self._prev = (index, time)
        return selected

    def finish(self) -> list[int]:
        selected: list[int] = []
        prev = self._prev
        if prev is not None:
            while self._target() - prev[1] <= self.end_tolerance:
                self._assign(prev[0], prev[1], self._target(), selected)
        return selected
