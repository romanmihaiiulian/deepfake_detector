"""Identity-level splits, assigned before any fake is generated, plus a leakage check.

Splits come first. Donor and target identities are only ever paired within one split, so
no identity or source video can appear in two splits. A face-swap dataset where every
identity donates to every other would otherwise collapse into a single connected
component, and no honest split would exist.

The leakage check is a union-find over identities and source videos: every connected
component must lie in exactly one split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from deepfake_bench.manifest import Sample, Split

PRIORITY: tuple[Split, ...] = ("test", "val", "train")
MIN_PER_SPLIT = 2  # a face swap needs a donor and a target from the same split


def _rank(seed: int, identity_id: str) -> str:
    return hashlib.sha256(f"{seed}:{identity_id}".encode()).hexdigest()


def assign_splits(
    identity_ids: Iterable[str],
    ratios: Mapping[Split, float],
    seed: int,
    existing: Mapping[str, Split] | None = None,
) -> dict[str, Split]:
    """Deterministic identity → split assignment.

    Identities that already have a split keep it. A new split opens, in priority order
    test, val, train, only when at least two unassigned identities can fill it. A split
    with one identity could never pair a donor with a target. Each open split first gets
    its two identities; the remainder follows the target ratios.
    """
    existing = dict(existing or {})
    pending = sorted((i for i in identity_ids if i not in existing), key=lambda i: _rank(seed, i))
    wanted = [s for s in PRIORITY if ratios.get(s, 0) > 0] or ["test"]
    counts: dict[Split, int] = dict.fromkeys(PRIORITY, 0)
    for split in existing.values():
        counts[split] = counts.get(split, 0) + 1

    opened = [s for s in wanted if counts.get(s, 0) > 0]
    spare = len(pending) - sum(max(0, MIN_PER_SPLIT - counts[s]) for s in opened)
    for split in wanted:
        if split not in opened and spare >= MIN_PER_SPLIT:
            opened.append(split)
            spare -= MIN_PER_SPLIT
    if not opened:
        opened = [wanted[0]]
    opened.sort(key=PRIORITY.index)

    assignment = dict(existing)
    queue = list(pending)
    for split in opened:  # phase 1: the minimum that makes pairing possible
        while counts[split] < MIN_PER_SPLIT and queue:
            assignment[queue.pop(0)] = split
            counts[split] += 1
    total = len(assignment) + len(queue)
    weight = sum(ratios.get(s, 0) for s in opened) or 1.0
    for identity in queue:  # phase 2: towards the target ratios
        split = max(
            opened,
            key=lambda s: (total * ratios.get(s, 0) / weight - counts[s], -PRIORITY.index(s)),
        )
        assignment[identity] = split
        counts[split] += 1
    return assignment


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def leakage_violations(samples: Iterable[Sample]) -> list[str]:
    """Human-readable violations; an empty list means the splits are clean."""
    union = _UnionFind()
    sample_list = list(samples)
    for sample in sample_list:
        video = f"video:{sample.source_video}"
        union.find(video)
        for identity in (sample.source_identity, sample.target_identity):
            if identity:
                union.union(video, f"identity:{identity}")
    splits_of: dict[str, set[str]] = {}
    members: dict[str, set[str]] = {}
    for sample in sample_list:
        root = union.find(f"video:{sample.source_video}")
        splits_of.setdefault(root, set()).add(sample.split)
        members.setdefault(root, set()).add(sample.source_video)
    return [
        f"component with videos {sorted(members[root])} spans splits {sorted(splits)}"
        for root, splits in splits_of.items()
        if len(splits) > 1
    ]
