"""Evaluation slices: the robustness views the handoff doc requires.

AUC-type metrics need both classes, so a slice that filters fakes by a property (swapper,
setting, dataset) keeps the reals of the same population. A per-swapper slice, for
example, contains that swapper's fakes plus every lab real video.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Record = Mapping[str, Any]
Predicate = Callable[[Record], bool]

# Transformation groups (M8 profiles); a slice per group plus one per profile.
GROUPS: dict[str, tuple[str, ...]] = {
    "recompressed": ("h264_crf23", "h264_crf28", "h264_crf35", "h265_crf28"),
    "low_resolution": ("res_720p", "res_480p"),
    "low_frame_rate": ("fps_15",),
    "reframed": ("crop_10", "letterbox"),
    "screen_recording": ("screen_recording",),
    "conferencing": ("conf_low", "conf_medium", "conf_high"),
}


@dataclass(frozen=True, slots=True)
class Slice:
    name: str
    description: str
    include: Predicate

    def select(self, records: Iterable[Record]) -> list[Record]:
        return [r for r in records if self.include(r)]


def _short_side(resolution: str | None) -> int | None:
    try:
        width, height = (int(v) for v in (resolution or "").split("x"))
    except ValueError:
        return None
    return min(width, height)


def resolution_bucket(resolution: str | None) -> str:
    side = _short_side(resolution)
    if side is None:
        return "unknown"
    if side >= 1080:
        return ">=1080p"
    if side >= 720:
        return "720p"
    if side >= 480:
        return "480p"
    return "<480p"


def is_lab(record: Record) -> bool:
    return bool(record.get("dataset", "lab") == "lab")


def field_is(key: str, value: object) -> Predicate:
    def include(record: Record) -> bool:
        return record.get(key) == value

    return include


def lab_transformation_in(names: Iterable[str]) -> Predicate:
    members = set(names)

    def include(record: Record) -> bool:
        return is_lab(record) and record.get("transformation") in members

    return include


def lab_reals_or(fake_filter: Predicate, split: str | None = None) -> Predicate:
    """Lab reals (optionally of one split) plus the fakes that pass `fake_filter`."""

    def include(record: Record) -> bool:
        if not is_lab(record) or (split is not None and record.get("split") != split):
            return False
        return record.get("label") == "real" or fake_filter(record)

    return include


def resolution_is(bucket: str) -> Predicate:
    def include(record: Record) -> bool:
        return resolution_bucket(record.get("resolution")) == bucket

    return include


def build_slices(records: Sequence[Record], held_out_swappers: Iterable[str] = ()) -> list[Slice]:
    held_out = set(held_out_swappers)
    lab = [r for r in records if is_lab(r)]
    slices = [
        Slice("all", "every scored sample", lambda r: True),
        Slice("lab", "our FaceFusion lab (pristine and variants)", is_lab),
    ]
    slices += [
        Slice(f"split:{split}", f"{split} split", field_is("split", split))
        for split in sorted({str(r["split"]) for r in records})
    ]
    if any(r["split"] == "test" for r in lab):
        slices.append(
            Slice(
                "identities:unseen",
                "test-split identities: never used to train or tune anything",
                field_is("split", "test"),
            )
        )
    present = {str(r["transformation"]) for r in lab}
    slices += [
        Slice(
            f"transformation:{name}",
            f"lab samples encoded as {name}",
            lab_transformation_in([name]),
        )
        for name in sorted(present)
    ]
    for group, members in GROUPS.items():
        found = [m for m in members if m in present]
        if found:
            slices.append(
                Slice(
                    f"group:{group}", f"{group}: {', '.join(found)}", lab_transformation_in(found)
                )
            )
    for model in sorted({str(r["swapper_model"]) for r in lab if r.get("swapper_model")}):
        slices.append(
            Slice(
                f"swapper:{model}",
                f"{model} fakes vs all lab reals",
                lab_reals_or(field_is("swapper_model", model)),
            )
        )
    if held_out and any(r["split"] == "test" for r in lab):
        listed = ", ".join(sorted(held_out))
        slices.append(
            Slice(
                "settings:unseen",
                f"test split, held-out swappers ({listed}) vs reals",
                lab_reals_or(lambda r: r.get("swapper_model") in held_out, split="test"),
            )
        )
        slices.append(
            Slice(
                "settings:seen",
                "test split, swappers also used in train/val vs reals",
                lab_reals_or(lambda r: r.get("swapper_model") not in held_out, split="test"),
            )
        )
    datasets = sorted({str(r["dataset"]) for r in records if r.get("dataset") not in (None, "lab")})
    slices += [
        Slice(f"tool:{name}", f"external dataset {name} (unseen tools)", field_is("dataset", name))
        for name in datasets
    ]
    slices += [
        Slice(f"quality:{tier}", f"predicted video quality {tier}", field_is("video_quality", tier))
        for tier in ("good", "fair", "poor", "insufficient")
        if any(r.get("video_quality") == tier for r in records)
    ]
    slices += [
        Slice(f"resolution:{bucket}", f"short side {bucket}", resolution_is(bucket))
        for bucket in sorted({resolution_bucket(r.get("resolution")) for r in records})
    ]
    return slices
