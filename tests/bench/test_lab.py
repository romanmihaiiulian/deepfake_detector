from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from deepfake_bench.cli import main as dfbench
from deepfake_bench.facefusion import FaceFusion, SwapSettings
from deepfake_bench.lab import (
    GenerationConfig,
    LabError,
    add_identity,
    add_source,
    assign_identity_splits,
    delete_identity,
    generate,
    import_external,
    plan,
)
from deepfake_bench.manifest import Registry, Sample
from deepfake_bench.paths import LabPaths
from deepfake_bench.splits import assign_splits, leakage_violations
from deepfake_core.faces import Detection
from deepfake_core.faces.align import TEMPLATE
from tests.conftest import barcode_frame, requires_ffmpeg

MakeVideo = Callable[..., Path]
BOX = (100, 60, 220, 180)  # where the stub "face" is, and where the fake swapper paints


class StubFaces:
    def detect(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        x1, y1, x2, _ = BOX
        landmarks = (TEMPLATE * (x2 - x1) + np.array([x1, y1], np.float32)).astype(np.float32)
        return [Detection(np.array(BOX, np.float32), landmarks, 0.95)]


CONFIG = GenerationConfig.model_validate(
    {
        "donors_per_target": 1,
        "swappers": [
            {"model": "inswapper_128", "pixel_boost": "256x256"},
            {"model": "ghost_2_256"},
        ],
        "held_out_for_test": ["ghost_2_256"],
        "splits": {"seed": 7, "ratios": {"train": 0.6, "val": 0.2, "test": 0.2}},
    }
)


def recipe(argv: list[str]) -> list[str]:
    """An encode's argv with its input and output files masked."""
    masked = list(argv)
    masked[masked.index("-i") + 1] = "<input>"
    masked[-1] = "<output>"
    return masked


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(LabPaths(tmp_path / "data"))


def populate(registry: Registry, make_video: MakeVideo, identities: int = 4) -> None:
    for i in range(identities):
        identity = f"p{i:04d}"
        add_identity(registry, identity, consent_ref=f"CONSENT-TEST-{i}")
        video = make_video(
            f"{identity}.mp4",
            frames=50,
            fps=25,
            image_for=lambda k, i=i: barcode_frame(k + 1000 * i),
        )
        add_source(registry, video, identity)
    assign_identity_splits(registry, CONFIG.splits)


# --------------------------------------------------------------------------- registration


def test_identities_need_a_consent_reference(registry: Registry) -> None:
    with pytest.raises(ValidationError, match="consent_ref is required"):
        add_identity(registry, "p0001", consent_ref="  ")
    with pytest.raises(ValidationError):
        add_identity(registry, "Jane Doe", consent_ref="C-1")  # names are not IDs


def test_sources_need_a_registered_identity(registry: Registry, make_video: MakeVideo) -> None:
    with pytest.raises(LabError, match="register it"):
        add_source(registry, make_video("x.mp4", frames=30), "ghost")


def test_sources_are_copied_and_deduplicated(registry: Registry, make_video: MakeVideo) -> None:
    add_identity(registry, "p0001", consent_ref="C-1")
    video = make_video("x.mp4", frames=30)
    first = add_source(registry, video, "p0001")
    again = add_source(registry, video, "p0001")
    assert first.video_id == again.video_id and len(registry.sources) == 1
    assert registry.paths.absolute(first.path).is_file()
    assert Registry(registry.paths).sources[first.video_id].consent_ref == "C-1"


# --------------------------------------------------------------------------- splits


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (2, {"test": 2}),
        (3, {"test": 3}),
        (4, {"test": 2, "val": 2}),
        (6, {"test": 2, "val": 2, "train": 2}),
    ],
)
def test_split_sizes(count: int, expected: dict[str, int]) -> None:
    ids = [f"p{i}" for i in range(count)]
    assignment = assign_splits(ids, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    sizes: dict[str, int] = {}
    for split in assignment.values():
        sizes[split] = sizes.get(split, 0) + 1
    assert sizes == expected


def test_split_assignment_is_deterministic_and_stable() -> None:
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    ids = [f"p{i}" for i in range(10)]
    first = assign_splits(ids, ratios, seed=3)
    assert first == assign_splits(list(reversed(ids)), ratios, seed=3)
    grown = assign_splits([*ids, "p10", "p11"], ratios, seed=3, existing=first)
    assert all(grown[i] == first[i] for i in ids)  # adding people never moves anyone


def _sample(sample_id: str, split: str, video: str, donor: str | None, target: str) -> Sample:
    return Sample.model_validate(
        {
            "sample_id": sample_id,
            "pair_id": video,
            "label": "manipulated" if donor else "real",
            "split": split,
            "path": f"{sample_id}.mp4",
            "sha256": "0" * 64,
            "generator": "facefusion" if donor else "none",
            "source_video": video,
            "source_identity": donor,
            "target_identity": target,
            "resolution": "320x240",
            "fps": 25.0,
            "duration": 2.0,
        }
    )


def test_leakage_check_finds_identities_shared_across_splits() -> None:
    clean = [_sample("a", "val", "v1", None, "p1"), _sample("b", "val", "v1", "p2", "p1")]
    assert leakage_violations(clean) == []
    leaky = [*clean, _sample("c", "test", "v3", "p2", "p3")]  # p2 donates into test
    [violation] = leakage_violations(leaky)
    assert "spans splits ['test', 'val']" in violation


# --------------------------------------------------------------------------- generation


def test_plan_pairs_within_splits_and_holds_out_swappers(
    registry: Registry, make_video: MakeVideo
) -> None:
    populate(registry, make_video)
    jobs = plan(registry, CONFIG)
    reals = [j for j in jobs if j.donor is None]
    fakes = [j for j in jobs if j.donor is not None]
    assert len(reals) == 4
    split_of = {i: v.split for i, v in registry.identities.items()}
    for job in fakes:
        assert split_of[job.donor] == job.split == split_of[job.source.identity_id]
        assert job.donor != job.source.identity_id
        assert job.swapper is not None
        if job.swapper.model == "ghost_2_256":
            assert job.split == "test"
    assert len(fakes) == 2 * 1 + 2 * 2  # val: one swapper; test: both
    assert [j.sample_id for j in jobs] == [j.sample_id for j in plan(registry, CONFIG)]


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_generation_end_to_end_with_the_fake_swapper(
    registry: Registry, make_video: MakeVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FACEFUSION_BOX", ",".join(map(str, BOX)))
    populate(registry, make_video)
    report = generate(registry, CONFIG, FaceFusion.fake(), StubFaces(), log=lambda _: None)
    assert len(report.created) == 10 and not report.rejected

    samples = registry.ok_samples()
    fakes = [s for s in samples if s.label == "manipulated"]
    reals = {s.pair_id: s for s in samples if s.label == "real"}
    assert len(fakes) == 6 and len(reals) == 4
    for fake in fakes:
        assert fake.generator == "facefusion"
        assert fake.generator_version == "FakeFaceFusion 0.0 (test double)"
        assert fake.swapper_license in {"Non-Commercial", "Apache-2.0"}
        assert fake.manipulation_coverage is not None and fake.manipulation_coverage >= 0.8
        assert "--face-swapper-model" in fake.generator_argv
        assert not any(a.startswith("/") for a in fake.generator_argv)  # portable provenance
        real = reals[fake.pair_id]  # every fake has its real counterpart
        assert real.target_identity == fake.target_identity != fake.source_identity
        assert recipe(real.transformation_argv) == recipe(fake.transformation_argv)  # parity
    for sample in samples:
        assert registry.paths.absolute(sample.path).is_file()
        assert sample.resolution == "320x240" and sample.transformation == "pristine"
    assert leakage_violations(samples) == []

    again = generate(
        Registry(registry.paths), CONFIG, FaceFusion.fake(), StubFaces(), log=lambda _: None
    )
    assert again.created == [] and again.skipped == 10


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_a_swap_that_changed_nothing_is_rejected(
    registry: Registry, make_video: MakeVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "noop")
    populate(registry, make_video)
    report = generate(registry, CONFIG, FaceFusion.fake(), StubFaces(), log=lambda _: None)
    assert len(report.rejected) == 6 and len(report.created) == 4
    rejected = [s for s in registry.samples.values() if s.status == "rejected"]
    assert all("face swapped in 0%" in (s.status_reason or "") for s in rejected)
    assert all(s.label == "manipulated" for s in rejected)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_withdrawing_consent_deletes_everything_derived(
    registry: Registry, make_video: MakeVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FACEFUSION_BOX", ",".join(map(str, BOX)))
    populate(registry, make_video)
    generate(registry, CONFIG, FaceFusion.fake(), StubFaces(), log=lambda _: None)
    victim = "p0000"
    involved = {
        s.sample_id
        for s in registry.samples.values()
        if victim in (s.source_identity, s.target_identity)
    }
    run = registry.paths.runs / "r1"
    (run / "traces").mkdir(parents=True)
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps({"sample_id": sid}) + "\n" for sid in registry.samples)
    )
    for sid in involved:
        (run / "traces" / f"{sid}.npz").write_bytes(b"x")

    report = delete_identity(registry, victim)
    assert report.samples == len(involved) and report.sources == 1
    reloaded = Registry(registry.paths)
    assert victim not in reloaded.identities
    assert not any(
        victim in (s.source_identity, s.target_identity) for s in reloaded.samples.values()
    )
    assert not (registry.paths.donors / f"{victim}.png").exists()
    remaining = {
        json.loads(line)["sample_id"]
        for line in (run / "predictions.jsonl").read_text().splitlines()
    }
    assert remaining.isdisjoint(involved) and remaining
    assert not any((run / "traces" / f"{sid}.npz").exists() for sid in involved)
    files = {p.name for p in registry.paths.root.rglob("*") if p.is_file()}
    assert not any(sid in name for sid in involved for name in files)


# --------------------------------------------------------------------------- FaceFusion contract


def test_facefusion_argv_matches_the_3_9_1_cli(tmp_path: Path) -> None:
    runner = FaceFusion(command=["python", "facefusion.py"], execution_providers=("coreml",))
    settings = SwapSettings("hyperswap_1a_256", pixel_boost="512x512", weight=0.6)
    argv = runner.argv(
        tmp_path / "donor.png", tmp_path / "target.mp4", tmp_path / "out.mp4", settings
    )
    assert argv[:6] == [
        "python",
        "facefusion.py",
        "headless-run",
        "--processors",
        "face_swapper",
        "--face-swapper-model",
    ]
    expected_pairs = {
        "--face-swapper-model": "hyperswap_1a_256",
        "--face-swapper-pixel-boost": "512x512",
        "--face-swapper-weight": "0.6",
        "--face-selector-mode": "one",
        "--face-mask-types": "box",
        "--execution-providers": "coreml",
        "--output-video-encoder": "libx264",
        "--output-video-quality": "100",
        "-s": str((tmp_path / "donor.png").resolve()),
        "-t": str((tmp_path / "target.mp4").resolve()),
        "-o": str((tmp_path / "out.mp4").resolve()),
    }
    for flag, value in expected_pairs.items():
        assert argv[argv.index(flag) + 1] == value, flag
    assert settings.license == "ResearchRAIL"


# --------------------------------------------------------------------------- external data & CLI


def test_import_external_marks_research_only(
    registry: Registry, make_video: MakeVideo, tmp_path: Path
) -> None:
    real, fake = make_video("real.mp4", frames=30), make_video("fake.webm", frames=30)
    csv = tmp_path / "celebdf.csv"
    csv.write_text(f"path,label,identity\n{real},real,id0\n{fake},manipulated,id0\n")
    added = import_external(registry, csv, "celeb-df-v2")
    assert len(added) == 2
    samples = [registry.samples[a] for a in added]
    assert all(s.split == "external" and s.research_only for s in samples)
    assert {s.label for s in samples} == {"real", "manipulated"}
    assert import_external(registry, csv, "celeb-df-v2") == []  # idempotent


def test_dfbench_cli(
    tmp_path: Path, make_video: MakeVideo, capsys: pytest.CaptureFixture[str]
) -> None:
    root = ["--data-root", str(tmp_path / "data")]
    assert dfbench([*root, "identity", "add", "p0001"]) == 1  # no consent reference
    for i in range(4):
        assert dfbench([*root, "identity", "add", f"p000{i}", "--consent-ref", f"C-{i}"]) == 0
        video = make_video(
            f"v{i}.mp4", frames=30, image_for=lambda k, i=i: barcode_frame(k + 100 * i)
        )
        assert dfbench([*root, "source", "add", str(video), "--identity", f"p000{i}"]) == 0
    assert dfbench([*root, "split"]) == 0
    assert dfbench([*root, "generate", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "planned" in out and "real" in out and "fake" in out
    assert dfbench([*root, "identity", "delete", "p0001"]) == 2  # needs --yes
    assert dfbench([*root, "identity", "delete", "p0001", "--yes"]) == 0
