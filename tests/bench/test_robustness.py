from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deepfake_bench.facefusion import FaceFusion
from deepfake_bench.lab import create_variants, delete_identity, generate
from deepfake_bench.manifest import Registry
from deepfake_bench.paths import LabPaths
from deepfake_bench.report import build_report, quality_analysis
from deepfake_bench.variants import PROFILES, apply_profile
from deepfake_core.video import probe
from tests.bench.test_lab import BOX, CONFIG, StubFaces, populate, recipe
from tests.conftest import requires_ffmpeg, write_video

MakeVideo = Callable[..., Path]


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize(
    ("profile", "size", "fps", "codec"),
    [
        ("conf_low", (640, 360), 15, "h264"),
        ("conf_medium", (960, 540), 24, "h264"),
        ("conf_high", (1280, 720), 30, "h264"),
        ("h265_crf28", (1280, 720), 30, "hevc"),
        ("res_480p", (854, 480), 30, "h264"),
        ("fps_15", (1280, 720), 15, "h264"),
        ("crop_10", (1152, 648), 30, "h264"),
        ("letterbox", (1280, 960), 30, "h264"),
        ("screen_recording", (1920, 1080), 30, "h264"),
    ],
)
def test_profiles_produce_the_intended_degradation(
    tmp_path: Path, profile: str, size: tuple[int, int], fps: int, codec: str
) -> None:
    source = write_video(
        tmp_path / "src.mp4", frames=30, fps=30, size=(1280, 720), codec_options={"crf": "0"}
    )
    encoded = apply_profile(PROFILES[profile], source, tmp_path / f"{profile}.mp4")
    info = probe(encoded.path)
    assert (info.display_width, info.display_height) == size
    assert info.fps == pytest.approx(fps)
    assert info.codec == codec and info.has_audio is False
    assert encoded.argv[0] == "ffmpeg" and encoded.ffmpeg_version.startswith("ffmpeg version")
    assert list(tmp_path.glob("*.call.webm")) == []  # intermediate passes are cleaned up


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_variants_keep_pairs_comparable(
    tmp_path: Path, make_video: MakeVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FACEFUSION_BOX", ",".join(map(str, BOX)))
    registry = Registry(LabPaths(tmp_path / "data"))
    populate(registry, make_video)
    generate(registry, CONFIG, FaceFusion.fake(), StubFaces(), log=lambda _: None)
    pristine = [s for s in registry.ok_samples() if s.transformation == "pristine"]

    report = create_variants(registry, ("conf_low", "h264_crf28"), log=lambda _: None)
    assert len(report.created) == 2 * len(pristine) and not report.missing_intermediate
    variants = [s for s in registry.ok_samples() if s.parent_sample_id]
    for variant in variants:
        parent = registry.samples[variant.parent_sample_id or ""]
        assert (variant.label, variant.pair_id, variant.split) == (
            parent.label,
            parent.pair_id,
            parent.split,
        )
        assert variant.swapper_model == parent.swapper_model
        assert registry.paths.absolute(variant.path).is_file()
    conf = [v for v in variants if v.transformation == "conf_low"]
    assert all(v.resolution == "320x240" and v.fps == pytest.approx(15) for v in conf)
    by_pair: dict[tuple[str, str], set[tuple[str, ...]]] = {}
    for v in variants:
        by_pair.setdefault((v.pair_id, v.transformation), set()).add(
            tuple(
                recipe(
                    v.transformation_argv[: v.transformation_argv.index("&&")]
                    if "&&" in v.transformation_argv
                    else v.transformation_argv
                )
            )
        )
    assert all(len(recipes) == 1 for recipes in by_pair.values())  # identical recipe per pair

    again = create_variants(
        Registry(registry.paths), ("conf_low", "h264_crf28"), log=lambda _: None
    )
    assert again.created == [] and again.skipped == len(report.created)

    delete_identity(registry, "p0000")
    left = Registry(registry.paths)
    assert not any("p0000" in (s.source_identity, s.target_identity) for s in left.samples.values())


def write_run(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "synthetic",
                "pipeline_version": "0.1.0",
                "config_profile": "uncalibrated-v0",
                "config_hash": "x",
                "operating_threshold": 0.8,
                "models": [{"name": "GenD", "revision": "0123456789abc"}],
                "held_out_swappers": [],
            }
        )
    )
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir


def synthetic(
    label: str, transformation: str, score: float, pair: int, **extra: Any
) -> dict[str, Any]:
    return {
        "sample_id": f"{label}-{transformation}-{pair}",
        "label": label,
        "split": "test",
        "pair_id": f"v{pair}",
        "transformation": transformation,
        "swapper_model": "inswapper_128" if label == "manipulated" else None,
        "dataset": "lab",
        "resolution": "1280x720",
        "score": score,
        "scores": {"mean": score, "median": score, "window_max": score},
        "verdict": "manipulated_likely"
        if score >= 0.8
        else "authentic_likely"
        if score <= 0.3
        else "inconclusive",
        "video_quality": "good" if transformation == "pristine" else "poor",
        **extra,
    }


def test_robustness_section_reports_false_positives_per_profile(tmp_path: Path) -> None:
    records = []
    for pair in range(8):
        records.append(synthetic("real", "pristine", 0.1, pair))
        records.append(synthetic("manipulated", "pristine", 0.95, pair))
        records.append(synthetic("real", "conf_low", 0.85 if pair < 4 else 0.2, pair))  # FPs
        records.append(synthetic("manipulated", "conf_low", 0.8, pair))  # now below half the reals
    report = build_report(write_run(tmp_path, records), bootstrap=50)
    conf = report["robustness"]["profiles"]["conf_low"]
    assert conf["fpr_on_reals"] == pytest.approx(0.5)
    assert conf["roc_auc"] == pytest.approx(0.5) and conf["delta_roc_auc"] == pytest.approx(-0.5)
    assert conf["quality_tiers"] == {"poor": 16}
    assert "group:conferencing" in report["slices"]
    markdown = (tmp_path / "run" / "report.md").read_text()
    assert "## Robustness (lab, per encoding profile)" in markdown
    assert "Not enough real videos per quality bin" in markdown
    assert (tmp_path / "run" / "fpr_by_profile.png").is_file()


def test_quality_analysis_suggests_a_poor_boundary_only_with_enough_data() -> None:
    rng = np.random.default_rng(0)
    records = []
    for i in range(200):
        size = float(rng.uniform(40, 400))
        false_positive = size < 120 and rng.random() < 0.6
        records.append(
            synthetic(
                "real",
                "pristine",
                0.9 if false_positive else 0.1,
                i,
                quality_metrics={"face_crop_size": size},
            )
        )
    result = quality_analysis(records, 0.8)
    crop = result["metrics"]["face_crop_size"]
    assert crop["status"] == "ok"
    assert crop["bins"][0]["fpr"] > 0.3 and crop["bins"][-1]["fpr"] == 0.0
    assert 100 <= crop["suggested_poor_below"] <= 160
    assert result["metrics"]["face_sharpness"]["status"] == "insufficient data"
    small = quality_analysis(records[:30], 0.8)
    assert small["metrics"]["face_crop_size"]["status"] == "insufficient data"
