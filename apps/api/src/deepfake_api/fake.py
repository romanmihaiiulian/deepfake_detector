"""A fake analyzer (DFD_ANALYZER=fake) for UI development and end-to-end tests.

The input is really validated, but no model runs. The result is canned and deterministic:
the same file always gives the same verdict, derived from its sha256. Results are marked
`processing.analyzer = "fake"`, `/health` reports it, and the UI shows a banner.
"""

from __future__ import annotations

import hashlib
import io
import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from deepfake_core import __version__
from deepfake_core.config import PipelineConfig
from deepfake_core.models import GEND, YUNET
from deepfake_core.pipeline import (
    AnalysisOutput,
    AnalysisTrace,
    EvidenceImage,
    ProgressCallback,
    model_info,
)
from deepfake_core.schemas import (
    AnalysisResult,
    DecisionThresholds,
    EvidenceFrame,
    ProcessingInfo,
    QualityDetails,
    SuspiciousInterval,
    TimelinePoint,
    TrackResult,
    Verdict,
    VideoMetadata,
)
from deepfake_core.video import validate

VERDICTS: tuple[Verdict, ...] = ("authentic_likely", "inconclusive", "manipulated_likely")
BASE_SCORE = {"authentic_likely": 0.12, "inconclusive": 0.5, "manipulated_likely": 0.9}


def _demo_crop(label: str) -> bytes:
    image = Image.new("RGB", (224, 224), (92, 98, 112))
    draw = ImageDraw.Draw(image)
    draw.ellipse((52, 32, 172, 192), fill=(150, 156, 170))
    draw.text((76, 196), label, fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class FakeAnalyzer:
    device = "none"

    def __init__(self, delay_s: float = 0.0, config: PipelineConfig | None = None) -> None:
        self.delay_s = delay_s
        self.config = config or PipelineConfig()
        self.models = [model_info(YUNET), model_info(GEND)]

    def warmup(self) -> None:
        return None

    def analyze(
        self,
        path: Path,
        *,
        analysis_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> AnalysisOutput:
        report = progress or (lambda stage, fraction: None)
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        report("validating", 0.0)
        info = validate(path, self.config.validation)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        verdict = VERDICTS[int(digest[16:24], 16) % 3]
        for step in range(10):
            report("analyzing", step / 10)
            time.sleep(self.delay_s / 10)
        report("scoring", 0.0)

        duration = info.duration or 5.0
        times = [round(i / 4, 3) for i in range(max(8, int(duration * 4)))]
        base = BASE_SCORE[verdict]
        burst = (duration / 3, 2 * duration / 3)
        timeline = []
        for t in times:
            score = base + rng.uniform(-0.06, 0.06)
            if verdict == "manipulated_likely" and burst[0] <= t <= burst[1]:
                score = 0.96 + rng.uniform(-0.02, 0.02)
            timeline.append(
                TimelinePoint(t=t, score=round(min(max(score, 0), 1), 4), quality="good")
            )
        mean = round(sum(p.score or 0 for p in timeline) / len(timeline), 4)
        intervals = (
            [
                SuspiciousInterval(
                    start=round(burst[0], 3),
                    end=round(burst[1], 3),
                    score=0.96,
                    peak=0.98,
                    track_id=0,
                )
            ]
            if verdict == "manipulated_likely"
            else []
        )
        evidence = [
            EvidenceFrame(evidence_id=f"e{i}", track_id=0, time=round(t, 3), score=0.9)
            for i, t in enumerate((burst[0], burst[1]))
        ]
        images = tuple(EvidenceImage(e.evidence_id, _demo_crop("DEMO")) for e in evidence)
        verdict_config = self.config.verdict
        result = AnalysisResult(
            analysis_id=analysis_id or uuid.uuid4().hex,
            verdict=verdict,
            manipulation_probability=mean,
            confidence="medium" if verdict != "inconclusive" else "low",
            video_quality="good",
            face_tracks=1,
            detector_scores={"gend": mean},
            suspicious_intervals=intervals,
            verdict_reasons=[
                {
                    "authentic_likely": "score_below_threshold",
                    "inconclusive": "score_in_uncertain_band",
                    "manipulated_likely": "score_above_threshold",
                }[verdict],
                "demo_mode",
            ],
            decision_thresholds=DecisionThresholds(
                authentic=verdict_config.authentic_threshold,
                manipulated=verdict_config.manipulated_threshold,
            ),
            quality=QualityDetails(level="good", reasons=[], metrics={}),
            tracks=[
                TrackResult(
                    track_id=0,
                    start=times[0],
                    end=times[-1],
                    detections=len(times),
                    scored_frames=len(times),
                    score=mean,
                    eligible=True,
                    quality="good",
                    quality_reasons=[],
                    timeline=timeline,
                )
            ],
            evidence=evidence,
            video=VideoMetadata(
                container=info.container,
                codec=info.codec,
                width=info.display_width,
                height=info.display_height,
                rotation=info.rotation,
                duration=info.duration,
                fps_nominal=info.fps,
                fps_effective=info.fps,
                bit_rate=info.bit_rate,
                has_audio=info.has_audio,
            ),
            models=self.models,
            processing=ProcessingInfo(
                pipeline_version=__version__,
                config_profile=self.config.profile,
                config_hash=self.config.config_hash(),
                analyzer="fake",
                device=self.device,
                sample_rate=4.0,
                frames_decoded=info.frame_count or 0,
                frames_sampled=len(times),
                faces_detected=len(times),
                frames_scored=len(times),
                started_at=started_at,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
                stage_ms={},
            ),
        )
        report("done", 1.0)
        return AnalysisOutput(result, images, AnalysisTrace(sample_rate=4.0, tracks=()))
