"""The analysis pipeline: one streaming pass over the video, then aggregation.

    validate → decode + sample → detect → track → align → face quality → GenD (batched)
             → track scores → video probability → quality gate → verdict
             → intervals → evidence

`Analyzer` loads the models once and is reused across videos. It is not thread-safe:
the API runs a single worker thread that owns it (ADR 0005).
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from deepfake_core import __version__
from deepfake_core.config import PipelineConfig
from deepfake_core.detectors.base import FrameDetector
from deepfake_core.faces import FaceDetector, FaceTracker, YuNetDetector, align_face, model_input
from deepfake_core.models import GEND, YUNET, ModelSpec, models_dir
from deepfake_core.quality import (
    FACE_METRICS,
    FaceQuality,
    Grade,
    Tier,
    combine,
    face_metrics,
    grade,
    grade_face,
    grade_video,
    median_tier,
)
from deepfake_core.quality.gate import QualityReason, reason_for
from deepfake_core.schemas import (
    AnalysisResult,
    EvidenceFrame,
    ModelInfo,
    ProcessingInfo,
    QualityDetails,
    SuspiciousInterval,
    TimelinePoint,
    TrackResult,
    VideoMetadata,
)
from deepfake_core.schemas import QualityReason as QualityReasonOut
from deepfake_core.scoring import (
    CandidatePool,
    FrameScore,
    Interval,
    TrackScore,
    decide,
    score_track,
    select_evidence,
    suspicious_intervals,
    video_probability,
)
from deepfake_core.video import VideoReader, validate
from deepfake_core.video.decode import DecodeStats
from deepfake_core.video.probe import VideoInfo

ProgressCallback = Callable[[str, float], None]


@dataclass(slots=True)
class _Observation:
    raw_track: int
    time: float
    quality: FaceQuality
    p_fake: float | None = None
    embedding: npt.NDArray[np.float16] | None = None


@dataclass(frozen=True, slots=True)
class TraceTrack:
    track_id: int
    times: npt.NDArray[np.float32]
    p_fake: npt.NDArray[np.float32]
    """NaN where the frame was not scored."""
    tiers: npt.NDArray[np.int8]
    embeddings: npt.NDArray[np.float16]
    """(N, D); zeros where the frame was not scored."""


@dataclass(frozen=True, slots=True)
class AnalysisTrace:
    """Per-detection data for research (input of the step-14 temporal model). The API
    discards it; the benchmark stores it."""

    sample_rate: float
    tracks: tuple[TraceTrack, ...]

    def save_npz(self, path: str | os.PathLike[str]) -> None:
        arrays: dict[str, npt.NDArray[np.generic]] = {
            "sample_rate": np.array(self.sample_rate, np.float32)
        }
        for track in self.tracks:
            prefix = f"track{track.track_id}"
            arrays[f"{prefix}_times"] = track.times
            arrays[f"{prefix}_p_fake"] = track.p_fake
            arrays[f"{prefix}_tiers"] = track.tiers
            arrays[f"{prefix}_embeddings"] = track.embeddings
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EvidenceImage:
    evidence_id: str
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class AnalysisOutput:
    result: AnalysisResult
    evidence: tuple[EvidenceImage, ...]
    trace: AnalysisTrace


def model_info(spec: ModelSpec) -> ModelInfo:
    return ModelInfo(
        name=spec.name,
        role=spec.role,
        version=spec.version,
        revision=spec.revision,
        sha256=spec.sha256,
        license=spec.license,
    )


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _reasons_out(reasons: Sequence[QualityReason]) -> list[QualityReasonOut]:
    return [
        QualityReasonOut(code=r.code, message=r.message, level=r.tier.label, metric=r.metric)
        for r in reasons
    ]


class Analyzer:
    """Loads the models once; `analyze()` processes one video at a time."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        models_dir: str | os.PathLike[str] | None = None,
        device: str | None = None,
        face_detector: FaceDetector | None = None,
        frame_detector: FrameDetector | None = None,
        models: Sequence[ModelInfo] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.face_detector: FaceDetector = face_detector or YuNetDetector(
            self.config.detection, models_dir
        )
        if frame_detector is None:
            from deepfake_core.detectors.gend import GenDDetector  # torch is imported lazily

            frame_detector = GenDDetector(
                models_dir, device=device, batch_size=self.config.gend.batch_size
            )
        self.frame_detector: FrameDetector = frame_detector
        self.device = str(getattr(frame_detector, "device", "cpu"))
        self.models = list(models) if models is not None else [model_info(YUNET), model_info(GEND)]

    def warmup(self) -> None:
        warm = getattr(self.frame_detector, "warmup", None)
        if callable(warm):
            warm()

    def analyze(
        self,
        path: str | os.PathLike[str],
        *,
        analysis_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> AnalysisOutput:
        config = self.config
        report = progress or (lambda stage, fraction: None)
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        stage_s: dict[str, float] = defaultdict(float)

        report("validating", 0.0)
        info = validate(path, config.validation)
        stage_s["validate"] = time.perf_counter() - start

        reader = VideoReader(path, info, config.sampling, config.validation)
        tracker = FaceTracker(config.tracking)
        observations: list[_Observation] = []
        pending: list[tuple[_Observation, npt.NDArray[np.uint8]]] = []
        pool = CandidatePool(max(4 * config.aggregation.evidence_count, 16))
        faces_detected = 0

        def flush() -> None:
            if not pending:
                return
            t = time.perf_counter()
            predictions = self.frame_detector.predict([crop for _, crop in pending])
            for (obs, crop), p_fake, embedding in zip(
                pending, predictions.p_fake, predictions.embeddings, strict=True
            ):
                obs.p_fake = float(p_fake)
                obs.embedding = embedding.astype(np.float16)
                pool.offer(obs.p_fake, obs.raw_track, obs.time, crop)
            pending.clear()
            stage_s["score"] += time.perf_counter() - t

        report("analyzing", 0.0)
        loop_start = time.perf_counter()
        for frame in reader:
            t = time.perf_counter()
            detections = self.face_detector.detect(frame.image)
            stage_s["detect"] += time.perf_counter() - t
            faces_detected += len(detections)
            raw_ids = tracker.update(frame.time, detections)
            t = time.perf_counter()
            for detection, raw_id in zip(detections, raw_ids, strict=True):
                aligned = align_face(frame.image, detection.landmarks, scale=config.alignment.scale)
                if aligned is None:
                    continue
                crop = model_input(aligned.crop, config.alignment.model_input)
                metrics = face_metrics(
                    crop,
                    crop_size=aligned.size,
                    detector_score=detection.score,
                    landmarks=detection.landmarks,
                    alignment_residual=aligned.residual,
                )
                obs = _Observation(raw_id, frame.time, grade_face(metrics, config.quality))
                observations.append(obs)
                if obs.quality.tier > Tier.INSUFFICIENT:
                    pending.append((obs, crop))
            stage_s["align"] += time.perf_counter() - t
            if len(pending) >= config.gend.batch_size:
                flush()
            if info.duration:
                report("analyzing", min(0.99, frame.time / info.duration))
        flush()
        loop = time.perf_counter() - loop_start
        stage_s["decode"] = max(0.0, loop - stage_s["detect"] - stage_s["align"] - stage_s["score"])

        report("scoring", 0.0)
        t = time.perf_counter()
        output = self._finalize(
            info,
            reader.stats,
            tracker,
            observations,
            pool,
            analysis_id=analysis_id or uuid.uuid4().hex,
            faces_detected=faces_detected,
            started_at=started_at,
        )
        stage_s["finalize"] = time.perf_counter() - t
        result = output.result.model_copy(
            update={
                "processing": output.result.processing.model_copy(
                    update={
                        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                        "stage_ms": {k: round(v * 1000, 1) for k, v in stage_s.items()},
                    }
                )
            }
        )
        report("done", 1.0)
        return AnalysisOutput(result, output.evidence, output.trace)

    # ------------------------------------------------------------------ aggregation

    def _track_quality(self, observations: Sequence[_Observation]) -> Grade:
        tier = median_tier(o.quality.tier for o in observations)
        medians: dict[str, float | None] = {}
        for metric in FACE_METRICS:
            values = [
                v for o in observations if (v := o.quality.grade.metrics.get(metric)) is not None
            ]
            medians[metric] = float(np.median(values)) if values else None
        bands = {name: getattr(self.config.quality, name) for name in FACE_METRICS}
        graded = grade(medians, bands)
        reasons = graded.reasons
        if tier < Tier.GOOD and not reasons:
            reasons = (
                QualityReason(
                    "variable_face_quality",
                    "Face quality varies a lot across the video.",
                    tier,
                    "face",
                    None,
                ),
            )
        return Grade(tier, reasons, medians)

    def _finalize(
        self,
        info: VideoInfo,
        stats: DecodeStats,
        tracker: FaceTracker,
        observations: Sequence[_Observation],
        pool: CandidatePool,
        *,
        analysis_id: str,
        faces_detected: int,
        started_at: datetime,
    ) -> AnalysisOutput:
        config = self.config
        id_map = tracker.id_map()
        by_track: dict[int, list[_Observation]] = defaultdict(list)
        for obs in observations:
            final = id_map.get(obs.raw_track)
            if final is not None:
                by_track[final].append(obs)

        tracks: list[TrackScore] = []
        for track_id in sorted(by_track):
            members = sorted(by_track[track_id], key=lambda o: o.time)
            frames = [FrameScore(o.time, o.p_fake, o.quality.tier) for o in members]
            tracks.append(
                score_track(track_id, frames, self._track_quality(members), config.aggregation)
            )

        probability, decisive = video_probability(tracks, config.aggregation.track_score)
        video_grade = grade_video(info, stats, config.quality)
        if decisive is not None:
            overall = combine(video_grade, decisive.quality)
        else:
            no_face = Grade(Tier.INSUFFICIENT, (reason_for("no_face", None, Tier.INSUFFICIENT),))
            best = max(tracks, key=lambda t: (t.quality.tier, t.scored_frames), default=None)
            overall = combine(video_grade, no_face, *([best.quality] if best else []))
        decision = decide(
            probability,
            overall.tier,
            scored_frames=decisive.scored_frames if decisive else 0,
            iqr=decisive.iqr if decisive else None,
            config=config.verdict,
        )

        sample_interval = 1.0 / stats.sample_rate if stats.sample_rate else 0.25
        intervals: list[Interval] = []
        for track in tracks:
            if track.eligible:
                points = [(f.time, f.p_fake) for f in track.timeline if f.p_fake is not None]
                intervals += suspicious_intervals(
                    track.track_id, points, sample_interval, config.aggregation
                )
        intervals.sort(key=lambda iv: (iv.start, iv.track_id))

        eligible_ids = {t.track_id for t in tracks if t.eligible}
        candidates = []
        for candidate in pool.candidates():
            final = id_map.get(candidate.track_id)
            if final is not None and final in eligible_ids:
                candidate.track_id = final
                candidates.append(candidate)
        picks = select_evidence(
            candidates,
            intervals,
            count=config.aggregation.evidence_count,
            min_separation_s=config.aggregation.evidence_min_separation_s,
        )

        result = AnalysisResult(
            analysis_id=analysis_id,
            verdict=decision.verdict,
            manipulation_probability=_r(probability),
            confidence=decision.confidence,
            video_quality=overall.tier.label,
            face_tracks=len(tracks),
            detector_scores={} if probability is None else {"gend": round(probability, 4)},
            suspicious_intervals=[
                SuspiciousInterval(
                    start=round(iv.start, 3),
                    end=round(iv.end, 3),
                    score=round(iv.score, 4),
                    peak=round(iv.peak, 4),
                    track_id=iv.track_id,
                )
                for iv in intervals
            ],
            verdict_reasons=list(decision.reasons),
            quality=QualityDetails(
                level=overall.tier.label,
                reasons=_reasons_out(overall.reasons),
                metrics={k: _r(v) for k, v in overall.metrics.items()},
            ),
            tracks=[
                TrackResult(
                    track_id=t.track_id,
                    start=round(t.start, 3),
                    end=round(t.end, 3),
                    detections=t.detections,
                    scored_frames=t.scored_frames,
                    score=_r(t.score(config.aggregation.track_score)),
                    eligible=t.eligible,
                    quality=t.quality.tier.label,
                    quality_reasons=_reasons_out(t.quality.reasons),
                    timeline=[
                        TimelinePoint(t=round(f.time, 3), score=_r(f.p_fake), quality=f.tier.label)
                        for f in t.timeline
                    ],
                )
                for t in tracks
            ],
            evidence=[
                EvidenceFrame(
                    evidence_id=p.evidence_id,
                    track_id=p.track_id,
                    time=round(p.time, 3),
                    score=round(p.p_fake, 4),
                )
                for p in picks
            ],
            video=VideoMetadata(
                container=info.container,
                codec=info.codec,
                width=info.display_width,
                height=info.display_height,
                rotation=info.rotation,
                duration=_r(stats.duration, 3),
                fps_nominal=_r(info.fps, 3),
                fps_effective=_r(stats.effective_fps, 3),
                bit_rate=info.bit_rate,
                has_audio=info.has_audio,
            ),
            models=self.models,
            processing=ProcessingInfo(
                pipeline_version=__version__,
                config_profile=config.profile,
                config_hash=config.config_hash(),
                analyzer="real",
                device=self.device,
                sample_rate=round(stats.sample_rate, 4),
                frames_decoded=stats.frames_decoded,
                frames_sampled=stats.samples,
                faces_detected=faces_detected,
                frames_scored=sum(1 for o in observations if o.p_fake is not None),
                started_at=started_at,
                duration_ms=0.0,
                stage_ms={},
            ),
        )
        trace = AnalysisTrace(
            sample_rate=stats.sample_rate,
            tracks=tuple(
                self._trace_track(track_id, by_track[track_id]) for track_id in sorted(by_track)
            ),
        )
        evidence = tuple(EvidenceImage(p.evidence_id, p.jpeg) for p in picks)
        return AnalysisOutput(result, evidence, trace)

    @staticmethod
    def _trace_track(track_id: int, members: Sequence[_Observation]) -> TraceTrack:
        ordered = sorted(members, key=lambda o: o.time)
        dim = next((o.embedding.shape[0] for o in ordered if o.embedding is not None), 0)
        embeddings = np.zeros((len(ordered), dim), np.float16)
        for i, obs in enumerate(ordered):
            if obs.embedding is not None:
                embeddings[i] = obs.embedding
        return TraceTrack(
            track_id=track_id,
            times=np.array([o.time for o in ordered], np.float32),
            p_fake=np.array(
                [np.nan if o.p_fake is None else o.p_fake for o in ordered], np.float32
            ),
            tiers=np.array([int(o.quality.tier) for o in ordered], np.int8),
            embeddings=embeddings,
        )


_SHARED: dict[tuple[str, str], Analyzer] = {}


def analyze_video(
    path: str | os.PathLike[str], config: PipelineConfig | None = None
) -> AnalysisResult:
    """Analyze one video with default models (loaded once per config and reused)."""
    config = config or PipelineConfig()
    key = (config.config_hash(), str(models_dir().resolve()))
    analyzer = _SHARED.get(key)
    if analyzer is None:
        analyzer = _SHARED[key] = Analyzer(config)
    return analyzer.analyze(path).result
