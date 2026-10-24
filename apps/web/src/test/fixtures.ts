import type { AnalysisResult, TimelinePoint } from "../api/types";

function timeline(scores: Array<number | null>): TimelinePoint[] {
  return scores.map((score, i) => ({ t: i / 4, score, quality: score === null ? "insufficient" : "good" }));
}

export function makeResult(overrides: Partial<AnalysisResult> = {}): AnalysisResult {
  const scores = [0.2, 0.25, 0.9, 0.95, 0.92, null, 0.3, 0.22];
  return {
    schema_version: "1.0",
    analysis_id: "0123456789abcdef0123456789abcdef",
    verdict: "manipulated_likely",
    manipulation_probability: 0.86,
    confidence: "medium",
    video_quality: "fair",
    face_tracks: 2,
    detector_scores: { gend: 0.86 },
    suspicious_intervals: [{ start: 0.5, end: 1.25, score: 0.92, peak: 0.95, track_id: 0 }],
    verdict_reasons: ["score_above_threshold", "reduced_video_quality"],
    decision_thresholds: { authentic: 0.3, manipulated: 0.8 },
    quality: {
      level: "fair",
      reasons: [{ code: "small_face", message: "Faces are small in the frame (150 px).", level: "fair", metric: "face_crop_size" }],
      metrics: { resolution_short_side: 720, face_crop_size: 150 },
    },
    tracks: [
      {
        track_id: 0,
        start: 0,
        end: 1.75,
        detections: 8,
        scored_frames: 7,
        score: 0.86,
        eligible: true,
        quality: "fair",
        quality_reasons: [],
        timeline: timeline(scores),
      },
      {
        track_id: 1,
        start: 0,
        end: 1.75,
        detections: 8,
        scored_frames: 8,
        score: 0.2,
        eligible: true,
        quality: "good",
        quality_reasons: [],
        timeline: timeline([0.2, 0.2, 0.21, 0.19, 0.2, 0.2, 0.18, 0.2]),
      },
    ],
    evidence: [{ evidence_id: "e0", track_id: 0, time: 0.75, score: 0.95, kind: "face_crop" }],
    video: {
      container: "mov,mp4,m4a,3gp,3g2,mj2",
      codec: "h264",
      width: 1280,
      height: 720,
      rotation: 0,
      duration: 2,
      fps_nominal: 30,
      fps_effective: 30,
      bit_rate: 2_000_000,
      has_audio: false,
    },
    models: [
      { name: "GenD CLIP ViT-L/14", role: "frame_detector", version: "GenD_CLIP_L_14", revision: "891ce014a0308386c4d7d25b3dcf436a22db5504", sha256: "d76f", license: "MIT" },
    ],
    processing: {
      pipeline_version: "0.1.0",
      config_profile: "uncalibrated-v0",
      config_hash: "abc",
      analyzer: "real",
      device: "mps",
      sample_rate: 4,
      frames_decoded: 60,
      frames_sampled: 8,
      faces_detected: 16,
      frames_scored: 15,
      started_at: "2026-10-24T10:00:00Z",
      duration_ms: 1500,
      stage_ms: {},
    },
    calibration: { method: "none", note: "Scores are raw detector probabilities." },
    disclaimer: "This is a probabilistic assessment, not proof.",
    ...overrides,
  };
}
