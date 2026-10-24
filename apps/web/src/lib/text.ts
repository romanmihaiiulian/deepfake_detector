import type { Confidence, QualityLevel, Verdict } from "../api/types";

export const VERDICT_TEXT: Record<Verdict, { title: string; summary: string }> = {
  manipulated_likely: {
    title: "Likely manipulated",
    summary:
      "Signs of face manipulation were found. Verify the person's identity through another " +
      "channel before relying on this video.",
  },
  authentic_likely: {
    title: "Likely authentic",
    summary:
      "No signs of face manipulation were found. This does not prove that the video is genuine.",
  },
  inconclusive: {
    title: "Inconclusive",
    summary: "The analysis could not reach a reliable conclusion.",
  },
};

export const REASON_TEXT: Record<string, string> = {
  score_above_threshold: "The manipulation score is above the decision threshold.",
  score_below_threshold: "The manipulation score is below the threshold for authentic video.",
  score_in_uncertain_band: "The score falls between the decision thresholds: the evidence is mixed.",
  poor_quality_needs_stronger_evidence:
    "Because of the low video quality, stronger evidence is needed before flagging manipulation.",
  reduced_video_quality: "Video quality is reduced, which lowers confidence.",
  insufficient_quality: "The video quality is too low for a reliable assessment.",
  no_face: "No face was visible long enough to analyze.",
  demo_mode: "Demo mode: this result is synthetic, no detection model ran.",
};

export const STAGE_TEXT: Record<string, string> = {
  uploading: "Uploading",
  queued: "Waiting for a free analysis slot",
  validating: "Checking the video",
  analyzing: "Finding and analyzing faces",
  scoring: "Scoring",
  done: "Finishing",
};

export const CONFIDENCE_TEXT: Record<Confidence, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence",
};

export const QUALITY_TEXT: Record<QualityLevel, string> = {
  good: "Good",
  fair: "Fair",
  poor: "Poor",
  insufficient: "Insufficient",
};

export function reasonText(code: string): string {
  return REASON_TEXT[code] ?? code.replaceAll("_", " ");
}

export function formatScore(score: number | null | undefined): string {
  return score === null || score === undefined ? "n/a" : score.toFixed(2);
}

export function formatSeconds(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return minutes > 0 ? `${minutes}:${rest.toFixed(1).padStart(4, "0")}` : `${rest.toFixed(1)} s`;
}
