"""Aggregation, suspicious intervals, verdicts and evidence."""

from deepfake_core.scoring.aggregate import FrameScore, TrackScore, score_track, video_probability
from deepfake_core.scoring.evidence import CandidatePool, EvidencePick, select_evidence
from deepfake_core.scoring.intervals import Interval, suspicious_intervals
from deepfake_core.scoring.verdict import Decision, decide

__all__ = [
    "CandidatePool",
    "Decision",
    "EvidencePick",
    "FrameScore",
    "Interval",
    "TrackScore",
    "decide",
    "score_track",
    "select_evidence",
    "suspicious_intervals",
    "video_probability",
]
