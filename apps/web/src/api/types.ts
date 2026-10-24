import type { components } from "./schema";

type Schemas = components["schemas"];

export type AnalysisCreated = Schemas["AnalysisCreated"];
export type AnalysisStatus = Schemas["AnalysisStatus"];
export type AnalysisResult = Schemas["AnalysisResult"];
export type TrackResult = Schemas["TrackResult"];
export type TimelinePoint = Schemas["TimelinePoint"];
export type SuspiciousInterval = Schemas["SuspiciousInterval"];
export type EvidenceFrame = Schemas["EvidenceFrame"];
export type QualityReason = Schemas["QualityReason"];
export type ModelInfo = Schemas["ModelInfo"];
export type Health = Schemas["Health"];
export type ErrorResponse = Schemas["ErrorResponse"];

export type Verdict = AnalysisResult["verdict"];
export type Confidence = AnalysisResult["confidence"];
export type QualityLevel = AnalysisResult["video_quality"];
