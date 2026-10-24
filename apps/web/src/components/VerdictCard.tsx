import type { AnalysisResult } from "../api/types";
import { CONFIDENCE_TEXT, QUALITY_TEXT, VERDICT_TEXT, formatScore, reasonText } from "../lib/text";
import { VerdictIcon } from "./Icons";

function ScoreMeter({ result }: { result: AnalysisResult }) {
  const score = result.manipulation_probability;
  const { authentic, manipulated } = result.decision_thresholds;
  const percent = (value: number) => `${Math.round(value * 1000) / 10}%`;
  return (
    <div className="meter">
      <div className="meter-label">
        <span>Manipulation score</span>
        <strong>{formatScore(score)}</strong>
      </div>
      <div
        className="meter-track"
        role="meter"
        aria-label="Manipulation score"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={score ?? undefined}
        aria-valuetext={score === null ? "not available" : score.toFixed(2)}
      >
        <div className="meter-band meter-band-ok" style={{ width: percent(authentic) }} />
        <div
          className="meter-band meter-band-danger"
          style={{ left: percent(manipulated), width: percent(1 - manipulated) }}
        />
        {score !== null && <div className="meter-marker" style={{ left: percent(score) }} />}
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>0</span>
        <span style={{ left: percent(authentic) }}>{authentic.toFixed(2)}</span>
        <span style={{ left: percent(manipulated) }}>{manipulated.toFixed(2)}</span>
        <span>1</span>
      </div>
      <p className="hint">
        Uncalibrated detector score. At or above {manipulated.toFixed(2)} the video is flagged; at
        or below {authentic.toFixed(2)} it is considered likely authentic.
      </p>
    </div>
  );
}

export function VerdictCard({ result }: { result: AnalysisResult }) {
  const text = VERDICT_TEXT[result.verdict];
  // Demo mode is announced by a page banner, not repeated as a reason.
  const reasons = result.verdict_reasons.filter((code) => code !== "demo_mode");
  return (
    <section className={`card verdict verdict-${result.verdict}`} aria-labelledby="verdict-title">
      <div className="verdict-head">
        <VerdictIcon verdict={result.verdict} />
        <div>
          <h2 id="verdict-title">{text.title}</h2>
          <p className="verdict-summary">{text.summary}</p>
        </div>
      </div>
      <ul className="chips" aria-label="Summary">
        <li className={`chip chip-confidence-${result.confidence}`}>
          {CONFIDENCE_TEXT[result.confidence]}
        </li>
        <li className={`chip chip-quality-${result.video_quality}`}>
          Video quality: {QUALITY_TEXT[result.video_quality]}
        </li>
        <li className="chip">
          {result.face_tracks} {result.face_tracks === 1 ? "face" : "faces"} tracked
        </li>
      </ul>
      <ScoreMeter result={result} />
      {reasons.length > 0 && (
        <ul className="reasons">
          {reasons.map((code) => (
            <li key={code}>{reasonText(code)}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
