import type { AnalysisResult } from "../api/types";
import { QUALITY_TEXT } from "../lib/text";

const METRIC_LABELS: Record<string, (value: number) => string> = {
  resolution_short_side: (v) => `${v.toFixed(0)} px resolution`,
  effective_fps: (v) => `${v.toFixed(1)} fps effective`,
  face_crop_size: (v) => `${v.toFixed(0)} px faces`,
  frame_availability: (v) => `${(v * 100).toFixed(0)}% frames available`,
};

export function QualityPanel({ result }: { result: AnalysisResult }) {
  const { quality } = result;
  const facts = Object.entries(METRIC_LABELS)
    .map(([key, label]) => {
      const value = quality.metrics[key];
      return value === null || value === undefined ? null : label(value);
    })
    .filter((fact): fact is string => fact !== null);
  return (
    <section className="card" aria-labelledby="quality-title">
      <h3 id="quality-title">
        Video quality <span className={`chip chip-quality-${quality.level}`}>{QUALITY_TEXT[quality.level]}</span>
      </h3>
      {quality.reasons.length > 0 ? (
        <ul className="reasons">
          {quality.reasons.map((reason) => (
            <li key={reason.code}>{reason.message}</li>
          ))}
        </ul>
      ) : (
        <p>No quality problems were detected.</p>
      )}
      {facts.length > 0 && <p className="hint">{facts.join(" · ")}</p>}
      <p className="hint">
        Low quality never raises the manipulation score. It makes the verdict more cautious.
      </p>
    </section>
  );
}
