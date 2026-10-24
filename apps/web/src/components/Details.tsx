import { evidenceUrl } from "../api/client";
import type { AnalysisResult } from "../api/types";
import { QUALITY_TEXT, formatScore, formatSeconds } from "../lib/text";

export function DetectorScores({ result }: { result: AnalysisResult }) {
  const entries = Object.entries(result.detector_scores);
  return (
    <section className="card" aria-labelledby="detectors-title">
      <h3 id="detectors-title">Detector scores</h3>
      {entries.length === 0 ? (
        <p>No detector produced a score (no eligible face).</p>
      ) : (
        <table className="data-table compact">
          <tbody>
            {entries.map(([name, score]) => (
              <tr key={name}>
                <th scope="row">{name === "gend" ? "GenD (frame detector)" : name}</th>
                <td>{formatScore(score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="hint">{result.calibration.note}</p>
    </section>
  );
}

export function TrackList({ result }: { result: AnalysisResult }) {
  if (result.tracks.length === 0) return null;
  return (
    <section className="card" aria-labelledby="tracks-title">
      <h3 id="tracks-title">Face tracks</h3>
      <ul className="tracks">
        {result.tracks.map((track) => (
          <li key={track.track_id}>
            <strong>Face {track.track_id + 1}</strong>: {formatSeconds(track.start)} to{" "}
            {formatSeconds(track.end)}, score {formatScore(track.score)}, quality{" "}
            {QUALITY_TEXT[track.quality].toLowerCase()}, {track.scored_frames} of {track.detections}{" "}
            frames scored
            {!track.eligible && <em> (not used for the verdict: too few usable frames)</em>}
          </li>
        ))}
      </ul>
    </section>
  );
}

export function EvidenceGrid({ result }: { result: AnalysisResult }) {
  if (result.evidence.length === 0) return null;
  return (
    <section className="card" aria-labelledby="evidence-title">
      <h3 id="evidence-title">Highest-scoring frames</h3>
      <div className="evidence">
        {result.evidence.map((item) => (
          <figure key={item.evidence_id}>
            <img
              src={evidenceUrl(result.analysis_id, item.evidence_id)}
              alt={`Face ${item.track_id + 1} at ${formatSeconds(item.time)}, score ${formatScore(item.score)}`}
              width={160}
              height={160}
              loading="lazy"
            />
            <figcaption>
              Face {item.track_id + 1} · {formatSeconds(item.time)} · {formatScore(item.score)}
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="hint">Aligned face crops, as the detector saw them. Deleted with the result.</p>
    </section>
  );
}

export function ProcessingDetails({ result }: { result: AnalysisResult }) {
  const { processing: p, video } = result;
  return (
    <details className="card details">
      <summary>Processing details</summary>
      <dl className="facts">
        <dt>Analysis ID</dt>
        <dd>
          <code>{result.analysis_id}</code>
        </dd>
        <dt>Pipeline</dt>
        <dd>
          {p.pipeline_version}, profile {p.config_profile} (config {p.config_hash})
        </dd>
        <dt>Analyzer</dt>
        <dd>
          {p.analyzer} on {p.device}
        </dd>
        <dt>Video</dt>
        <dd>
          {video.codec} in {video.container.split(",")[0]}, {video.width}×{video.height},{" "}
          {video.duration !== null ? formatSeconds(video.duration) : "unknown length"}
          {video.fps_nominal !== null && `, ${video.fps_nominal.toFixed(1)} fps`}
        </dd>
        <dt>Frames</dt>
        <dd>
          {p.frames_decoded} decoded, {p.frames_sampled} sampled at {p.sample_rate} Hz,{" "}
          {p.faces_detected} faces detected, {p.frames_scored} scored
        </dd>
        <dt>Duration</dt>
        <dd>{(p.duration_ms / 1000).toFixed(1)} s</dd>
      </dl>
      <table className="data-table compact">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Version</th>
            <th scope="col">Revision</th>
            <th scope="col">License</th>
          </tr>
        </thead>
        <tbody>
          {result.models.map((model) => (
            <tr key={model.name}>
              <td>{model.name}</td>
              <td>{model.version}</td>
              <td>
                <code title={model.sha256}>{model.revision.slice(0, 10)}</code>
              </td>
              <td>{model.license}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
