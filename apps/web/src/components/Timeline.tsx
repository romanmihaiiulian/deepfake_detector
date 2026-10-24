import { useCallback, useId, useMemo, useState } from "react";

import type { AnalysisResult, TimelinePoint } from "../api/types";
import { QUALITY_TEXT, formatScore, formatSeconds } from "../lib/text";

const DEFAULT_WIDTH = 720;
const MIN_WIDTH = 300;
const HEIGHT = 220;
const M = { top: 12, right: 16, bottom: 28, left: 40 };
const INNER_H = HEIGHT - M.top - M.bottom;
const DASHES = ["", "6 4", "2 3", "10 3 2 3"];
const TICK_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];

type Hover = { track: number; point: TimelinePoint };

function niceStep(duration: number, maxTicks: number): number {
  return TICK_STEPS.find((step) => duration / step <= maxTicks) ?? 300;
}

/** SVG path for one track; `null` scores (unscored frames) break the line. */
function pathFor(points: TimelinePoint[], x: (t: number) => number, y: (s: number) => number): string {
  let path = "";
  let drawing = false;
  for (const point of points) {
    if (point.score === null || point.score === undefined) {
      drawing = false;
      continue;
    }
    path += `${drawing ? "L" : "M"}${x(point.t).toFixed(1)},${y(point.score).toFixed(1)}`;
    drawing = true;
  }
  return path;
}

export function Timeline({ result }: { result: AnalysisResult }) {
  const [hover, setHover] = useState<Hover | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const hatchId = useId();
  // Draw at the container's real pixel width, so text keeps its size on narrow screens.
  const measure = useCallback((node: HTMLElement | null) => {
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(MIN_WIDTH, Math.round(entry.contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  const { tracks, suspicious_intervals: intervals, decision_thresholds: thresholds } = result;

  const duration = useMemo(() => {
    const last = Math.max(0, ...tracks.flatMap((t) => t.timeline.map((p) => p.t)));
    return Math.max(result.video.duration ?? 0, last, 1);
  }, [tracks, result.video.duration]);
  const innerW = width - M.left - M.right;
  const x = (t: number) => M.left + (t / duration) * innerW;
  const y = (s: number) => M.top + (1 - s) * INNER_H;
  const step = niceStep(duration, Math.max(2, Math.floor(innerW / 80)));
  const ticks = Array.from({ length: Math.floor(duration / step) + 1 }, (_, i) => i * step);

  if (tracks.length === 0) {
    return (
      <section className="card" aria-labelledby="timeline-title">
        <h3 id="timeline-title">Timeline</h3>
        <p>No face was tracked, so there is no timeline.</p>
      </section>
    );
  }

  function onMove(event: React.MouseEvent<SVGRectElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const t = ((event.clientX - box.left) / box.width) * duration;
    let best: Hover | null = null;
    let bestDistance = Infinity;
    tracks.forEach((track, index) => {
      for (const point of track.timeline) {
        const distance = Math.abs(point.t - t);
        if (point.score !== null && distance < bestDistance) {
          bestDistance = distance;
          best = { track: index, point };
        }
      }
    });
    setHover(best);
  }

  const summary =
    `Manipulation score over time for ${tracks.length} face track${tracks.length === 1 ? "" : "s"}` +
    (intervals.length ? `, with ${intervals.length} suspicious interval${intervals.length === 1 ? "" : "s"}.` : ".");

  return (
    <section className="card" aria-labelledby="timeline-title">
      <div className="card-head">
        <h3 id="timeline-title">Timeline</h3>
        <button type="button" className="link-button" onClick={() => setShowTable((v) => !v)}>
          {showTable ? "Show chart" : "Show as table"}
        </button>
      </div>
      {showTable ? (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Face</th>
                <th scope="col">Time</th>
                <th scope="col">Score</th>
                <th scope="col">Frame quality</th>
              </tr>
            </thead>
            <tbody>
              {tracks.flatMap((track) =>
                track.timeline.map((point) => (
                  <tr key={`${track.track_id}-${point.t}`}>
                    <td>Face {track.track_id + 1}</td>
                    <td>{formatSeconds(point.t)}</td>
                    <td>{point.score === null ? "not scored" : formatScore(point.score)}</td>
                    <td>{QUALITY_TEXT[point.quality]}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <figure className="timeline" ref={measure}>
          <svg viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label={summary}>
            <defs>
              <pattern id={hatchId} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="6" height="6" className="interval-fill" />
                <line x1="0" y1="0" x2="0" y2="6" className="interval-hatch" />
              </pattern>
            </defs>
            {intervals.map((interval) => (
              <rect
                key={`${interval.track_id}-${interval.start}`}
                data-testid="interval"
                x={x(interval.start)}
                y={M.top}
                width={Math.max(2, x(interval.end) - x(interval.start))}
                height={INNER_H}
                fill={`url(#${hatchId})`}
              >
                <title>
                  Suspicious: {formatSeconds(interval.start)} to {formatSeconds(interval.end)}, face{" "}
                  {interval.track_id + 1}, score {formatScore(interval.score)}
                </title>
              </rect>
            ))}
            {[0, 0.5, 1].map((value) => (
              <g key={value}>
                <line className="grid" x1={M.left} x2={width - M.right} y1={y(value)} y2={y(value)} />
                <text className="axis" x={M.left - 6} y={y(value) + 4} textAnchor="end">
                  {value}
                </text>
              </g>
            ))}
            <line className="threshold threshold-danger" x1={M.left} x2={width - M.right} y1={y(thresholds.manipulated)} y2={y(thresholds.manipulated)} />
            <line className="threshold threshold-ok" x1={M.left} x2={width - M.right} y1={y(thresholds.authentic)} y2={y(thresholds.authentic)} />
            {ticks.map((t) => (
              <text key={t} className="axis" x={x(t)} y={HEIGHT - 8} textAnchor="middle">
                {formatSeconds(t)}
              </text>
            ))}
            {tracks.map((track, index) => (
              <path
                key={track.track_id}
                data-testid="track-line"
                className={`track-line track-${index % 4}`}
                d={pathFor(track.timeline, x, y)}
                strokeDasharray={DASHES[index % 4]}
              />
            ))}
            {hover && hover.point.score !== null && (
              <g className="hover" pointerEvents="none">
                <line x1={x(hover.point.t)} x2={x(hover.point.t)} y1={M.top} y2={M.top + INNER_H} />
                <circle cx={x(hover.point.t)} cy={y(hover.point.score)} r={4} className={`track-${hover.track % 4}`} />
              </g>
            )}
            <rect
              x={M.left}
              y={M.top}
              width={innerW}
              height={INNER_H}
              fill="transparent"
              onMouseMove={onMove}
              onMouseLeave={() => setHover(null)}
            />
          </svg>
          <figcaption>
            <div className="legend">
              {tracks.map((track, index) => (
                <span key={track.track_id} className="legend-item">
                  <svg width="28" height="10" aria-hidden="true">
                    <line x1="0" y1="5" x2="28" y2="5" className={`track-line track-${index % 4}`} strokeDasharray={DASHES[index % 4]} />
                  </svg>
                  Face {track.track_id + 1}
                </span>
              ))}
              {intervals.length > 0 && (
                <span className="legend-item">
                  <span className="legend-swatch" aria-hidden="true" /> Suspicious interval
                </span>
              )}
            </div>
            <div className="tooltip" aria-live="polite">
              {hover && hover.point.score !== null
                ? `Face ${tracks[hover.track]!.track_id + 1} at ${formatSeconds(hover.point.t)}: score ${formatScore(hover.point.score)} (quality ${QUALITY_TEXT[hover.point.quality].toLowerCase()})`
                : "Hover over the chart to see individual frame scores."}
            </div>
          </figcaption>
        </figure>
      )}
    </section>
  );
}
