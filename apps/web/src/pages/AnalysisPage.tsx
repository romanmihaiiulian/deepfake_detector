import { useEffect, useState } from "react";

import { ApiError, deleteAnalysis, getAnalysis } from "../api/client";
import type { AnalysisResult, AnalysisStatus } from "../api/types";
import { DetectorScores, EvidenceGrid, ProcessingDetails, TrackList } from "../components/Details";
import { QualityPanel } from "../components/QualityPanel";
import { Timeline } from "../components/Timeline";
import { ProgressPanel } from "../components/Upload";
import { VerdictCard } from "../components/VerdictCard";
import { navigate } from "../router";

type State =
  | { kind: "loading" }
  | { kind: "status"; status: AnalysisStatus }
  | { kind: "missing" }
  | { kind: "error"; message: string };

const POLL_MS = 1000;

function downloadJson(result: AnalysisResult) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `analysis-${result.analysis_id}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function ResultView({ result }: { result: AnalysisResult }) {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onDelete() {
    if (!window.confirm("Delete this result and its face snapshots now? This cannot be undone.")) {
      return;
    }
    setDeleting(true);
    try {
      await deleteAnalysis(result.analysis_id);
      navigate("/", { replace: true });
    } catch (caught) {
      setDeleting(false);
      setError(caught instanceof ApiError ? caught.message : "Could not delete the result.");
    }
  }

  return (
    <>
      {result.processing.analyzer === "fake" && (
        <p className="banner" role="status">
          Demo mode: this result is synthetic, and no detection model ran.
        </p>
      )}
      <VerdictCard result={result} />
      <div className="grid-2">
        <QualityPanel result={result} />
        <DetectorScores result={result} />
      </div>
      <Timeline result={result} />
      <EvidenceGrid result={result} />
      <TrackList result={result} />
      <ProcessingDetails result={result} />
      <aside className="disclaimer" aria-label="Disclaimer">
        {result.disclaimer}
      </aside>
      <div className="actions">
        <button type="button" className="button" onClick={() => downloadJson(result)}>
          Download JSON
        </button>
        <button type="button" className="button danger" onClick={onDelete} disabled={deleting}>
          {deleting ? "Deleting…" : "Delete my data"}
        </button>
        <button type="button" className="button primary" onClick={() => navigate("/")}>
          Analyze another video
        </button>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </>
  );
}

export function AnalysisPage({ id }: { id: string }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let failures = 0;

    async function poll() {
      try {
        const status = await getAnalysis(id);
        if (cancelled) return;
        failures = 0;
        setState({ kind: "status", status });
        if (status.status === "queued" || status.status === "processing") {
          timer = window.setTimeout(poll, POLL_MS);
        }
      } catch (caught) {
        if (cancelled) return;
        if (caught instanceof ApiError && caught.status === 404) {
          setState({ kind: "missing" });
          return;
        }
        failures += 1;
        if (failures > 5) {
          setState({ kind: "error", message: "Lost contact with the server. Reload to try again." });
          return;
        }
        timer = window.setTimeout(poll, POLL_MS * 2 ** failures);
      }
    }

    // The page is keyed by analysis ID (App.tsx), so a new ID remounts it in "loading".
    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [id]);

  let body;
  if (state.kind === "loading") {
    body = <ProgressPanel stage="queued" progress={null} />;
  } else if (state.kind === "missing") {
    body = (
      <section className="card">
        <h2>This analysis no longer exists</h2>
        <p>Results are deleted 24 hours after the analysis, or earlier on request.</p>
        <button type="button" className="button primary" onClick={() => navigate("/")}>
          Analyze a video
        </button>
      </section>
    );
  } else if (state.kind === "error") {
    body = (
      <p className="error" role="alert">
        {state.message}
      </p>
    );
  } else if (state.status.status === "failed") {
    body = (
      <section className="card" role="alert">
        <h2>The analysis failed</h2>
        <p>{state.status.error?.message ?? "Unknown error."}</p>
        <button type="button" className="button primary" onClick={() => navigate("/")}>
          Try another video
        </button>
      </section>
    );
  } else if (state.status.status === "completed" && state.status.result) {
    body = <ResultView result={state.status.result} />;
  } else {
    body = (
      <ProgressPanel
        stage={state.status.status === "queued" ? "queued" : (state.status.stage ?? "analyzing")}
        progress={state.status.status === "queued" ? null : state.status.progress}
      />
    );
  }
  return <main className="page">{body}</main>;
}
