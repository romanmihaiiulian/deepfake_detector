import { useEffect, useRef, useState } from "react";

import { ApiError, getHealth, uploadVideo } from "../api/client";
import type { Health } from "../api/types";
import { Dropzone, ProgressPanel } from "../components/Upload";
import { validateFile } from "../lib/validation";
import { navigate } from "../router";

export function UploadPage() {
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((value) => !cancelled && setHealth(value))
      .catch(() => !cancelled && setHealth(null));
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleFile(file: File) {
    const problem = validateFile(file);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    setProgress(0);
    abort.current = new AbortController();
    try {
      const created = await uploadVideo(file, setProgress, abort.current.signal);
      navigate(`/analyses/${created.analysis_id}`);
    } catch (caught) {
      setProgress(null);
      setError(caught instanceof ApiError ? caught.message : "The upload failed.");
    }
  }

  return (
    <main className="page">
      <section className="intro">
        <h1>Check a video for face manipulation</h1>
        <p>
          Upload a recording, for example from a remote interview, to look for signs of face
          swapping. The result is a probabilistic assessment, not proof. Use it as a reason to
          verify identity, never as the sole basis for a decision about a person.
        </p>
      </section>
      {health?.analyzer === "fake" && (
        <p className="banner" role="status">
          Demo mode: no detection model runs on this server, and results are synthetic.
        </p>
      )}
      {health?.status === "loading" && (
        <p className="banner" role="status">
          The detection models are still loading. Uploads are queued until they are ready.
        </p>
      )}
      {progress === null ? (
        <Dropzone onFile={handleFile} />
      ) : (
        <ProgressPanel stage="uploading" progress={progress} onCancel={() => abort.current?.abort()} />
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <section className="privacy">
        <h2>What happens to your video</h2>
        <ul>
          <li>The video is deleted as soon as the analysis finishes.</li>
          <li>
            The result and up to four face snapshots are kept for 24 hours, so that you can come
            back to the link. You can delete them earlier.
          </li>
          <li>No faces are recognised or identified, and nothing is used for training.</li>
        </ul>
      </section>
    </main>
  );
}
