import { useRef, useState } from "react";

import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB } from "../lib/validation";
import { STAGE_TEXT } from "../lib/text";

export function Dropzone({ onFile }: { onFile: (file: File) => void }) {
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  return (
    <div
      className={`dropzone${dragging ? " dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer.files[0];
        if (file) onFile(file);
      }}
    >
      <p className="dropzone-title">Drop a video here</p>
      <p className="hint">
        MP4, MOV or WEBM · up to {MAX_UPLOAD_MB} MB · 1 second to 10 minutes
      </p>
      <button type="button" className="button primary" onClick={() => input.current?.click()}>
        Choose a video
      </button>
      <input
        ref={input}
        type="file"
        className="visually-hidden"
        data-testid="file-input"
        aria-label="Choose a video file"
        accept={[...ACCEPTED_EXTENSIONS, "video/mp4", "video/quicktime", "video/webm"].join(",")}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.target.value = "";
        }}
      />
    </div>
  );
}

export function ProgressPanel({
  stage,
  progress,
  onCancel,
}: {
  stage: string;
  progress: number | null;
  onCancel?: () => void;
}) {
  const label = STAGE_TEXT[stage] ?? stage;
  const percent = progress === null ? null : Math.round(progress * 100);
  return (
    <section className="card progress-panel" aria-live="polite">
      <p className="progress-stage">
        {label}
        {percent !== null && ` · ${percent}%`}
      </p>
      <div
        className={`progress${percent === null ? " indeterminate" : ""}`}
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
      >
        <div className="progress-bar" style={{ width: percent === null ? undefined : `${percent}%` }} />
      </div>
      {onCancel && (
        <button type="button" className="link-button" onClick={onCancel}>
          Cancel
        </button>
      )}
    </section>
  );
}
