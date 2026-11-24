import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import type { GenerationJob } from "../lib/api";

export function ProgressBanner({ job }: { job: GenerationJob | null }) {
  if (!job) return null;

  if (job.status === "failed") {
    return (
      <div className="alert alert-danger stack" role="alert">
        <div className="row-gap">
          <AlertCircle size={18} aria-hidden="true" />
          <strong>Synthesis failed</strong>
        </div>
        <span>{job.error_message || "An unexpected error stopped speech synthesis."}</span>
      </div>
    );
  }

  if (job.status === "completed") {
    return (
      <div className="alert alert-success row-between" role="status" aria-live="polite">
        <span className="row-gap">
          <CheckCircle2 size={18} aria-hidden="true" />
          <strong>{job.status_message || "Speech ready"}</strong>
        </span>
        <span className="mono">
          {job.device_used.toUpperCase()} · {job.precision_used || "float32"}
          {job.elapsed_sec != null ? ` · ${job.elapsed_sec.toFixed(1)}s` : ""}
        </span>
      </div>
    );
  }

  return (
    <div className="card-muted stack" role="status" aria-live="polite">
      <div className="row-between">
        <span className="row-gap">
          <Loader2 size={16} aria-hidden="true" />
          <strong>{job.status_message || "Preparing synthesis..."}</strong>
        </span>
        <span className="mono">{job.progress_pct}%</span>
      </div>
      <div className="progress-track" aria-label="Generation progress">
        <div className="progress-fill" style={{ width: `${job.progress_pct}%` }} />
      </div>
      <span className="field-hint">
        First run loads the pinned V2 weights (~24s). Short clips typically take ~30–40s on CPU float32 after load.
      </span>
    </div>
  );
}

