import { Download, AlertTriangle } from "lucide-react";

interface AudioPlayerProps {
  src: string;
  title: string;
  durationSec?: number | null;
  downloadHref?: string;
  missingFile?: boolean;
}

export function AudioPlayer({
  src,
  title,
  durationSec,
  downloadHref,
  missingFile,
}: AudioPlayerProps) {
  if (missingFile) {
    return (
      <div className="alert alert-warning row-gap" role="status">
        <AlertTriangle size={16} aria-hidden="true" />
        <span>Audio file is missing from local storage on disk.</span>
      </div>
    );
  }

  return (
    <div className="audio-box">
      <div className="row-between">
        <span className="field-hint">
          {title}
          {durationSec != null ? ` · ${durationSec.toFixed(2)}s WAV` : ""}
        </span>
        {downloadHref && (
          <a
            href={downloadHref}
            className="btn btn-secondary btn-sm"
            download
          >
            <Download size={14} aria-hidden="true" />
            Download WAV
          </a>
        )}
      </div>
      <audio controls preload="metadata" src={src} aria-label={title} />
    </div>
  );
}

