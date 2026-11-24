import { useEffect, useState } from "react";
import { Search, Trash2, RefreshCw } from "lucide-react";
import { api, type GenerationJob } from "../lib/api";
import { AudioPlayer } from "../components/AudioPlayer";

export function HistoryPage() {
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState<"all" | "en" | "hi">("all");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  async function loadJobs(q = query, lang = language) {
    setLoading(true);
    setError(null);
    try {
      const list = await api.listGenerations(q, lang);
      setJobs(list);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const timer = setTimeout(() => {
      loadJobs(query, language);
    }, 150);
    return () => clearTimeout(timer);
  }, [query, language]);

  async function handleDelete(id: string) {
    try {
      await api.deleteGeneration(id);
      setConfirmDeleteId(null);
      await loadJobs(query, language);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete entry.");
    }
  }

  return (
    <div className="stack">
      <div className="page-header">
        <div>
          <h1 className="page-title">Generation History</h1>
          <p className="page-subtitle">
            Search, replay, export WAV files, or remove past local speech generations stored in SQLite on E:.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => loadJobs(query, language)}
        >
          <RefreshCw size={15} aria-hidden="true" />
          Refresh
        </button>
      </div>

      <div className="card row-between">
        <div style={{ flex: "1 1 260px", position: "relative" }}>
          <label htmlFor="history-search" className="sr-only">
            Search prompt or voice name
          </label>
          <Search
            size={16}
            aria-hidden="true"
            style={{
              position: "absolute",
              left: 12,
              top: "50%",
              transform: "translateY(-50%)",
              color: "var(--color-text-muted)",
            }}
          />
          <input
            id="history-search"
            type="search"
            className="input"
            style={{ paddingLeft: 36 }}
            placeholder="Search by text or voice name..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        <div className="segmented" role="group" aria-label="Filter by language">
          {(["all", "en", "hi"] as const).map((lang) => (
            <button
              key={lang}
              type="button"
              className="segmented-btn"
              aria-pressed={language === lang}
              onClick={() => setLanguage(lang)}
            >
              {lang === "all" ? "All languages" : lang === "en" ? "English" : "Hindi"}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      {loading ? (
        <div className="stack" aria-busy="true">
          <div className="skeleton" />
          <div className="skeleton" />
        </div>
      ) : jobs.length === 0 ? (
        <div className="card empty-state">
          {query || language !== "all"
            ? "No generated clips match your search or language filter."
            : "No generated audio yet. Create your first clip from the Studio tab."}
        </div>
      ) : (
        <div className="stack">
          {jobs.map((job) => (
            <article key={job.id} className="card stack">
              <div className="row-between">
                <div className="row-gap">
                  <strong>{job.voice_name_snapshot}</strong>
                  <span className="badge">{job.language.toUpperCase()}</span>
                  <span
                    className={`badge ${
                      job.status === "completed"
                        ? "badge-success"
                        : job.status === "failed"
                        ? "badge-danger"
                        : "badge-accent"
                    }`}
                  >
                    {job.status}
                  </span>
                  <span className="badge">
                    {job.device_used.toUpperCase()} · {job.precision_used || "float32"}
                  </span>
                </div>
                <span className="field-hint mono">
                  {new Date(job.created_at).toLocaleString()}
                  {job.elapsed_sec != null ? ` · took ${job.elapsed_sec.toFixed(1)}s` : ""}
                </span>
              </div>

              <p style={{ margin: 0, overflowWrap: "anywhere" }}>{job.text}</p>

              {job.status === "failed" && job.error_message && (
                <div className="alert alert-danger">{job.error_message}</div>
              )}

              {job.status === "completed" && (
                <AudioPlayer
                  src={`/api/generations/${job.id}/audio`}
                  downloadHref={`/api/generations/${job.id}/audio?download=1`}
                  title={`Generated clip (${job.language.toUpperCase()})`}
                  durationSec={job.audio_duration_sec}
                  missingFile={!job.audio_exists}
                />
              )}

              <div className="row-between">
                <span className="field-hint mono">
                  Params: exag {job.exaggeration.toFixed(2)} · cfg {job.cfg_weight.toFixed(2)} · temp{" "}
                  {job.temperature.toFixed(2)}
                </span>

                {confirmDeleteId === job.id ? (
                  <div className="row-gap">
                    <button
                      type="button"
                      className="btn btn-danger btn-sm"
                      onClick={() => handleDelete(job.id)}
                    >
                      Confirm delete
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => setConfirmDeleteId(null)}
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setConfirmDeleteId(job.id)}
                  >
                    <Trash2 size={14} aria-hidden="true" />
                    Delete
                  </button>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
