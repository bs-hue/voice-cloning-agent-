import { useEffect, useState } from "react";
import { Cpu, HardDrive, CheckCircle2, AlertTriangle, RefreshCw, Save } from "lucide-react";
import {
  api,
  type AppSettings,
  type FeasibilityResponse,
  type HealthStatus,
} from "../lib/api";

export function SetupPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [feasibility, setFeasibility] = useState<FeasibilityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [preferredDevice, setPreferredDevice] = useState<"cpu" | "cuda" | "auto">("cpu");
  const [chunkMaxChars, setChunkMaxChars] = useState<number>(320);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  async function refreshAll() {
    setLoading(true);
    setError(null);
    try {
      const [h, s, f] = await Promise.all([
        api.getHealth(),
        api.getSettings(),
        api.getFeasibility(),
      ]);
      setHealth(h);
      setSettings(s);
      setFeasibility(f);
      setPreferredDevice(s.preferred_device);
      setChunkMaxChars(s.chunk_max_chars);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load status.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshAll();
  }, []);

  async function handleSaveSettings(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setSavedMsg(null);
    setError(null);
    try {
      const updated = await api.updateSettings({
        preferred_device: preferredDevice,
        chunk_max_chars: chunkMaxChars,
      });
      setSettings(updated);
      setSavedMsg("Settings saved locally to SQLite.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update settings.");
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="stack" aria-busy="true">
        <div className="skeleton" />
        <div className="skeleton" style={{ height: 180 }} />
      </div>
    );
  }

  const cpuRuns = feasibility?.cpu_fp32?.runs ?? {};
  const cudaRuns = feasibility?.cuda_fp32?.runs ?? {};

  return (
    <div className="stack">
      <div className="page-header">
        <div>
          <h1 className="page-title">Engine & Storage Dashboard</h1>
          <p className="page-subtitle">
            Live WSL2 memory, E: / C: drive capacity, pinned V2 model status, and measured Stage 1 benchmark telemetry.
          </p>
        </div>
        <button type="button" className="btn btn-secondary" onClick={refreshAll}>
          <RefreshCw size={15} aria-hidden="true" />
          Refresh Telemetry
        </button>
      </div>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      {health && (
        <div className="grid-stats">
          <div className="card stack" style={{ gap: "var(--space-2)" }}>
            <div className="row-between">
              <span className="field-label">Pinned Offline Model</span>
              <span className={`badge ${health.model.cached_offline ? "badge-success" : "badge-danger"}`}>
                {health.model.cached_offline ? "Cached Offline" : "Missing"}
              </span>
            </div>
            <strong className="mono">{health.model.t3_model}</strong>
            <span className="field-hint mono">rev {health.model.revision.slice(0, 12)} · {health.model.precision_policy}</span>
          </div>

          <div className="card stack" style={{ gap: "var(--space-2)" }}>
            <div className="row-between">
              <span className="field-label">Compute Hardware</span>
              <Cpu size={16} aria-hidden="true" />
            </div>
            <strong>{health.gpu.gpu_name || "CPU Mode"}</strong>
            <span className="field-hint">
              {health.gpu.cuda_available
                ? `CUDA available (${health.gpu.vram_total_mb} MB VRAM)`
                : "CUDA unavailable"}{" "}
              · Loaded: {health.model.loaded_device ?? "Not loaded yet"}
            </span>
          </div>

          <div className="card stack" style={{ gap: "var(--space-2)" }}>
            <div className="row-between">
              <span className="field-label">WSL2 Memory & Swap</span>
              <span className="badge badge-success">
                {health.storage.swap_vhdx_exists ? "Swap on E:" : "Swap idle"}
              </span>
            </div>
            <strong>
              {health.memory.ram_total_mb != null
                ? `${(health.memory.ram_total_mb / 1024).toFixed(1)} GB RAM cap`
                : "WSL RAM telemetry"}
            </strong>
            <span className="field-hint">
              {health.memory.ram_available_mb != null
                ? `${(health.memory.ram_available_mb / 1024).toFixed(1)} GB free · `
                : ""}
              swap.vhdx: {health.storage.swap_vhdx_mb.toFixed(1)} MB / 4096 MB max
            </span>
          </div>

          <div className="card stack" style={{ gap: "var(--space-2)" }}>
            <div className="row-between">
              <span className="field-label">Drive Space (E: / C:)</span>
              <HardDrive size={16} aria-hidden="true" />
            </div>
            <strong>
              E: {health.storage.e_drive ? `${health.storage.e_drive.free_gb} GB free` : "N/A"}
            </strong>
            <span className="field-hint">
              C: {health.storage.c_drive ? `${health.storage.c_drive.free_gb} GB free` : "N/A"} ·{" "}
              {health.counts.voices} profiles · {health.counts.generations} clips
            </span>
          </div>
        </div>
      )}

      <div className="grid-2">
        <form className="card stack" onSubmit={handleSaveSettings}>
          <h2 className="card-title" style={{ margin: 0 }}>
            Default Synthesis Settings
          </h2>

          <div className="field-group" style={{ marginBottom: 0 }}>
            <label htmlFor="pref-device" className="field-label">
              Default processing device
            </label>
            <select
              id="pref-device"
              className="select"
              value={preferredDevice}
              onChange={(e) => setPreferredDevice(e.target.value as "cpu" | "cuda" | "auto")}
            >
              <option value="cpu">CPU float32 (Recommended default · steady 29–37s/clip)</option>
              <option value="cuda">GPU CUDA float32 (Experimental · 4 GB VRAM ceiling)</option>
              <option value="auto">Auto (Resolves to CPU float32)</option>
            </select>
            <span className="field-hint">
              Forced fp16 is permanently excluded because Stage 1 audio output was severely distorted.
            </span>
          </div>

          <div className="field-group" style={{ marginBottom: 0 }}>
            <label htmlFor="chunk-chars" className="field-label">
              Sentence chunk limit (characters)
            </label>
            <input
              id="chunk-chars"
              type="number"
              min={60}
              max={400}
              className="input"
              value={chunkMaxChars}
              onChange={(e) => setChunkMaxChars(Number(e.target.value))}
            />
            <span className="field-hint">
              Prompts longer than this limit split cleanly on sentence boundaries. Current saved: {settings?.chunk_max_chars ?? 320}.
            </span>
          </div>

          {savedMsg && (
            <div className="alert alert-success row-gap" role="status">
              <CheckCircle2 size={16} aria-hidden="true" />
              <span>{savedMsg}</span>
            </div>
          )}

          <div>
            <button type="submit" className="btn btn-primary" disabled={saving}>
              <Save size={15} aria-hidden="true" />
              {saving ? "Saving..." : "Save Settings"}
            </button>
          </div>
        </form>

        <div className="card stack">
          <h2 className="card-title" style={{ margin: 0 }}>
            Human Reference Readiness
          </h2>
          {feasibility && (
            <>
              <div
                className={`alert row-gap ${
                  feasibility.human_reference.present ? "alert-success" : "alert-warning"
                }`}
              >
                {feasibility.human_reference.present ? (
                  <CheckCircle2 size={18} aria-hidden="true" />
                ) : (
                  <AlertTriangle size={18} aria-hidden="true" />
                )}
                <span>
                  {feasibility.human_reference.present
                    ? `Found ref_human.wav (${feasibility.human_reference.duration_sec ?? "?"}s) ready for voice-quality evaluation.`
                    : "ref_human.wav is not in data\\feasibility yet. You can still upload any authorized clip directly on the Voice Profiles tab."}
                </span>
              </div>
              <span className="field-hint mono" style={{ overflowWrap: "anywhere" }}>
                Expected path: {feasibility.human_reference.expected_path}
              </span>
              <p className="field-hint" style={{ margin: 0 }}>
                {feasibility.synthesized_warning}
              </p>
            </>
          )}
        </div>
      </div>

      {feasibility && (feasibility.cpu_fp32 || feasibility.cuda_fp32) && (
        <div className="card stack">
          <h2 className="card-title" style={{ margin: 0 }}>
            Stage 1 Measured Benchmark Summary (Offline t3_mtl23ls_v2.safetensors)
          </h2>

          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Mode / Precision</th>
                  <th>Model Load</th>
                  <th>Run 1 EN (gen / audio)</th>
                  <th>Run 1 HI (gen / audio)</th>
                  <th>Run 2 EN (gen / audio)</th>
                  <th>Run 2 HI (gen / audio)</th>
                  <th>Peak RAM / VRAM</th>
                </tr>
              </thead>
              <tbody>
                {feasibility.cpu_fp32 && (
                  <tr>
                    <td>
                      <strong>CPU float32</strong> (Default)
                    </td>
                    <td className="mono">{feasibility.cpu_fp32.load_sec}s</td>
                    <td className="mono">
                      {cpuRuns.run1_en?.gen_sec}s / {cpuRuns.run1_en?.audio_sec}s
                    </td>
                    <td className="mono">
                      {cpuRuns.run1_hi?.gen_sec}s / {cpuRuns.run1_hi?.audio_sec}s
                    </td>
                    <td className="mono">
                      {cpuRuns.run2_en?.gen_sec}s / {cpuRuns.run2_en?.audio_sec}s
                    </td>
                    <td className="mono">
                      {cpuRuns.run2_hi?.gen_sec}s / {cpuRuns.run2_hi?.audio_sec}s
                    </td>
                    <td className="mono">{feasibility.cpu_fp32.peak_rss_overall_mb} MB RSS</td>
                  </tr>
                )}
                {feasibility.cuda_fp32 && (
                  <tr>
                    <td>
                      <strong>GPU float32</strong> (Experimental)
                    </td>
                    <td className="mono">{feasibility.cuda_fp32.load_sec}s</td>
                    <td className="mono">
                      {cudaRuns.run1_en?.gen_sec}s / {cudaRuns.run1_en?.audio_sec}s
                    </td>
                    <td className="mono">
                      {cudaRuns.run1_hi?.gen_sec}s / {cudaRuns.run1_hi?.audio_sec}s
                    </td>
                    <td className="mono">
                      {cudaRuns.run2_en?.gen_sec}s / {cudaRuns.run2_en?.audio_sec}s
                    </td>
                    <td className="mono">
                      {cudaRuns.run2_hi?.gen_sec}s / {cudaRuns.run2_hi?.audio_sec}s
                    </td>
                    <td className="mono">
                      {cudaRuns.run2_hi?.nvidia_smi_used_mb ?? "?"} / 4095.5 MB VRAM
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {feasibility.clips.length > 0 && (
            <details className="card-muted">
              <summary style={{ cursor: "pointer", fontWeight: 600 }}>
                Play Stage 1 Benchmark Clips ({feasibility.clips.length} synthetic-reference WAVs)
              </summary>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                  gap: "var(--space-3)",
                  marginTop: "var(--space-3)",
                }}
              >
                {feasibility.clips.map((clip) => (
                  <div key={clip.filename} className="audio-box">
                    <span className="mono">
                      {clip.filename}
                      {clip.duration_sec != null ? ` (${clip.duration_sec}s)` : ""}
                    </span>
                    <audio controls preload="none" src={clip.url} aria-label={clip.filename} />
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
