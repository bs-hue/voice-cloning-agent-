import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, Trash2, ShieldCheck, Mic } from "lucide-react";
import { api, type VoiceProfile } from "../lib/api";
import { AudioPlayer } from "../components/AudioPlayer";

export function VoicesPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [languageTag, setLanguageTag] = useState<"bilingual" | "en" | "hi">("bilingual");
  const [consentChecked, setConsentChecked] = useState(false);
  const [consentNote, setConsentNote] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragover, setDragover] = useState(false);

  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  async function refreshVoices() {
    try {
      const list = await api.listVoices();
      setVoices(list);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load voice profiles.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshVoices();
  }, []);

  function onFilePicked(selected: File | null) {
    setFormError(null);
    setFile(selected);
    if (selected && !name.trim()) {
      const stem = selected.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
      if (stem.length >= 2) {
        setName(stem.slice(0, 60));
      }
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);
    if (name.trim().length < 2) {
      setFormError("Enter a profile name (at least 2 characters).");
      return;
    }
    if (!file) {
      setFormError("Choose a clean reference audio recording (.wav recommended, 6–12 seconds).");
      return;
    }
    if (!consentChecked) {
      setFormError("Confirm speaker authorization before saving a voice profile.");
      return;
    }

    const fd = new FormData();
    fd.append("name", name.trim());
    fd.append("language_tag", languageTag);
    fd.append("consent_confirmed", "true");
    fd.append("consent_note", consentNote.trim());
    fd.append("audio", file);

    setSaving(true);
    try {
      await api.createVoice(fd);
      setName("");
      setConsentChecked(false);
      setConsentNote("");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refreshVoices();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Failed to save voice profile.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    try {
      await api.deleteVoice(id);
      setConfirmDeleteId(null);
      await refreshVoices();
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Could not delete voice profile.");
    }
  }

  return (
    <div className="stack">
      <div className="page-header">
        <div>
          <h1 className="page-title">Voice Profiles</h1>
          <p className="page-subtitle">
            Upload an authorized 6–12 second clean speech recording. Audio is normalized locally on E: to 24 kHz mono PCM WAV.
          </p>
        </div>
        <span className="badge badge-success">
          <ShieldCheck size={14} aria-hidden="true" />
          Authorization Required
        </span>
      </div>

      {loadError && (
        <div className="alert alert-danger" role="alert">
          {loadError}
        </div>
      )}

      <div className="grid-2">
        <form className="card" onSubmit={handleSubmit} noValidate>
          <h2 className="card-title">Add Authorized Voice</h2>

          <div className="field-group">
            <span className="field-label">Reference audio file</span>
            <div
              className={`dropzone${dragover ? " dragover" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragover(true);
              }}
              onDragLeave={() => setDragover(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragover(false);
                onFilePicked(e.dataTransfer.files?.[0] ?? null);
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
            >
              <Upload size={22} aria-hidden="true" style={{ marginBottom: "var(--space-2)" }} />
              <div style={{ fontWeight: 600 }}>
                {file ? file.name : "Click or drop a voice recording"}
              </div>
              <div className="field-hint">
                Accepted: WAV, MP3, FLAC, OGG, M4A (min 3s; 6–12s clean WAV recommended — Chatterbox V2 uses up to 6s prompt / 10s decoder conditioning)
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.flac,.ogg,.m4a,.webm,audio/*"
                className="sr-only"
                aria-label="Choose reference audio file"
                onChange={(e) => onFilePicked(e.target.files?.[0] ?? null)}
              />
            </div>
          </div>

          <div className="field-group">
            <label htmlFor="voice-name" className="field-label">
              Profile name
            </label>
            <input
              id="voice-name"
              type="text"
              className="input"
              maxLength={80}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., My Studio Voice"
              disabled={saving}
            />
          </div>

          <div className="field-group">
            <label htmlFor="voice-lang-tag" className="field-label">
              Primary language tag
            </label>
            <select
              id="voice-lang-tag"
              className="select"
              value={languageTag}
              onChange={(e) => setLanguageTag(e.target.value as "bilingual" | "en" | "hi")}
              disabled={saving}
            >
              <option value="bilingual">Bilingual (English + Hindi)</option>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
            </select>
          </div>

          <div className="field-group">
            <label className="row-gap" style={{ alignItems: "flex-start", cursor: "pointer" }}>
              <input
                id="voice-consent"
                type="checkbox"
                checked={consentChecked}
                onChange={(e) => setConsentChecked(e.target.checked)}
                disabled={saving}
                style={{ marginTop: 4 }}
              />
              <span className="field-label" style={{ fontWeight: 500 }}>
                I confirm this recording is my own voice or I have explicit permission from the speaker to clone it locally.
              </span>
            </label>
          </div>

          <div className="field-group">
            <label htmlFor="voice-consent-note" className="field-label">
              Consent note (optional)
            </label>
            <input
              id="voice-consent-note"
              type="text"
              className="input"
              maxLength={240}
              value={consentNote}
              onChange={(e) => setConsentNote(e.target.value)}
              placeholder="e.g., Recorded personally on laptop mic, 2026-11-11"
              disabled={saving}
            />
          </div>

          {formError && (
            <div className="alert alert-danger" role="alert" style={{ marginBottom: "var(--space-4)" }}>
              {formError}
            </div>
          )}

          <button type="submit" className="btn btn-primary" disabled={saving} aria-busy={saving}>
            <Mic size={16} aria-hidden="true" />
            {saving ? "Validating & saving..." : "Save Voice Profile"}
          </button>
        </form>

        <div className="stack">
          <h2 className="card-title" style={{ margin: 0 }}>
            Saved Profiles ({voices.length})
          </h2>

          {loading ? (
            <div className="stack" aria-busy="true">
              <div className="skeleton" />
              <div className="skeleton" />
            </div>
          ) : voices.length === 0 ? (
            <div className="card empty-state">
              No voice profiles saved yet. Add your authorized recording on the left to start cloning locally.
            </div>
          ) : (
            voices.map((v) => (
              <article key={v.id} className="card stack">
                <div className="row-between">
                  <div style={{ minWidth: 0 }}>
                    <h3 style={{ margin: 0, fontSize: 16, overflowWrap: "anywhere" }}>{v.name}</h3>
                    <span className="field-hint">
                      {v.language_tag} · {v.duration_sec.toFixed(2)}s · {v.sample_rate} Hz
                    </span>
                  </div>
                  <div className="row-gap">
                    <span className="badge badge-success">Authorized</span>
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      onClick={() => navigate(`/?voice=${encodeURIComponent(v.id)}`)}
                    >
                      Use in Studio
                    </button>
                  </div>
                </div>

                {v.consent_note && (
                  <p className="field-hint" style={{ margin: 0 }}>
                    Note: {v.consent_note}
                  </p>
                )}

                <AudioPlayer
                  src={`/api/voices/${v.id}/audio`}
                  title="Normalized reference clip"
                  durationSec={v.duration_sec}
                  missingFile={!v.audio_exists}
                />

                <div className="row-between">
                  <span className="field-hint mono">
                    Added {new Date(v.created_at).toLocaleString()}
                  </span>
                  {confirmDeleteId === v.id ? (
                    <div className="row-gap">
                      <button
                        type="button"
                        className="btn btn-danger btn-sm"
                        onClick={() => handleDelete(v.id)}
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
                      onClick={() => setConfirmDeleteId(v.id)}
                    >
                      <Trash2 size={14} aria-hidden="true" />
                      Delete
                    </button>
                  )}
                </div>
              </article>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
