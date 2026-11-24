import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Sparkles, RotateCcw, SlidersHorizontal, ArrowRight, AlertTriangle } from "lucide-react";
import { api, type AppSettings, type GenerationJob, type VoiceProfile } from "../lib/api";
import { AudioPlayer } from "../components/AudioPlayer";
import { ProgressBanner } from "../components/ProgressBanner";

const DEVANAGARI_RE = /[\u0900-\u097F]/;
const ROMAN_HINDI_RE =
  /\b(upaay|kundli|aapke|aapki|aapko|apni|apna|nahi|hote|hain|karein|karen|banaiye|sirf|lekin|dekha|jata|batati|liye|sab|koi|yeh|isme|milta|der|mat|kijiye|namaste|kya|kaise|aur)\b/i;

const STYLE_PRESETS = [
  { id: "voice_match", label: "Closer Voice Match (Default)", exaggeration: 0.65, cfgWeight: 0.5, temperature: 0.75 },
  { id: "expressive", label: "Extra Expressive", exaggeration: 0.75, cfgWeight: 0.4, temperature: 0.8 },
  { id: "calm", label: "Calm & Steady", exaggeration: 0.45, cfgWeight: 0.55, temperature: 0.7 },
] as const;

const SAMPLE_PROMPTS: Record<"en" | "hi", string[]> = {
  en: [
    "Hello, this is a local voice cloning test running offline on my computer.",
    "Welcome to SwaraLab. Every voice sample and generated clip stays locally on my drive.",
  ],
  hi: [
    "नमस्ते, यह मेरे कंप्यूटर पर स्थानीय आवाज़ क्लोनिंग का परीक्षण है।",
    "स्वरा-लैब में आपका स्वागत है। आपकी आवाज़ और ऑडियो पूरी तरह आपके कंप्यूटर पर सुरक्षित रहते हैं।",
  ],
};

export function StudioPage() {
  const [searchParams] = useSearchParams();
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [voiceId, setVoiceId] = useState<string>("");
  const [language, setLanguage] = useState<"en" | "hi">("en");
  const [text, setText] = useState<string>("");
  const [deviceOverride, setDeviceOverride] = useState<"cpu" | "cuda" | "auto">("auto");
  const [exaggeration, setExaggeration] = useState<number>(0.65);
  const [cfgWeight, setCfgWeight] = useState<number>(0.5);
  const [temperature, setTemperature] = useState<number>(0.75);

  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<GenerationJob | null>(null);

  useEffect(() => {
    let mounted = true;
    Promise.all([api.listVoices(), api.getSettings()])
      .then(([vList, s]) => {
        if (!mounted) return;
        setVoices(vList);
        setSettings(s);
        setDeviceOverride(s.preferred_device === "cpu" ? "auto" : s.preferred_device);
        const requestedVoice = searchParams.get("voice");
        if (requestedVoice && vList.some((v) => v.id === requestedVoice)) {
          setVoiceId(requestedVoice);
        } else if (vList.length > 0) {
          setVoiceId(vList[0].id);
        }
      })
      .catch((err: Error) => {
        if (mounted) setLoadError(err.message);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [searchParams]);

  useEffect(() => {
    if (!activeJob || activeJob.status === "completed" || activeJob.status === "failed") {
      return;
    }
    const jobId = activeJob.id;
    const es = new EventSource(`/api/generations/${jobId}/events`);
    es.onmessage = (ev) => {
      try {
        const updated = JSON.parse(ev.data) as GenerationJob & { error?: string };
        if (updated.error && !updated.id) {
          es.close();
          return;
        }
        setActiveJob(updated);
        if (updated.status === "completed" || updated.status === "failed") {
          es.close();
        }
      } catch {
        // ignore malformed keepalive
      }
    };
    es.onerror = () => {
      es.close();
    };
    const poll = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/generations/${jobId}`);
        if (res.ok) {
          const latest = (await res.json()) as GenerationJob;
          setActiveJob(latest);
        }
      } catch {
        // ignore transient poll error
      }
    }, 1500);
    return () => {
      es.close();
      window.clearInterval(poll);
    };
  }, [activeJob?.id, activeJob?.status]);

  const chunkLimit = settings?.chunk_max_chars ?? 320;
  const hasDevanagari = DEVANAGARI_RE.test(text);
  const hasRomanHindi = ROMAN_HINDI_RE.test(text);
  const isBusy = submitting || (activeJob != null && (activeJob.status === "queued" || activeJob.status === "running"));

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);
    if (!voiceId) {
      setFormError("Select or create a saved voice profile first.");
      return;
    }
    if (!text.trim()) {
      setFormError("Enter English or Hindi text to synthesize.");
      return;
    }
    setSubmitting(true);
    try {
      const created = await api.createGeneration({
        voice_profile_id: voiceId,
        text: text.trim(),
        language,
        device: deviceOverride,
        exaggeration,
        cfg_weight: cfgWeight,
        temperature,
      });
      setActiveJob(created);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Could not start generation.");
    } finally {
      setSubmitting(false);
    }
  }

  function resetTuning() {
    setExaggeration(0.65);
    setCfgWeight(0.5);
    setTemperature(0.75);
  }

  async function handleCancelActive() {
    if (!activeJob) return;
    try {
      await api.cancelGeneration(activeJob.id);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Could not cancel job.");
    }
  }

  if (loading) {
    return (
      <div className="stack" aria-busy="true">
        <div className="skeleton" />
        <div className="skeleton" style={{ height: 220 }} />
      </div>
    );
  }

  return (
    <div className="stack">
      <div className="page-header">
        <div>
          <h1 className="page-title">Speech Studio</h1>
          <p className="page-subtitle">
            Generate offline English and Hindi speech using a saved authorized voice profile and pinned Chatterbox V2 (float32).
          </p>
        </div>
        <Link to="/voices" className="btn btn-secondary">
          Manage Voice Profiles
          <ArrowRight size={15} aria-hidden="true" />
        </Link>
      </div>

      {loadError && (
        <div className="alert alert-danger" role="alert">
          {loadError}
        </div>
      )}

      {voices.length === 0 ? (
        <div className="card empty-state stack">
          <h2 className="card-title">No voice profiles saved yet</h2>
          <p className="page-subtitle">
            Upload an authorized 6–12 second voice recording on the Voice Profiles page before generating speech.
          </p>
          <div>
            <Link to="/voices" className="btn btn-primary">
              Add Authorized Voice Profile
            </Link>
          </div>
        </div>
      ) : (
        <div className="grid-2">
          <form className="card" onSubmit={handleGenerate} noValidate>
            <h2 className="card-title">Synthesis Prompt</h2>

            <div className="field-group">
              <label htmlFor="studio-voice" className="field-label">
                Voice profile
              </label>
              <select
                id="studio-voice"
                className="select"
                value={voiceId}
                onChange={(e) => setVoiceId(e.target.value)}
                disabled={isBusy}
              >
                {voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} ({v.language_tag}, {v.duration_sec.toFixed(1)}s)
                  </option>
                ))}
              </select>
            </div>

            <div className="field-group">
              <span className="field-label" id="lang-label">
                Output language
              </span>
              <div className="segmented" role="group" aria-labelledby="lang-label">
                <button
                  type="button"
                  className="segmented-btn"
                  aria-pressed={language === "en"}
                  onClick={() => setLanguage("en")}
                  disabled={isBusy}
                >
                  English (en)
                </button>
                <button
                  type="button"
                  className="segmented-btn"
                  aria-pressed={language === "hi"}
                  onClick={() => setLanguage("hi")}
                  disabled={isBusy}
                >
                  Hindi Devanagari (hi)
                </button>
              </div>
            </div>

            <div className="field-group">
              <div className="row-between">
                <label htmlFor="studio-text" className="field-label">
                  Script text
                </label>
                <span className="field-hint mono">
                  {text.length} / 3000 chars {text.length > chunkLimit ? "· auto-chunked" : ""}
                </span>
              </div>
              <textarea
                id="studio-text"
                className="textarea"
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={isBusy}
                maxLength={3000}
                placeholder={
                  language === "hi"
                    ? "यहाँ हिंदी देवनागरी में वाक्य लिखें..."
                    : "Enter English text to synthesize with the selected voice..."
                }
              />
              <div className="row-gap">
                <span className="field-hint">Quick sample:</span>
                {SAMPLE_PROMPTS[language].map((sample, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setText(sample)}
                    disabled={isBusy}
                  >
                    Sample {idx + 1}
                  </button>
                ))}
              </div>
              {hasDevanagari && language === "en" && (
                <div className="alert alert-warning row-between" role="status">
                  <span className="row-gap">
                    <AlertTriangle size={16} aria-hidden="true" />
                    <span>Devanagari Hindi script detected while English (en) is active. Switch to Hindi for accurate pronunciation.</span>
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setLanguage("hi")}
                    disabled={isBusy}
                  >
                    Switch to Hindi (hi)
                  </button>
                </div>
              )}
              {hasRomanHindi && (
                <div className="alert row-gap" role="status">
                  <AlertTriangle size={16} aria-hidden="true" />
                  <span className="field-hint" style={{ color: "var(--color-text)" }}>
                    Romanized Hindi words detected (such as upaay or kundli). Write Hindi words in Devanagari (उपाय, कुंडली) and keep genuine English words in English for accurate pronunciation.
                  </span>
                </div>
              )}
            </div>

            <div className="field-group">
              <label htmlFor="studio-device" className="field-label">
                Compute device (float32 only; fp16 excluded)
              </label>
              <select
                id="studio-device"
                className="select"
                value={deviceOverride}
                onChange={(e) => setDeviceOverride(e.target.value as "cpu" | "cuda" | "auto")}
                disabled={isBusy}
              >
                <option value="auto">Auto · Sequential GPU CUDA float32 (~2.5× faster, ≤2.3 GB VRAM)</option>
                <option value="cuda">GPU CUDA float32 (Sequential T3 + S3Gen)</option>
                <option value="cpu">CPU float32 (8-core fallback · slower)</option>
              </select>
            </div>

            <details className="card-muted" style={{ marginBottom: "var(--space-4)" }}>
              <summary className="row-gap" style={{ cursor: "pointer", fontWeight: 600 }}>
                <SlidersHorizontal size={15} aria-hidden="true" />
                Supported engine controls
              </summary>
              <div className="stack" style={{ marginTop: "var(--space-3)" }}>
                <p className="field-hint" style={{ margin: 0 }}>
                  Only controls exposed by ChatterboxMultilingualTTS are shown (speed, pitch, and seed are unsupported by this model). Higher exaggeration and lower CFG weight improve vocal expression.
                </p>

                <div className="row-gap">
                  <span className="field-hint">Delivery preset:</span>
                  {STYLE_PRESETS.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        setExaggeration(preset.exaggeration);
                        setCfgWeight(preset.cfgWeight);
                        setTemperature(preset.temperature);
                      }}
                      disabled={isBusy}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>

                <div>
                  <div className="row-between">
                    <label htmlFor="ctrl-exag" className="field-label">
                      Exaggeration
                    </label>
                    <span className="mono">{exaggeration.toFixed(2)}</span>
                  </div>
                  <input
                    id="ctrl-exag"
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={exaggeration}
                    onChange={(e) => setExaggeration(Number(e.target.value))}
                    style={{ width: "100%" }}
                    disabled={isBusy}
                  />
                </div>

                <div>
                  <div className="row-between">
                    <label htmlFor="ctrl-cfg" className="field-label">
                      CFG Weight
                    </label>
                    <span className="mono">{cfgWeight.toFixed(2)}</span>
                  </div>
                  <input
                    id="ctrl-cfg"
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={cfgWeight}
                    onChange={(e) => setCfgWeight(Number(e.target.value))}
                    style={{ width: "100%" }}
                    disabled={isBusy}
                  />
                </div>

                <div>
                  <div className="row-between">
                    <label htmlFor="ctrl-temp" className="field-label">
                      Temperature
                    </label>
                    <span className="mono">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    id="ctrl-temp"
                    type="range"
                    min="0.2"
                    max="1.2"
                    step="0.05"
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    style={{ width: "100%" }}
                    disabled={isBusy}
                  />
                </div>

                <div>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={resetTuning}
                    disabled={isBusy}
                  >
                    <RotateCcw size={13} aria-hidden="true" />
                    Reset defaults
                  </button>
                </div>
              </div>
            </details>

            {formError && (
              <div className="alert alert-danger" role="alert" style={{ marginBottom: "var(--space-4)" }}>
                {formError}
              </div>
            )}

            <div className="row-gap">
              <button type="submit" className="btn btn-primary btn-lg" disabled={isBusy} aria-busy={isBusy}>
                <Sparkles size={16} aria-hidden="true" />
                {isBusy ? "Generating speech..." : "Generate Speech"}
              </button>
              {activeJob && (activeJob.status === "queued" || activeJob.status === "running") && (
                <button type="button" className="btn btn-secondary btn-lg" onClick={handleCancelActive}>
                  Cancel Job
                </button>
              )}
            </div>
          </form>

          <div className="stack">
            <div className="card stack">
              <div className="row-between">
                <h2 className="card-title" style={{ margin: 0 }}>
                  Output & Status
                </h2>
                <span className="badge">V2 · t3_mtl23ls_v2</span>
              </div>

              {!activeJob ? (
                <div className="empty-state">
                  <p style={{ margin: 0 }}>
                    Choose a profile, enter English or Hindi text, and click <strong>Generate Speech</strong> to play and download a WAV file.
                  </p>
                </div>
              ) : (
                <div className="stack">
                  <ProgressBanner job={activeJob} />
                  {activeJob.status === "completed" && activeJob.audio_exists && (
                    <AudioPlayer
                      src={`/api/generations/${activeJob.id}/audio`}
                      downloadHref={`/api/generations/${activeJob.id}/audio?download=1`}
                      title={`${activeJob.voice_name_snapshot} (${activeJob.language.toUpperCase()})`}
                      durationSec={activeJob.audio_duration_sec}
                    />
                  )}
                  <div className="row-between">
                    <span className="field-hint mono">Job {activeJob.id.slice(0, 8)}</span>
                    <Link to="/history" className="btn btn-secondary btn-sm">
                      View in History
                    </Link>
                  </div>
                </div>
              )}
            </div>

            <div className="card-muted stack">
              <strong style={{ fontSize: 14 }}>Voice Matching &amp; Speed Tips</strong>
              <p className="field-hint" style={{ margin: 0 }}>
                Sequential GPU CUDA (float32) runs T3 and S3Gen one stage at a time under ~2.3 GB VRAM (~2.5× faster than CPU). Chatterbox only reads the <strong>first 6–10 seconds</strong> of your reference clip for voice timbre/prosody—so a clean 8–12s sample with zero background noise/music and higher CFG Weight (0.50) matches the speaker much better than a long 30s+ clip.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
