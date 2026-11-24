export interface VoiceProfile {
  id: string;
  name: string;
  language_tag: "en" | "hi" | "bilingual";
  reference_audio_path: string;
  duration_sec: number;
  sample_rate: number;
  consent_confirmed: number;
  consent_note: string;
  created_at: string;
  updated_at: string;
  audio_exists?: boolean;
}

export interface GenerationJob {
  id: string;
  voice_profile_id: string | null;
  voice_name_snapshot: string;
  text: string;
  language: "en" | "hi";
  engine_id: string;
  device_used: "auto" | "cuda" | "cpu";
  precision_used?: string;
  exaggeration: number;
  cfg_weight: number;
  temperature: number;
  status: "queued" | "running" | "completed" | "failed";
  progress_pct: number;
  status_message: string;
  error_message: string | null;
  output_audio_path: string | null;
  audio_duration_sec: number | null;
  elapsed_sec: number | null;
  created_at: string;
  updated_at: string;
  audio_exists?: boolean;
}

export interface AppSettings {
  id: number;
  preferred_device: "auto" | "cuda" | "cpu";
  chunk_max_chars: number;
  updated_at: string;
}

export interface DiskStats {
  total_gb: number;
  used_gb: number;
  free_gb: number;
}

export interface HealthStatus {
  status: string;
  engine_mode: string;
  model: {
    repo_id: string;
    revision: string;
    t3_model: string;
    precision_policy: string;
    cached_offline: boolean;
    loaded_device: string | null;
  };
  gpu: {
    cuda_available: boolean;
    gpu_name: string | null;
    vram_total_mb: number | null;
  };
  memory: {
    ram_total_mb?: number;
    ram_available_mb?: number;
    swap_total_mb?: number;
    swap_free_mb?: number;
  };
  storage: {
    e_drive: DiskStats | null;
    c_drive: DiskStats | null;
    swap_vhdx_exists: boolean;
    swap_vhdx_mb: number;
    db_path: string;
  };
  counts: {
    voices: number;
    generations: number;
    queued: number;
  };
}

export interface BenchClipMetrics {
  gen_sec: number;
  audio_sec: number;
  rtf: number | null;
  peak_alloc_mb?: number;
  peak_reserved_mb?: number;
  nvidia_smi_used_mb?: number;
}

export interface BenchReport {
  mode: string;
  device: string;
  precision: string;
  load_sec: number;
  peak_rss_overall_mb: number;
  runs: Record<string, BenchClipMetrics>;
}

export interface FeasibilityResponse {
  cpu_fp32: BenchReport | null;
  cuda_fp32: BenchReport | null;
  human_reference: {
    expected_path: string;
    present: boolean;
    duration_sec: number | null;
  };
  synthesized_warning: string;
  clips: Array<{
    filename: string;
    duration_sec: number | null;
    url: string;
  }>;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // keep fallback message
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getHealth: () => requestJson<HealthStatus>("/api/health"),
  getFeasibility: () => requestJson<FeasibilityResponse>("/api/feasibility"),
  getSettings: () => requestJson<AppSettings>("/api/settings"),
  updateSettings: (payload: Pick<AppSettings, "preferred_device" | "chunk_max_chars">) =>
    requestJson<AppSettings>("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  listVoices: () => requestJson<VoiceProfile[]>("/api/voices"),
  createVoice: (form: FormData) =>
    requestJson<VoiceProfile>("/api/voices", {
      method: "POST",
      body: form,
    }),
  deleteVoice: (id: string) =>
    requestJson<{ ok: boolean }>(`/api/voices/${id}`, { method: "DELETE" }),
  listGenerations: (q = "", language = "all") => {
    const params = new URLSearchParams();
    if (q.trim()) params.set("q", q.trim());
    if (language !== "all") params.set("language", language);
    const qs = params.toString();
    return requestJson<GenerationJob[]>(`/api/generations${qs ? `?${qs}` : ""}`);
  },
  createGeneration: (payload: {
    voice_profile_id: string;
    text: string;
    language: "en" | "hi";
    device?: "auto" | "cuda" | "cpu";
    exaggeration: number;
    cfg_weight: number;
    temperature: number;
  }) =>
    requestJson<GenerationJob>("/api/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  cancelGeneration: (id: string) =>
    requestJson<{ ok: boolean; status: string }>(`/api/generations/${id}/cancel`, { method: "POST" }),
  deleteGeneration: (id: string) =>
    requestJson<{ ok: boolean }>(`/api/generations/${id}`, { method: "DELETE" }),
};

