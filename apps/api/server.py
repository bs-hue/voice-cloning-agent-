"""FastAPI local backend for SwaraLab (127.0.0.1 single-user studio + Stage 1 telemetry)."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apps.api.audio_utils import inspect_and_normalize_reference, read_wav_duration
from apps.api.db import DB_PATH, DATA_DIR, get_connection, init_db, row_to_dict, utc_now_iso
from apps.api.engine import (
    PINNED_HF_REVISION,
    PINNED_REPO_ID,
    PINNED_T3_MODEL,
    ChatterboxV2Engine,
    JobCancelledError,
)

ROOT = Path(__file__).resolve().parents[2]
VOICES_DIR = DATA_DIR / "voices"
OUTPUTS_DIR = DATA_DIR / "outputs"
FEASIBILITY_DIR = ROOT / "data" / "feasibility"
WEB_DIST_DIR = ROOT / "apps" / "web" / "dist"

VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

engine = ChatterboxV2Engine()
job_queue: queue.Queue[str] = queue.Queue()
_worker_stop = threading.Event()
_worker_thread: threading.Thread | None = None
_cancel_lock = threading.Lock()
_cancelled_jobs: set[str] = set()


def _is_cancelled(job_id: str) -> bool:
    with _cancel_lock:
        return job_id in _cancelled_jobs


def _mark_cancelled(job_id: str) -> None:
    with _cancel_lock:
        _cancelled_jobs.add(job_id)


def _clear_cancelled(job_id: str) -> None:
    with _cancel_lock:
        _cancelled_jobs.discard(job_id)


def _resolve_stored_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (ROOT / p)


def _to_rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now_iso()
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    conn = get_connection()
    try:
        conn.execute(f"UPDATE generation_jobs SET {cols} WHERE id = ?", vals)
        conn.commit()
    finally:
        conn.close()


def _worker_loop() -> None:
    while not _worker_stop.is_set():
        try:
            job_id = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            conn = get_connection()
            try:
                job = row_to_dict(
                    conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
                )
                settings = row_to_dict(conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone())
                profile = None
                if job and job["voice_profile_id"]:
                    profile = row_to_dict(
                        conn.execute(
                            "SELECT * FROM voice_profiles WHERE id = ?", (job["voice_profile_id"],)
                        ).fetchone()
                    )
            finally:
                conn.close()

            if not job or job["status"] != "queued":
                continue

            if not profile:
                _update_job(
                    job_id,
                    status="failed",
                    progress_pct=0,
                    status_message="Voice profile missing",
                    error_message="The selected voice profile was deleted before synthesis started.",
                )
                continue

            ref_path = _resolve_stored_path(profile["reference_audio_path"])
            if not ref_path.exists():
                _update_job(
                    job_id,
                    status="failed",
                    progress_pct=0,
                    status_message="Reference file missing",
                    error_message="Reference WAV file is missing from data/voices on disk.",
                )
                continue

            preferred_device = job["device_used"] or (settings["preferred_device"] if settings else "cpu")
            chunk_max_chars = int(settings["chunk_max_chars"]) if settings else 320
            out_path = OUTPUTS_DIR / f"{job_id}.wav"

            _update_job(
                job_id,
                status="running",
                progress_pct=8,
                status_message=f"Preparing synthesis ({preferred_device.upper()} float32)...",
            )

            started = time.perf_counter()

            def report_progress(pct: int, msg: str) -> None:
                _update_job(job_id, progress_pct=max(0, min(100, int(pct))), status_message=msg)

            res = engine.synthesize(
                text=job["text"],
                language=job["language"],
                ref_wav_path=ref_path,
                out_wav_path=out_path,
                preferred_device=preferred_device,
                chunk_max_chars=chunk_max_chars,
                exaggeration=float(job["exaggeration"]),
                cfg_weight=float(job["cfg_weight"]),
                temperature=float(job["temperature"]),
                on_progress=report_progress,
                should_cancel=lambda: _is_cancelled(job_id),
            )
            elapsed = round(time.perf_counter() - started, 2)
            _update_job(
                job_id,
                status="completed",
                progress_pct=100,
                device_used=res["device_used"],
                precision_used=res["precision_used"],
                output_audio_path=_to_rel_path(out_path),
                audio_duration_sec=res["audio_duration_sec"],
                elapsed_sec=elapsed,
                status_message=f"Ready ({res['audio_duration_sec']:.1f}s WAV in {elapsed:.1f}s)",
            )
        except JobCancelledError as exc:
            _update_job(
                job_id,
                status="failed",
                progress_pct=0,
                status_message="Cancelled",
                error_message=str(exc),
            )
        except Exception as exc:
            _update_job(
                job_id,
                status="failed",
                progress_pct=0,
                status_message="Synthesis failed",
                error_message=str(exc),
            )
        finally:
            _clear_cancelled(job_id)
            job_queue.task_done()


def _requeue_interrupted_jobs() -> None:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, status FROM generation_jobs WHERE status IN ('queued', 'running') ORDER BY created_at ASC"
        ).fetchall()
        for r in rows:
            if r["status"] == "running":
                conn.execute(
                    "UPDATE generation_jobs SET status = 'failed', error_message = ?, status_message = ? WHERE id = ?",
                    ("Server restarted before synthesis completed.", "Interrupted on restart", r["id"]),
                )
            else:
                job_queue.put(r["id"])
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_thread
    init_db()
    _requeue_interrupted_jobs()
    _worker_stop.clear()
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="swaralab-worker")
    _worker_thread.start()
    yield
    _worker_stop.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=2)


app = FastAPI(title="SwaraLab Local API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SettingsUpdate(BaseModel):
    preferred_device: Literal["auto", "cuda", "cpu"]
    chunk_max_chars: int = Field(ge=60, le=400)


class GenerationCreate(BaseModel):
    voice_profile_id: str
    text: str = Field(min_length=1, max_length=3000)
    language: Literal["en", "hi"]
    device: Literal["auto", "cuda", "cpu"] | None = None
    exaggeration: float = Field(default=0.65, ge=0.1, le=1.5)
    cfg_weight: float = Field(default=0.50, ge=0.1, le=1.0)
    temperature: float = Field(default=0.75, ge=0.2, le=1.2)


def _disk_stats_gb(path_str: str) -> dict | None:
    p = Path(path_str)
    if not p.exists():
        return None
    usage = shutil.disk_usage(p)
    return {
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def _wsl_ram_info() -> dict:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {}
    vals: dict[str, int] = {}
    for line in meminfo.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            vals[parts[0].rstrip(":")] = int(parts[1])
    return {
        "ram_total_mb": round(vals.get("MemTotal", 0) / 1024, 1),
        "ram_available_mb": round(vals.get("MemAvailable", 0) / 1024, 1),
        "swap_total_mb": round(vals.get("SwapTotal", 0) / 1024, 1),
        "swap_free_mb": round(vals.get("SwapFree", 0) / 1024, 1),
    }


def _gpu_info() -> dict:
    try:
        import torch

        avail = torch.cuda.is_available()
        if not avail:
            return {"cuda_available": False, "gpu_name": None, "vram_total_mb": None}
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda_available": True,
            "gpu_name": props.name,
            "vram_total_mb": round(props.total_memory / (1024 * 1024), 1),
        }
    except Exception:
        return {"cuda_available": False, "gpu_name": None, "vram_total_mb": None}


@app.get("/api/health")
def get_health():
    gpu = _gpu_info()
    ram = _wsl_ram_info()
    swap_vhdx = Path("/mnt/e/Antigravity_Data/wsl/swap.vhdx")
    conn = get_connection()
    try:
        profile_count = conn.execute("SELECT COUNT(*) FROM voice_profiles").fetchone()[0]
        job_count = conn.execute("SELECT COUNT(*) FROM generation_jobs").fetchone()[0]
    finally:
        conn.close()

    return {
        "status": "ok",
        "engine_mode": "mock" if os.getenv("SWARALAB_ENGINE_MODE") == "mock" else "chatterbox-v2",
        "model": {
            "repo_id": PINNED_REPO_ID,
            "revision": PINNED_HF_REVISION,
            "t3_model": PINNED_T3_MODEL,
            "precision_policy": "float32-only (fp16 excluded)",
            "cached_offline": engine.is_model_cached(),
            "loaded_device": engine.loaded_device(),
        },
        "gpu": gpu,
        "memory": ram,
        "storage": {
            "e_drive": _disk_stats_gb("/mnt/e") or _disk_stats_gb(str(ROOT)),
            "c_drive": _disk_stats_gb("/mnt/c"),
            "swap_vhdx_exists": swap_vhdx.exists(),
            "swap_vhdx_mb": round(swap_vhdx.stat().st_size / (1024 * 1024), 2) if swap_vhdx.exists() else 0.0,
            "db_path": str(DB_PATH),
        },
        "counts": {
            "voices": profile_count,
            "generations": job_count,
            "queued": job_queue.qsize(),
        },
    }


@app.get("/api/feasibility")
def get_feasibility():
    def read_json(name: str):
        p = FEASIBILITY_DIR / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    stage1_clips = []
    for fname in (
        "ref_authorized.wav",
        "stage1_cpu_fp32_run1_en.wav",
        "stage1_cpu_fp32_run1_hi.wav",
        "stage1_cpu_fp32_run2_en.wav",
        "stage1_cpu_fp32_run2_hi.wav",
        "stage1_cuda_fp32_run1_en.wav",
        "stage1_cuda_fp32_run1_hi.wav",
        "stage1_cuda_fp32_run2_en.wav",
        "stage1_cuda_fp32_run2_hi.wav",
    ):
        fp = FEASIBILITY_DIR / fname
        if fp.exists():
            stage1_clips.append(
                {
                    "filename": fname,
                    "duration_sec": read_wav_duration(fp),
                    "url": f"/api/feasibility/audio/{fname}",
                }
            )

    human_ref = FEASIBILITY_DIR / "ref_human.wav"
    return {
        "cpu_fp32": read_json("bench_cpu_fp32.json"),
        "cuda_fp32": read_json("bench_cuda_fp32.json"),
        "human_reference": {
            "expected_path": "E:\\Antigravity project\\voice cloning agent\\data\\feasibility\\ref_human.wav",
            "present": human_ref.exists(),
            "duration_sec": read_wav_duration(human_ref) if human_ref.exists() else None,
        },
        "synthesized_warning": "Stage 1 clips used a locally synthesized reference (ref_authorized.wav) and do not establish personal voice accuracy.",
        "clips": stage1_clips,
    }


@app.get("/api/feasibility/audio/{filename}")
def get_feasibility_audio(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not (filename.startswith("stage1_") or filename == "ref_authorized.wav"):
        raise HTTPException(status_code=404, detail="Stage 1 clip not found.")
    fp = FEASIBILITY_DIR / filename
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(fp, media_type="audio/wav", filename=filename)


@app.get("/api/settings")
def get_settings():
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


@app.put("/api/settings")
def update_settings(payload: SettingsUpdate):
    now = utc_now_iso()
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE app_settings SET preferred_device = ?, chunk_max_chars = ?, updated_at = ? WHERE id = 1",
            (payload.preferred_device, payload.chunk_max_chars, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


@app.get("/api/voices")
def list_voices():
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM voice_profiles ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["audio_exists"] = _resolve_stored_path(d["reference_audio_path"]).exists()
            out.append(d)
        return out
    finally:
        conn.close()


@app.post("/api/voices", status_code=201)
async def create_voice(
    name: str = Form(...),
    language_tag: Literal["en", "hi", "bilingual"] = Form("bilingual"),
    consent_confirmed: bool = Form(...),
    consent_note: str = Form(""),
    audio: UploadFile = File(...),
):
    clean_name = name.strip()
    if len(clean_name) < 2 or len(clean_name) > 80:
        raise HTTPException(status_code=400, detail="Profile name must be between 2 and 80 characters.")
    if not consent_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Explicit speaker authorization confirmation is required before saving a voice profile.",
        )

    profile_id = str(uuid.uuid4())
    ext = Path(audio.filename or "upload.wav").suffix.lower() or ".wav"
    tmp_upload = VOICES_DIR / f"{profile_id}_raw{ext}"
    dest_wav = VOICES_DIR / f"{profile_id}.wav"

    try:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Uploaded file exceeds 50 MB limit.")
        tmp_upload.write_bytes(content)
        duration_sec, sample_rate = inspect_and_normalize_reference(tmp_upload, dest_wav)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_upload.unlink(missing_ok=True)

    now = utc_now_iso()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO voice_profiles (
                id, name, language_tag, reference_audio_path,
                duration_sec, sample_rate, consent_confirmed, consent_note,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                profile_id,
                clean_name,
                language_tag,
                _to_rel_path(dest_wav),
                duration_sec,
                sample_rate,
                consent_note.strip()[:240],
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM voice_profiles WHERE id = ?", (profile_id,)).fetchone()
        res = dict(row)
        res["audio_exists"] = True
        return res
    finally:
        conn.close()


@app.get("/api/voices/{voice_id}/audio")
def get_voice_audio(voice_id: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT reference_audio_path, name FROM voice_profiles WHERE id = ?", (voice_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    fp = _resolve_stored_path(row["reference_audio_path"])
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Reference audio file is missing on disk.")
    return FileResponse(fp, media_type="audio/wav", filename=f"{row['name']}.wav")


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT reference_audio_path FROM voice_profiles WHERE id = ?", (voice_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Voice profile not found.")
        conn.execute("DELETE FROM voice_profiles WHERE id = ?", (voice_id,))
        conn.commit()
    finally:
        conn.close()
    _resolve_stored_path(row["reference_audio_path"]).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/generations", status_code=201)
def create_generation(payload: GenerationCreate):
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Enter text to synthesize.")

    conn = get_connection()
    try:
        profile = conn.execute(
            "SELECT * FROM voice_profiles WHERE id = ?", (payload.voice_profile_id,)
        ).fetchone()
        if not profile:
            raise HTTPException(status_code=404, detail="Selected voice profile was not found.")
        settings = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        chosen_device = payload.device or (settings["preferred_device"] if settings else "cpu")

        job_id = str(uuid.uuid4())
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO generation_jobs (
                id, voice_profile_id, voice_name_snapshot, text, language,
                engine_id, device_used, precision_used, exaggeration, cfg_weight, temperature,
                status, progress_pct, status_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'chatterbox-multilingual-v2', ?, 'float32', ?, ?, ?, 'queued', 2, ?, ?, ?)
            """,
            (
                job_id,
                profile["id"],
                profile["name"],
                clean_text,
                payload.language,
                chosen_device,
                payload.exaggeration,
                payload.cfg_weight,
                payload.temperature,
                "Queued for local synthesis...",
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        conn.close()

    job_queue.put(job_id)
    res = dict(row)
    res["audio_exists"] = False
    return res


@app.get("/api/generations")
def list_generations(
    q: str = Query("", max_length=120),
    language: str = Query("all"),
):
    sql = "SELECT * FROM generation_jobs WHERE 1=1"
    params: list[str] = []
    if language in ("en", "hi"):
        sql += " AND language = ?"
        params.append(language)
    if q.strip():
        sql += " AND (LOWER(text) LIKE ? OR LOWER(voice_name_snapshot) LIKE ?)"
        like = f"%{q.strip().lower()}%"
        params.extend([like, like])
    sql += " ORDER BY created_at DESC"

    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            out_path = d.get("output_audio_path")
            d["audio_exists"] = bool(out_path and _resolve_stored_path(out_path).exists())
            out.append(d)
        return out
    finally:
        conn.close()


@app.get("/api/generations/{job_id}")
def get_generation(job_id: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    d = dict(row)
    out_path = d.get("output_audio_path")
    d["audio_exists"] = bool(out_path and _resolve_stored_path(out_path).exists())
    return d


@app.get("/api/generations/{job_id}/events")
async def stream_generation_events(job_id: str):
    async def event_stream():
        last_payload = None
        while True:
            conn = get_connection()
            try:
                row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
            finally:
                conn.close()

            if not row:
                yield f"data: {json.dumps({'error': 'Generation job not found'})}\n\n"
                break

            d = dict(row)
            out_path = d.get("output_audio_path")
            d["audio_exists"] = bool(out_path and _resolve_stored_path(out_path).exists())
            encoded = json.dumps(d)
            if encoded != last_payload:
                yield f"data: {encoded}\n\n"
                last_payload = encoded

            if d["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.35)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/generations/{job_id}/audio")
def get_generation_audio(job_id: str, download: int = Query(0)):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    if not row or not row["output_audio_path"]:
        raise HTTPException(status_code=404, detail="Generated audio not found.")
    fp = _resolve_stored_path(row["output_audio_path"])
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Generated WAV file is missing from disk.")
    safe_voice = "".join(c if c.isalnum() or c in "-_" else "_" for c in row["voice_name_snapshot"])[:32]
    fname = f"swaralab_{safe_voice}_{row['language']}_{job_id[:8]}.wav"
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{fname}"'
    return FileResponse(fp, media_type="audio/wav", filename=fname, headers=headers)


@app.post("/api/generations/{job_id}/cancel")
def cancel_generation(job_id: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT status FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        status = row["status"]
        if status in ("completed", "failed"):
            return {"ok": True, "status": status}
        _mark_cancelled(job_id)
        if status == "queued":
            conn.execute(
                "UPDATE generation_jobs SET status = 'failed', progress_pct = 0, status_message = 'Cancelled', error_message = 'Cancelled by user.', updated_at = ? WHERE id = ?",
                (utc_now_iso(), job_id),
            )
            conn.commit()
            return {"ok": True, "status": "failed"}
        conn.execute(
            "UPDATE generation_jobs SET status_message = 'Cancelling after current chunk...', updated_at = ? WHERE id = ?",
            (utc_now_iso(), job_id),
        )
        conn.commit()
        return {"ok": True, "status": "running"}
    finally:
        conn.close()


@app.delete("/api/generations/{job_id}")
def delete_generation(job_id: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT output_audio_path FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        conn.execute("DELETE FROM generation_jobs WHERE id = ?", (job_id,))
        conn.commit()
    finally:
        conn.close()
    if row["output_audio_path"]:
        _resolve_stored_path(row["output_audio_path"]).unlink(missing_ok=True)
    return {"ok": True}


if WEB_DIST_DIR.exists():
    assets_dir = WEB_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        target = WEB_DIST_DIR / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(WEB_DIST_DIR / "index.html")
