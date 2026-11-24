import gc
import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
OUT_DIR = ROOT / "data" / "feasibility"

# Force configurable caches onto E: (/mnt/e on WSL2) before importing torch / transformers / hf_hub
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hf" / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
os.environ["TEMP"] = str(CACHE_DIR / "tmp")
os.environ["TMP"] = str(CACHE_DIR / "tmp")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

HF_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_MODEL = "t3_mtl23ls_v2.safetensors"

EN_TEXT = "Hello, this is a local voice cloning feasibility test running on my computer."
HI_TEXT = "नमस्ते, यह मेरे कंप्यूटर पर स्थानीय आवाज़ क्लोनिंग का परीक्षण है।"


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / (1024 * 1024), 2)


def get_process_rss_mb() -> float:
    status_file = Path("/proc/self/status")
    if status_file.exists():
        for line in status_file.read_text().splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return round(kb / 1024, 1)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return round(ru.ru_maxrss / 1024, 1)


def nvidia_smi_used_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def save_wav(tensor_wav: torch.Tensor, sr: int, path: Path) -> float:
    arr = tensor_wav.squeeze().detach().cpu().numpy()
    sf.write(str(path), arr, sr)
    return round(len(arr) / float(sr), 2)


def run_device_bench(ckpt_dir: Path, device: str, ref_wav: Path, use_fp16: bool = False) -> dict:
    label = f"{device}{'_fp16' if use_fp16 else '_fp32'}"
    res: dict = {"label": label, "device": device, "use_fp16": use_fp16, "status": "ok"}

    if device == "cuda":
        if not torch.cuda.is_available():
            return {"label": label, "status": "skipped", "error": "CUDA not available"}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    vram_before = nvidia_smi_used_mb() if device == "cuda" else None
    rss_before = get_process_rss_mb()

    t0 = time.perf_counter()
    model = None
    try:
        model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device=device, t3_model=T3_MODEL)
        if use_fp16 and device == "cuda":
            model.t3.half()
            model.s3gen.half()
            model.ve.half()
        res["load_sec"] = round(time.perf_counter() - t0, 2)
        if device == "cuda":
            res["load_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
            res["load_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)

        clips = {}
        for lang, text in [("en", EN_TEXT), ("hi", HI_TEXT)]:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            gt0 = time.perf_counter()
            if use_fp16 and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    wav = model.generate(text, language_id=lang, audio_prompt_path=str(ref_wav))
            else:
                wav = model.generate(text, language_id=lang, audio_prompt_path=str(ref_wav))
            elapsed = round(time.perf_counter() - gt0, 2)
            out_path = OUT_DIR / f"{label}_{lang}.wav"
            dur = save_wav(wav, model.sr, out_path)
            clip_info = {
                "file": str(out_path),
                "gen_sec": elapsed,
                "audio_sec": dur,
                "rtf": round(elapsed / dur, 2) if dur > 0 else None,
            }
            if device == "cuda":
                clip_info["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                clip_info["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)
                clip_info["nvidia_smi_used_mb"] = nvidia_smi_used_mb()
            clips[lang] = clip_info

        res["clips"] = clips
        res["rss_after_mb"] = get_process_rss_mb()
        res["rss_before_mb"] = rss_before
        res["nvidia_smi_before_mb"] = vram_before
    except Exception as exc:
        res["status"] = "failed"
        res["error"] = f"{type(exc).__name__}: {exc}"
        if device == "cuda":
            res["nvidia_smi_at_fail_mb"] = nvidia_smi_used_mb()
            res["peak_alloc_at_fail_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    finally:
        if model is not None:
            del model
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return res


def test_offline_reload(ref_wav: Path) -> dict:
    """Verify loading strictly from local HF cache with HF_HUB_OFFLINE=1."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        local_dir = Path(
            snapshot_download(
                repo_id="ResembleAI/chatterbox",
                repo_type="model",
                revision=HF_REVISION,
                local_files_only=True,
                allow_patterns=[
                    "ve.pt",
                    T3_MODEL,
                    "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt",
                    "Cangjie5_TC.json",
                ],
            )
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = ChatterboxMultilingualTTS.from_local(local_dir, device=device, t3_model=T3_MODEL)
        t0 = time.perf_counter()
        wav = m.generate("Offline check complete.", language_id="en", audio_prompt_path=str(ref_wav))
        dur = save_wav(wav, m.sr, OUT_DIR / "offline_check_en.wav")
        elapsed = round(time.perf_counter() - t0, 2)
        del m
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return {"status": "ok", "device": device, "gen_sec": elapsed, "audio_sec": dur}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_wav = OUT_DIR / "ref_authorized.wav"
    if not ref_wav.exists():
        raise FileNotFoundError(f"Missing reference clip: {ref_wav}")

    dl_t0 = time.perf_counter()
    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision=HF_REVISION,
            allow_patterns=[
                "ve.pt",
                T3_MODEL,
                "s3gen.pt",
                "grapheme_mtl_merged_expanded_v1.json",
                "conds.pt",
                "Cangjie5_TC.json",
            ],
        )
    )
    dl_sec = round(time.perf_counter() - dl_t0, 2)

    results = {
        "hf_revision": HF_REVISION,
        "t3_model": T3_MODEL,
        "download_sec": dl_sec,
        "ckpt_dir": str(ckpt_dir),
        "ckpt_size_mb": dir_size_mb(ckpt_dir),
        "hf_cache_mb": dir_size_mb(CACHE_DIR / "hf"),
        "pip_cache_mb": dir_size_mb(CACHE_DIR / "pip"),
        "venv_mb": dir_size_mb(Path("/opt/swaralab/.venv")),
        "runs": [],
    }

    gpu_fp32 = run_device_bench(ckpt_dir, "cuda", ref_wav, use_fp16=False)
    results["runs"].append(gpu_fp32)

    peak_res = max(
        (c.get("peak_reserved_mb", 0) for c in gpu_fp32.get("clips", {}).values()),
        default=0,
    )
    if gpu_fp32.get("status") != "ok" or peak_res > 3200:
        gpu_fp16 = run_device_bench(ckpt_dir, "cuda", ref_wav, use_fp16=True)
        results["runs"].append(gpu_fp16)

    cpu_run = run_device_bench(ckpt_dir, "cpu", ref_wav, use_fp16=False)
    results["runs"].append(cpu_run)

    results["offline_test"] = test_offline_reload(ref_wav)

    e_mount = Path("/mnt/e") if Path("/mnt/e").exists() else Path("E:\\")
    c_mount = Path("/mnt/c") if Path("/mnt/c").exists() else Path("C:\\")
    results["disk_after_gb"] = {
        "E_free_gb": round(shutil.disk_usage(e_mount).free / (1024**3), 2),
        "C_free_gb": round(shutil.disk_usage(c_mount).free / (1024**3), 2),
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
