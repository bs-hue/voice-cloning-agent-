"""Clean, single-mode offline Stage 1 hardware feasibility benchmark."""

import argparse
import gc
import importlib.metadata
import json
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
OUT_DIR = ROOT / "data" / "feasibility"

os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hf" / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

HF_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_MODEL = "t3_mtl23ls_v2.safetensors"

EN_TEXT = "Hello, this is a local voice cloning feasibility test running on my computer."
HI_TEXT = "नमस्ते, यह मेरे कंप्यूटर पर स्थानीय आवाज़ क्लोनिंग का परीक्षण है।"


def pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def system_used_ram_mb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    vals = {}
    for line in meminfo.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            vals[parts[0].rstrip(":")] = int(parts[1])
    if "MemTotal" in vals and "MemAvailable" in vals:
        return round((vals["MemTotal"] - vals["MemAvailable"]) / 1024, 1)
    return None


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


def save_wav(wav: torch.Tensor, sr: int, path: Path) -> float:
    arr = wav.squeeze().detach().cpu().numpy()
    sf.write(str(path), arr, sr)
    return round(len(arr) / float(sr), 2)


def generate_once(model, text: str, lang: str, ref_wav: Path, out_path: Path, device: str, use_fp16: bool) -> dict:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if use_fp16 and device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            wav = model.generate(text, language_id=lang, audio_prompt_path=str(ref_wav))
    else:
        wav = model.generate(text, language_id=lang, audio_prompt_path=str(ref_wav))
    elapsed = round(time.perf_counter() - t0, 2)
    dur = save_wav(wav, model.sr, out_path)
    info = {
        "file": str(out_path),
        "gen_sec": elapsed,
        "audio_sec": dur,
        "rtf": round(elapsed / dur, 2) if dur > 0 else None,
        "rss_mb": rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
    }
    if device == "cuda":
        info["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        info["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)
        info["nvidia_smi_used_mb"] = nvidia_smi_used_mb()
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cuda_fp32", "cuda_fp16", "cpu_fp32"], required=True)
    args = parser.parse_args()

    device = "cuda" if args.mode.startswith("cuda") else "cpu"
    use_fp16 = args.mode == "cuda_fp16"
    ref_wav = OUT_DIR / "ref_authorized.wav"

    ckpt_dir = Path(
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

    out = {
        "mode": args.mode,
        "device": device,
        "precision": "float16" if use_fp16 else "float32",
        "offline_mode": True,
        "hf_revision": HF_REVISION,
        "t3_model": T3_MODEL,
        "supports_en": "en" in SUPPORTED_LANGUAGES,
        "supports_hi": "hi" in SUPPORTED_LANGUAGES,
        "supported_language_count": len(SUPPORTED_LANGUAGES),
        "env": {
            "python": os.sys.version.split()[0],
            "torch": pkg_version("torch"),
            "torchaudio": pkg_version("torchaudio"),
            "chatterbox-tts": pkg_version("chatterbox-tts"),
            "transformers": pkg_version("transformers"),
            "diffusers": pkg_version("diffusers"),
            "safetensors": pkg_version("safetensors"),
            "numba": pkg_version("numba"),
            "numpy": pkg_version("numpy"),
            "resemble-perth": pkg_version("resemble-perth"),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
            if torch.cuda.is_available()
            else None,
        },
        "rss_before_mb": rss_mb(),
        "sys_ram_before_mb": system_used_ram_mb(),
        "vram_before_mb": nvidia_smi_used_mb() if device == "cuda" else None,
        "status": "ok",
    }

    out_json = OUT_DIR / f"bench_{args.mode}.json"
    model = None
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device=device, t3_model=T3_MODEL)
        if use_fp16 and device == "cuda":
            model.t3.half()
            model.s3gen.half()
            model.ve.half()
        out["load_sec"] = round(time.perf_counter() - t0, 2)
        out["rss_after_load_mb"] = rss_mb()
        if device == "cuda":
            out["load_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
            out["load_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)
            out["vram_after_load_mb"] = nvidia_smi_used_mb()

        runs = {}
        for run_idx in (1, 2):
            for lang, text in (("en", EN_TEXT), ("hi", HI_TEXT)):
                key = f"run{run_idx}_{lang}"
                wav_path = OUT_DIR / f"stage1_{args.mode}_{key}.wav"
                runs[key] = generate_once(model, text, lang, ref_wav, wav_path, device, use_fp16)
        out["runs"] = runs
        out["peak_rss_overall_mb"] = peak_rss_mb()
        out["sys_ram_after_mb"] = system_used_ram_mb()
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    finally:
        if model is not None:
            del model
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"DONE {args.mode} -> {out_json}")


if __name__ == "__main__":
    main()

