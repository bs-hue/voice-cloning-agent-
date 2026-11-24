"""Audio validation, normalization, and WAV duration inspection helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
TARGET_SAMPLE_RATE = 24000
MIN_DURATION_SEC = 3.0
MAX_DURATION_SEC = 120.0
MIN_RMS_AMPLITUDE = 0.002


def _find_ffmpeg() -> str | None:
    for candidate in (
        shutil.which("ffmpeg"),
        "/mnt/c/Users/Yogita Thakur/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    winget_root = Path("/mnt/c/Users/Yogita Thakur/AppData/Local/Microsoft/WinGet/Packages")
    if winget_root.exists():
        for exe in winget_root.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"):
            return str(exe)
    return None


def inspect_and_normalize_reference(src_path: Path, dest_wav_path: Path) -> Tuple[float, int]:
    """Validate an uploaded voice clip and save a clean mono 24 kHz 16-bit PCM WAV."""
    ext = src_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext or 'unknown'}'. Use .wav, .mp3, .flac, .ogg, or .m4a."
        )

    dest_wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio: np.ndarray | None = None
    sr: int = TARGET_SAMPLE_RATE

    try:
        data, file_sr = sf.read(str(src_path), dtype="float32", always_2d=False)
        audio = np.asarray(data, dtype=np.float32)
        sr = int(file_sr)
    except Exception:
        ffmpeg_bin = _find_ffmpeg()
        if not ffmpeg_bin:
            raise ValueError(
                "Could not decode this audio format directly. Please upload a clean 16-bit PCM .wav file."
            )
        tmp_decoded = dest_wav_path.with_suffix(".tmp.wav")
        proc = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(src_path),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(tmp_decoded),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0 or not tmp_decoded.exists():
            tmp_decoded.unlink(missing_ok=True)
            raise ValueError(
                "Could not decode audio file with ffmpeg. Please export a clean 16-bit .wav clip."
            )
        try:
            data, file_sr = sf.read(str(tmp_decoded), dtype="float32", always_2d=False)
            audio = np.asarray(data, dtype=np.float32)
            sr = int(file_sr)
        finally:
            tmp_decoded.unlink(missing_ok=True)

    if audio is None or audio.size == 0:
        raise ValueError("Uploaded audio file is empty.")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Uploaded audio contains invalid or corrupted samples.")

    import librosa

    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    # Trim dead silence at the start/end so Chatterbox's first 6-10s conditioning window captures active speech
    trimmed, _ = librosa.effects.trim(audio, top_db=28)
    if trimmed.size > 0:
        audio = trimmed

    duration_sec = len(audio) / float(sr)
    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(
            f"Reference recording is too short ({duration_sec:.1f}s of active speech). Provide at least {MIN_DURATION_SEC:.0f} seconds (6-12 seconds recommended)."
        )
    if duration_sec > MAX_DURATION_SEC:
        raise ValueError(
            f"Reference recording is too long ({duration_sec:.1f}s). Trim to under {MAX_DURATION_SEC:.0f} seconds (6-12 seconds recommended)."
        )

    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < MIN_RMS_AMPLITUDE:
        raise ValueError(
            "Uploaded audio appears silent or too quiet to extract speaker characteristics."
        )

    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = audio * (0.95 / peak)

    sf.write(str(dest_wav_path), audio, sr, subtype="PCM_16")
    return round(len(audio) / float(sr), 2), sr


def read_wav_duration(wav_path: Path) -> float | None:
    if not wav_path.exists():
        return None
    try:
        info = sf.info(str(wav_path))
        return round(float(info.duration), 2)
    except Exception:
        return None

