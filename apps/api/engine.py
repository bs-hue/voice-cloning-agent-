"""Offline Chatterbox Multilingual V2 wrapper (CPU float32 default, CUDA float32 experimental)."""

from __future__ import annotations

import gc
import os
import re
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"

# Pin caches and strictly offline HF/Transformers lookup before importing ML libs
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hf" / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("TMPDIR", str(CACHE_DIR / "tmp"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PINNED_REPO_ID = "ResembleAI/chatterbox"
PINNED_HF_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
PINNED_T3_MODEL = "t3_mtl23ls_v2.safetensors"

_EN_UNDER_20 = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_HI_0_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]


def _int_to_en(n: int) -> str:
    if n < 20:
        return _EN_UNDER_20[n]
    if n < 100:
        return _EN_TENS[n // 10] + ("" if n % 10 == 0 else f" {_EN_UNDER_20[n % 10]}")
    if n < 1000:
        rem = n % 100
        return f"{_EN_UNDER_20[n // 100]} hundred" + ("" if rem == 0 else f" {_int_to_en(rem)}")
    if 2000 <= n <= 2099:
        rem = n % 100
        if rem == 0:
            return "two thousand"
        if rem < 10:
            return f"two thousand {_EN_UNDER_20[rem]}"
        return f"twenty {_int_to_en(rem)}"
    if n < 1_000_000:
        rem = n % 1000
        return f"{_int_to_en(n // 1000)} thousand" + ("" if rem == 0 else f" {_int_to_en(rem)}")
    return str(n)


def _int_to_hi(n: int) -> str:
    if n < 100:
        return _HI_0_99[n]
    if n < 1000:
        rem = n % 100
        return f"{_HI_0_99[n // 100]} सौ" + ("" if rem == 0 else f" {_int_to_hi(rem)}")
    if n < 100_000:
        rem = n % 1000
        return f"{_int_to_hi(n // 1000)} हज़ार" + ("" if rem == 0 else f" {_int_to_hi(rem)}")
    return str(n)


_HI_LOANWORDS: dict[str, str] = {
    "universal": "यूनिवर्सल",
    "ready-made": "रेडी-मेड",
    "readymade": "रेडीमेड",
    "ready": "रेडी",
    "made": "मेड",
    "basic": "बेसिक",
    "planets": "प्लैनेट्स",
    "planet": "प्लैनेट",
    "real": "रियल",
    "impact": "इम्पैक्ट",
    "behaviour": "बिहेवियर",
    "behavior": "बिहेवियर",
    "life": "लाइफ",
    "repeat": "रिपीट",
    "patterns": "पैटर्न्स",
    "pattern": "पैटर्न",
    "career": "करियर",
    "job": "जॉब",
    "finance": "फाइनेंस",
    "love": "लव",
    "health": "हेल्थ",
    "marriage": "मैरिज",
    "aspect": "ऐस्पेक्ट",
    "aspects": "ऐस्पेक्ट्स",
    "clear": "क्लीयर",
    "explanation": "एक्सप्लेनेशन",
    "practical": "प्रैक्टिकल",
    "remedies": "रेमेडीज़",
    "remedy": "रेमेडी",
    "simple": "सिंपल",
    "doable": "डूएबल",
    "guidance": "गाइडेंस",
    "order": "ऑर्डर",
    "successful": "सक्सेसफुल",
    "success": "सक्सेस",
    "year": "ईयर",
    "years": "ईयर्स",
    "report": "रिपोर्ट",
    "reports": "रिपोर्ट्स",
    "future": "फ्यूचर",
    "business": "बिज़नेस",
    "money": "मनी",
    "problem": "प्रॉब्लम",
    "problems": "प्रॉब्लम्स",
    "solution": "सॉल्यूशन",
    "solutions": "सॉल्यूशंस",
    "family": "फैमिली",
    "office": "ऑफिस",
    "online": "ऑनलाइन",
    "offline": "ऑफलाइन",
    "system": "सिस्टम",
    "voice": "वॉयस",
    "video": "वीडियो",
    "audio": "ऑडियो",
    "time": "टाइम",
    "free": "फ्री",
    "fast": "फास्ट",
    "best": "बेस्ट",
    "good": "गुड",
    "new": "न्यू",
    "today": "टुडे",
    "now": "नाउ",
    "link": "लिंक",
    "click": "क्लिक",
    "call": "कॉल",
    "message": "मैसेज",
    "email": "ईमेल",
    "phone": "फोन",
    "whatsapp": "व्हाट्सऐप",
    "youtube": "यूट्यूब",
    "google": "गूगल",
    "ai": "एआई",
    "app": "ऐप",
}


def _latin_word_to_devanagari(word: str) -> str:
    """Map English/Hinglish Latin tokens to Devanagari so Hindi G2P pronounces them accurately."""
    key = word.lower()
    if key in _HI_LOANWORDS:
        return _HI_LOANWORDS[key]
    # Handle hyphenated words part-by-part (e.g. "step-by-step")
    if "-" in key:
        return "-".join(_latin_word_to_devanagari(part) for part in key.split("-") if part)

    # Lightweight phonetic fallback for unknown English words in Hindi mode
    s = key
    if len(s) > 3 and s.endswith("e") and not s.endswith(("ee", "ie", "oe", "ye")):
        s = s[:-1]
    replacements = [
        ("tion", "शन"),
        ("sion", "ज़न"),
        ("ture", "चर"),
        ("ch", "च"),
        ("sh", "श"),
        ("th", "थ"),
        ("ph", "फ"),
        ("qu", "क्व"),
        ("ck", "क"),
        ("ng", "ंग"),
        ("oo", "ऊ"),
        ("ee", "ई"),
        ("ea", "ई"),
        ("ai", "ए"),
        ("ay", "ए"),
        ("ow", "ओ"),
        ("ou", "आउ"),
    ]
    for src, dst in replacements:
        s = s.replace(src, dst)
    char_map = {
        "a": "अ", "b": "ब", "c": "क", "d": "ड", "e": "ए", "f": "फ",
        "g": "ग", "h": "ह", "i": "इ", "j": "ज", "k": "क", "l": "ल",
        "m": "म", "n": "न", "o": "ओ", "p": "प", "q": "क", "r": "र",
        "s": "स", "t": "ट", "u": "उ", "v": "व", "w": "व", "x": "क्स",
        "y": "य", "z": "ज़",
    }
    return "".join(char_map.get(ch, ch) for ch in s)


def clean_text_for_tts(text: str, language: str = "hi") -> str:
    """Normalize bullets, emojis/checkmarks, quotes, line breaks, numbers, and Hinglish English words."""
    # Replace decorative bullets/checkmarks/icons so MTLTokenizer does not vocalize stray symbols (e.g. ✔ -> 90)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[•✔✓✅☑️★☆➤▪▫🔹🔸👉👇⚡🔥✨]+", " ", text)
    text = re.sub(r'["“”]+', " ", text)
    text = text.replace("…", ", ").replace("...", ", ")

    # Preserve pauses at paragraph & line boundaries before collapsing whitespace
    raw_lines = [line.strip() for line in text.split("\n")]
    joined_parts: list[str] = []
    for idx, line in enumerate(raw_lines):
        if not line:
            if joined_parts and not joined_parts[-1].endswith((".", "!", "?", "।")):
                joined_parts[-1] = joined_parts[-1].rstrip(",;:- ") + ("।" if language == "hi" else ".")
            continue
        next_is_blank = idx + 1 < len(raw_lines) and not raw_lines[idx + 1]
        if not line.endswith((".", "!", "?", "।", ",", ";", ":")):
            if next_is_blank or idx == len(raw_lines) - 1:
                line += "।" if language == "hi" else "."
            else:
                line += ","
        joined_parts.append(line)

    cleaned = " ".join(joined_parts)
    cleaned = re.sub(r"\s+([,.!?;:।])", r"\1", cleaned)
    cleaned = re.sub(r",{2,}", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    def _replace_number(match: re.Match[str]) -> str:
        raw = match.group(0)
        val = int(raw)
        if val > 99999:
            return raw
        return _int_to_hi(val) if language == "hi" else _int_to_en(val)

    cleaned = re.sub(r"\b\d{1,5}\b", _replace_number, cleaned)

    if language == "hi":
        cleaned = re.sub(
            r"\b[A-Za-z]+(?:-[A-Za-z]+)*\b",
            lambda m: _latin_word_to_devanagari(m.group(0)),
            cleaned,
        )

    return cleaned


def split_text_chunks(text: str, max_chars: int = 220) -> list[str]:
    """Split text along paragraph/sentence boundaries while keeping chunks large enough for natural prosody."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    sentences = re.split(r"(?<=[.!?।])\s+", clean)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            clauses = re.split(r"(?<=[,;:])\s+", sent)
            part = ""
            for clause in clauses:
                if len(clause) > max_chars:
                    if part:
                        chunks.append(part.strip())
                        part = ""
                    for w in clause.split(" "):
                        if len(part) + len(w) + 1 <= max_chars:
                            part = f"{part} {w}".strip()
                        else:
                            if part:
                                chunks.append(part)
                            part = w
                elif len(part) + len(clause) + 1 <= max_chars:
                    part = f"{part} {clause}".strip()
                else:
                    if part:
                        chunks.append(part.strip())
                    part = clause
            if part:
                chunks.append(part.strip())
        elif len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks or [clean]


def _apply_edge_fade(arr: np.ndarray, sr: int, fade_ms: float = 10.0) -> np.ndarray:
    n = min(int(sr * fade_ms / 1000.0), len(arr) // 4)
    if n <= 1:
        return arr
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = arr.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


class JobCancelledError(RuntimeError):
    """Raised when a user cancels a queued or running synthesis job."""


class ChatterboxV2Engine:
    """Single-model offline V2 inference manager (never uses fp16)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._loaded_device: str | None = None
        self._ckpt_dir: Path | None = None

    def resolve_checkpoint_dir(self) -> Path | None:
        if self._ckpt_dir and self._ckpt_dir.exists():
            return self._ckpt_dir
        try:
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    repo_id=PINNED_REPO_ID,
                    repo_type="model",
                    revision=PINNED_HF_REVISION,
                    local_files_only=True,
                    allow_patterns=[
                        "ve.pt",
                        PINNED_T3_MODEL,
                        "s3gen.pt",
                        "grapheme_mtl_merged_expanded_v1.json",
                        "conds.pt",
                        "Cangjie5_TC.json",
                    ],
                )
            )
            self._ckpt_dir = path
            return path
        except Exception:
            return None

    def is_model_cached(self) -> bool:
        ckpt = self.resolve_checkpoint_dir()
        return bool(ckpt and (ckpt / PINNED_T3_MODEL).exists() and (ckpt / "s3gen.pt").exists())

    def loaded_device(self) -> str | None:
        return self._loaded_device

    def resolve_device(self, preferred: str, text_length: int = 0) -> tuple[str, str | None]:
        """Use Sequential CUDA (T3 then S3Gen on GPU, <=2.3 GB VRAM) when CUDA is selected or auto."""
        if preferred in ("cuda", "auto"):
            import torch

            if torch.cuda.is_available():
                return "cuda", None
            if preferred == "cuda":
                raise RuntimeError(
                    "GPU (CUDA) mode was requested, but CUDA is unavailable inside SwaraLabWSL."
                )
            return "cpu", "CUDA unavailable — running on CPU float32."
        return "cpu", None

    def _ensure_loaded(self, device: str, on_progress: Callable[[int, str], None]) -> None:
        try:
            import torch

            # Physical cores (8 on Ryzen 7 7435HS) avoid hyperthread contention (~1.6x faster than 12 threads)
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        except Exception:
            pass

        # Sequential CUDA keeps base model on CPU and swaps t3/s3gen onto GPU one at a time (<=2.3 GB VRAM)
        base_device = "cpu"
        if self._model is not None and self._loaded_device == base_device:
            return

        if self._model is not None:
            on_progress(12, f"Unloading previous model from {self._loaded_device}...")
            del self._model
            self._model = None
            self._loaded_device = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        ckpt_dir = self.resolve_checkpoint_dir()
        if not ckpt_dir:
            raise RuntimeError(
                f"Pinned offline V2 checkpoint ({PINNED_T3_MODEL}, rev {PINNED_HF_REVISION[:8]}) was not found in data/cache/hf."
            )

        on_progress(20, f"Loading Chatterbox Multilingual V2 ({device}, float32)...")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._model = ChatterboxMultilingualTTS.from_local(
            ckpt_dir,
            device=base_device,
            t3_model=PINNED_T3_MODEL,
        )
        self._loaded_device = base_device

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        ref_wav_path: Path,
        out_wav_path: Path,
        preferred_device: str = "auto",
        chunk_max_chars: int = 220,
        exaggeration: float = 0.65,
        cfg_weight: float = 0.50,
        temperature: float = 0.75,
        on_progress: Callable[[int, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict:
        progress = on_progress or (lambda pct, msg: None)
        cancelled = should_cancel or (lambda: False)

        if os.getenv("SWARALAB_ENGINE_MODE") == "mock":
            if cancelled():
                raise JobCancelledError("Cancelled by user.")
            sr = 24000
            dur = 1.2
            t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
            tone = 0.12 * np.sin(2 * np.pi * (220.0 if language == "en" else 260.0) * t)
            out_wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_wav_path), tone, sr, subtype="PCM_16")
            progress(100, "Mock synthesis complete (test mode)")
            return {"device_used": "cpu", "precision_used": "float32", "audio_duration_sec": dur}

        with self._lock:
            if cancelled():
                raise JobCancelledError("Cancelled by user.")

            target_device, device_note = self.resolve_device(preferred_device, text_length=len(text.strip()))
            if device_note:
                progress(10, device_note)
            self._ensure_loaded(target_device, progress)

            normalized_text = clean_text_for_tts(text, language=language)
            chunks = split_text_chunks(normalized_text, max_chars=min(chunk_max_chars, 240))
            if not chunks:
                raise ValueError("Prompt text cannot be empty.")

            if cancelled():
                raise JobCancelledError("Cancelled by user.")

            progress(28, "Extracting speaker voice profile...")
            self._model.prepare_conditionals(str(ref_wav_path), exaggeration=float(exaggeration))

            audio_segments: list[np.ndarray] = []
            sr = int(self._model.sr)
            silence_gap = np.zeros(int(sr * 0.08), dtype=np.float32)
            total = len(chunks)

            if target_device == "cuda":
                import torch
                import torch.nn.functional as F
                from chatterbox.mtl_tts import S3_TOKEN_RATE, S3GEN_SR, drop_invalid_tokens, punc_norm

                all_speech_tokens: list[torch.Tensor] = []
                t3_cond = self._model.conds.t3.to(device="cuda")
                try:
                    self._model.t3.to("cuda")
                    for idx, chunk in enumerate(chunks):
                        if cancelled():
                            raise JobCancelledError("Cancelled by user.")
                        pct = 32 + int((idx / max(total, 1)) * 43)
                        progress(
                            pct,
                            f"Generating speech tokens on GPU CUDA (float32) — part {idx + 1} of {total}...",
                        )
                        text_tokens = self._model.tokenizer.text_to_tokens(
                            punc_norm(chunk),
                            language_id=language.lower() if language else None,
                        ).to("cuda")
                        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
                        text_tokens = F.pad(text_tokens, (1, 0), value=self._model.t3.hp.start_text_token)
                        text_tokens = F.pad(text_tokens, (0, 1), value=self._model.t3.hp.stop_text_token)
                        with torch.inference_mode():
                            st = self._model.t3.inference(
                                t3_cond=t3_cond,
                                text_tokens=text_tokens,
                                max_new_tokens=1000,
                                temperature=float(temperature),
                                cfg_weight=float(cfg_weight),
                            )
                            st = drop_invalid_tokens(st[0]).to("cpu")
                        all_speech_tokens.append(st)
                finally:
                    self._model.t3.to("cpu")
                    del t3_cond
                    torch.cuda.empty_cache()

                if cancelled():
                    raise JobCancelledError("Cancelled by user.")

                gen_cond = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in self._model.conds.gen.items()}
                try:
                    self._model.s3gen.to("cuda")
                    for idx, st in enumerate(all_speech_tokens):
                        if cancelled():
                            raise JobCancelledError("Cancelled by user.")
                        pct = 76 + int((idx / max(total, 1)) * 18)
                        progress(
                            pct,
                            f"Rendering waveform on GPU CUDA (float32) — part {idx + 1} of {total}...",
                        )
                        with torch.inference_mode():
                            wav, _ = self._model.s3gen.inference(
                                speech_tokens=st.to("cuda"),
                                ref_dict=gen_cond,
                            )
                            wav_np = wav.squeeze(0).detach().cpu().numpy()
                        n_tok = int(st.shape[-1])
                        wav_np = wav_np[: max(1, n_tok - 1) * (S3GEN_SR // S3_TOKEN_RATE)]
                        arr = self._model.watermarker.apply_watermark(wav_np, sample_rate=sr).astype(np.float32)
                        if arr.size > 0:
                            audio_segments.append(_apply_edge_fade(arr, sr))
                            if idx < total - 1:
                                audio_segments.append(silence_gap)
                finally:
                    self._model.s3gen.to("cpu")
                    del gen_cond
                    torch.cuda.empty_cache()
            else:
                for idx, chunk in enumerate(chunks):
                    if cancelled():
                        raise JobCancelledError("Cancelled by user.")
                    pct = 35 + int((idx / max(total, 1)) * 55)
                    progress(
                        pct,
                        f"Generating speech on CPU (float32) — part {idx + 1} of {total}...",
                    )
                    wav_tensor = self._model.generate(
                        chunk,
                        language_id=language,
                        audio_prompt_path=None,
                        exaggeration=float(exaggeration),
                        cfg_weight=float(cfg_weight),
                        temperature=float(temperature),
                    )
                    arr = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
                    if arr.size > 0:
                        audio_segments.append(_apply_edge_fade(arr, sr))
                        if idx < total - 1:
                            audio_segments.append(silence_gap)

            if target_device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if not audio_segments:
                raise RuntimeError("Engine produced an empty audio buffer.")

            combined = np.concatenate(audio_segments)
            out_wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_wav_path), combined, sr, subtype="PCM_16")
            dur = round(len(combined) / float(sr), 2)
            progress(100, f"Completed ({dur:.1f}s WAV ready)")
            return {
                "device_used": target_device,
                "precision_used": "float32",
                "audio_duration_sec": dur,
            }
