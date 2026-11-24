"""Quick CPU float32 diagnostic comparing Hinglish-in-en vs cleaned hi vs Devanagari+English with tuned prosody."""

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
OUT_DIR = ROOT / "data" / "feasibility"

os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hf" / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import soundfile as sf
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

REF_WAV = ROOT / "data" / "voices" / "b81e3869-89da-4e02-a68b-c327f7fea3a7.wav"
HF_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_MODEL = "t3_mtl23ls_v2.safetensors"

# Sample slice of user's exact text:
TEXT_HI_MIXED = (
    "Universal upaay sab ke liye hote hain, lekin sahi upaay sirf aapke liye, "
    "jo aapko batati hai aapki Surbhi Kundli. Yeh koi ready-made ya basic kundli nahi hai. "
    "Abhi order karein apni Surbhi Kundli, aur banaiye दो हज़ार छब्बीस ko apna successful year."
)

TEXT_HI_DEVANAGARI = (
    "यूनिवर्सल उपाय सब के लिए होते हैं, लेकिन सही उपाय सिर्फ आपके लिए, "
    "जो आपको बताती है आपकी सुरभि कुंडली। यह कोई रेडीमेड या बेसिक कुंडली नहीं है। "
    "अभी ऑर्डर करें अपनी सुरभि कुंडली, और बनाइए दो हज़ार छब्बीस को अपना सक्सेसफुल ईयर।"
)

TEXT_HI_HYBRID = (
    "Universal उपाय सब के लिए होते हैं, लेकिन सही उपाय सिर्फ आपके लिए, "
    "जो आपको बताती है आपकी सुरभि कुंडली। यह कोई ready-made या basic कुंडली नहीं है। "
    "अभी order करें अपनी सुरभि कुंडली, और बनाइए दो हज़ार छब्बीस को अपना successful year."
)


def main() -> None:
    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision=HF_REVISION,
            local_files_only=True,
            allow_patterns=["ve.pt", T3_MODEL, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
        )
    )
    model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device="cpu", t3_model=T3_MODEL)

    cases = [
        ("diag_hi_roman_tuned.wav", TEXT_HI_MIXED, "hi", 0.70, 0.35, 0.80),
        ("diag_hi_hybrid_tuned.wav", TEXT_HI_HYBRID, "hi", 0.70, 0.35, 0.80),
        ("diag_hi_deva_tuned.wav", TEXT_HI_DEVANAGARI, "hi", 0.70, 0.35, 0.80),
    ]
    for fname, txt, lang, exag, cfg, temp in cases:
        t0 = time.perf_counter()
        wav = model.generate(
            txt,
            language_id=lang,
            audio_prompt_path=str(REF_WAV),
            exaggeration=exag,
            cfg_weight=cfg,
            temperature=temp,
        )
        arr = wav.squeeze().detach().cpu().numpy()
        out = OUT_DIR / fname
        sf.write(str(out), arr, model.sr, subtype="PCM_16")
        print(f"SAVED {fname} dur={len(arr)/model.sr:.2f}s elapsed={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

