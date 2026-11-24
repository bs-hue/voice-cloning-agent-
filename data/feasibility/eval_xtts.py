"""Benchmark and evaluate Coqui XTTS-v2 zero-shot voice cloning with tuned parameters."""

import os
import time
import torch
from pathlib import Path

# PyTorch 2.6 defaults weights_only=True; allow TTS checkpoints to load safely
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"

os.environ["TTS_HOME"] = str(CACHE_DIR / "tts")
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

ref_clean_6s = str(ROOT / "data" / "feasibility" / "sakshi_clean_6s.wav")
out_dir = ROOT / "data" / "feasibility"
out_dir.mkdir(parents=True, exist_ok=True)

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

tests = [
    (
        "xtts_sakshi_promo.wav",
        "सुरभि कुंडली में आपको मिलता है क्लियर एक्सप्लेनेशन, प्रैक्टिकल रेमेडीज़ और सही मार्गदर्शन।",
    ),
    (
        "xtts_sakshi_devotional.wav",
        "पौराणिक कथाओं के अनुसार, जब संसार में अत्यधिक अशांति फैलने लगी, तब देवी बगलामुखी का प्राकट्य हुआ।",
    ),
]

for out_name, prompt in tests:
    out_path = str(out_dir / out_name)
    print(f"\n--- Generating: {out_name} ---")
    t0 = time.perf_counter()
    tts.tts_to_file(
        text=prompt,
        speaker_wav=ref_clean_6s,
        language="hi",
        file_path=out_path,
        temperature=0.65,
        repetition_penalty=2.5,
        top_k=50,
        top_p=0.85,
    )
    dt = time.perf_counter() - t0
    import soundfile as sf
    data, sr = sf.read(out_path)
    dur = len(data) / sr
    print(f"Generated {out_name} in {dt:.2f}s | Audio: {dur:.2f}s | RTF: {dt/dur:.2f}x")

print("\nDone all XTTS-v2 tests.")