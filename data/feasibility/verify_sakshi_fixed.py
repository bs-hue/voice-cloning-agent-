"""Verify updated SwaraLab engine on Sakshi ji full script using CPU float32 + clean text + tuned prosody."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.engine import ChatterboxV2Engine

REF_WAV = ROOT / "data" / "voices" / "b81e3869-89da-4e02-a68b-c327f7fea3a7.wav"
OUT_WAV = ROOT / "data" / "feasibility" / "sakshi_ji_fixed_full_hi.wav"

# Same full script with Hindi words written in Devanagari and genuine English words + original bullets/checkmarks/year:
FULL_SCRIPT = """Universal उपाय सब के लिए होते हैं…
लेकिन सही उपाय सिर्फ आपके लिए।

जो आपको बताती है आपकी सुरभि कुंडली।
यह कोई ready-made या basic कुंडली नहीं है।
इसमें देखा जाता है:

• आपके planets का real impact
• आपके नक्षत्र का behaviour
• और life के repeat होते patterns

सुरभि कुंडली में
Career, job, Finance, love, Health, marriage
life के हर aspect के लिए मिलता है:
✔ Clear explanation
✔ Practical remedies
✔ Simple, doable guidance

तो देर मत कीजिए।
अभी order करें अपनी सुरभि कुंडली
और बनाइए 2026 को अपना successful year।"""


def main() -> None:
    eng = ChatterboxV2Engine()
    t0 = time.perf_counter()
    res = eng.synthesize(
        text=FULL_SCRIPT,
        language="hi",
        ref_wav_path=REF_WAV,
        out_wav_path=OUT_WAV,
        preferred_device="cpu",
        chunk_max_chars=320,
        exaggeration=0.50,
        cfg_weight=0.35,
        temperature=0.80,
    )
    print(f"DONE {OUT_WAV} res={res} total_sec={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
