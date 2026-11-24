"""Isolated API & workflow tests for SwaraLab backend."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient


class SwaraLabApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(dir="/mnt/e/Antigravity project/voice cloning agent/data/cache/tmp")
        tmp_path = Path(cls.tmp.name)

        os.environ["SWARALAB_ENGINE_MODE"] = "mock"

        import apps.api.db as db_mod
        import apps.api.server as srv_mod

        cls.db_mod = db_mod
        cls.srv_mod = srv_mod

        db_mod.DB_PATH = tmp_path / "test_studio.db"
        srv_mod.DB_PATH = db_mod.DB_PATH
        srv_mod.VOICES_DIR = tmp_path / "voices"
        srv_mod.OUTPUTS_DIR = tmp_path / "outputs"
        srv_mod.VOICES_DIR.mkdir(parents=True, exist_ok=True)
        srv_mod.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # Patch get_connection defaults used inside server
        orig_get_conn = db_mod.get_connection

        def test_get_connection():
            return orig_get_conn(db_mod.DB_PATH)

        db_mod.init_db(db_mod.DB_PATH)
        srv_mod.get_connection = test_get_connection

        cls.client_cm = TestClient(srv_mod.app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)
        cls.tmp.cleanup()

    def _make_wav_bytes(self, duration_sec: float = 4.0, sr: int = 24000, silent: bool = False) -> bytes:
        p = Path(self.tmp.name) / f"sample_{time.time_ns()}.wav"
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
        wave = np.zeros_like(t) if silent else (0.15 * np.sin(2 * np.pi * 220 * t))
        sf.write(str(p), wave, sr, subtype="PCM_16")
        data = p.read_bytes()
        p.unlink(missing_ok=True)
        return data

    def test_01_health_and_default_cpu_setting(self) -> None:
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        body = h.json()
        self.assertEqual(body["model"]["t3_model"], "t3_mtl23ls_v2.safetensors")
        self.assertTrue(body["model"]["cached_offline"])

        s = self.client.get("/api/settings")
        self.assertEqual(s.status_code, 200)
        self.assertEqual(s.json()["preferred_device"], "cpu")

    def test_02_reject_unchecked_consent_and_invalid_audio(self) -> None:
        wav_bytes = self._make_wav_bytes(4.0)
        r_no_consent = self.client.post(
            "/api/voices",
            data={"name": "Test Voice", "language_tag": "bilingual", "consent_confirmed": "false"},
            files={"audio": ("clip.wav", wav_bytes, "audio/wav")},
        )
        self.assertEqual(r_no_consent.status_code, 400)

        short_bytes = self._make_wav_bytes(1.2)
        r_short = self.client.post(
            "/api/voices",
            data={"name": "Too Short", "language_tag": "en", "consent_confirmed": "true"},
            files={"audio": ("short.wav", short_bytes, "audio/wav")},
        )
        self.assertEqual(r_short.status_code, 400)

        silent_bytes = self._make_wav_bytes(4.0, silent=True)
        r_silent = self.client.post(
            "/api/voices",
            data={"name": "Silent", "language_tag": "en", "consent_confirmed": "true"},
            files={"audio": ("silent.wav", silent_bytes, "audio/wav")},
        )
        self.assertEqual(r_silent.status_code, 400)

    def test_03_full_profile_generation_history_download_delete_flow(self) -> None:
        wav_bytes = self._make_wav_bytes(4.5)
        r_create = self.client.post(
            "/api/voices",
            data={
                "name": "Studio Test Speaker",
                "language_tag": "bilingual",
                "consent_confirmed": "true",
                "consent_note": "Automated test sample",
            },
            files={"audio": ("valid.wav", wav_bytes, "audio/wav")},
        )
        self.assertEqual(r_create.status_code, 201, r_create.text)
        voice = r_create.json()

        r_voice_audio = self.client.get(f"/api/voices/{voice['id']}/audio")
        self.assertEqual(r_voice_audio.status_code, 200)
        self.assertTrue(r_voice_audio.content.startswith(b"RIFF"))

        r_gen = self.client.post(
            "/api/generations",
            json={
                "voice_profile_id": voice["id"],
                "text": "Hello from SwaraLab.",
                "language": "en",
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
            },
        )
        self.assertEqual(r_gen.status_code, 201, r_gen.text)
        job_id = r_gen.json()["id"]

        final_job = None
        for _ in range(40):
            polled = self.client.get(f"/api/generations/{job_id}").json()
            if polled["status"] in ("completed", "failed"):
                final_job = polled
                break
            time.sleep(0.1)

        self.assertIsNotNone(final_job)
        self.assertEqual(final_job["status"], "completed")
        self.assertTrue(final_job["audio_exists"])

        r_dl = self.client.get(f"/api/generations/{job_id}/audio?download=1")
        self.assertEqual(r_dl.status_code, 200)
        self.assertIn("attachment;", r_dl.headers.get("content-disposition", ""))
        self.assertTrue(r_dl.content.startswith(b"RIFF"))

        r_hist_en = self.client.get("/api/generations?language=en&q=SwaraLab")
        self.assertEqual(r_hist_en.status_code, 200)
        self.assertEqual(len(r_hist_en.json()), 1)

        r_hist_hi = self.client.get("/api/generations?language=hi")
        self.assertEqual(r_hist_hi.status_code, 200)
        self.assertEqual(len(r_hist_hi.json()), 0)

        self.assertEqual(self.client.delete(f"/api/generations/{job_id}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/voices/{voice['id']}").status_code, 200)


    def test_04_text_cleaning_and_chunking(self) -> None:
        from apps.api.engine import clean_text_for_tts, split_text_chunks

        raw = "Universal उपाय सब के लिए होते हैं…\n• आपके planets का real impact\n✔ Practical remedies\n2026"
        cleaned_hi = clean_text_for_tts(raw, language="hi")
        self.assertNotIn("✔", cleaned_hi)
        self.assertNotIn("•", cleaned_hi)
        self.assertIn("दो हज़ार छब्बीस", cleaned_hi)

        cleaned_en = clean_text_for_tts("Success in 2026 ✔", language="en")
        self.assertIn("twenty twenty six", cleaned_en)
        self.assertNotIn("✔", cleaned_en)

        chunks = split_text_chunks(cleaned_hi, max_chars=320)
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()

