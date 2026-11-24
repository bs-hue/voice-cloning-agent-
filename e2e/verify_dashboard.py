"""End-to-end browser verification of the built SwaraLab dashboard (desktop + mobile)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import urllib.request
import wave
import math
import struct
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SCREENS_DIR = ROOT / "replica" / "clone-screens"
CHROME_EXE = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PORT = 8011
BASE_URL = f"http://127.0.0.1:{PORT}"


def write_test_wav(path: Path, duration_sec: float = 4.0, sr: int = 24000) -> None:
    n = int(sr * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            val = int(32767 * 0.15 * math.sin(2 * math.pi * 220 * i / sr))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))


def wait_for_health(timeout_sec: float = 30.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Server did not become healthy in time.")


def main() -> None:
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=str(ROOT / "data" / "cache" / "tmp")))
    test_wav = tmp_dir / "sample_voice.wav"
    write_test_wav(test_wav)

    wsl_tmp = f"/mnt/e/Antigravity project/voice cloning agent/data/cache/tmp/{tmp_dir.name}"
    srv_proc = subprocess.Popen(
        [
            "wsl",
            "--cd",
            str(ROOT),
            "-d",
            "SwaraLabWSL",
            "-e",
            "env",
            f"SWARALAB_DATA_DIR={wsl_tmp}",
            "SWARALAB_ENGINE_MODE=mock",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "/opt/swaralab/.venv/bin/uvicorn",
            "apps.api.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for_health()
        console_errors: list[str] = []
        page_errors: list[str] = []
        server_5xx: list[str] = []

        with sync_playwright() as p:
            launch_kwargs = {"headless": True}
            if CHROME_EXE.exists():
                launch_kwargs["executable_path"] = str(CHROME_EXE)
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.on(
                "response",
                lambda r: server_5xx.append(f"{r.status} {r.url}") if r.status >= 500 else None,
            )

            # 1. Studio empty state
            page.goto(BASE_URL, wait_until="networkidle")
            page.get_by_role("heading", name="Speech Studio").wait_for()
            page.get_by_role("link", name="Add Authorized Voice Profile").click()

            # 2. Voice Profiles: consent error + valid upload
            page.get_by_role("heading", name="Voice Profiles").wait_for()
            page.locator('input[type="file"]').set_input_files(str(test_wav))
            page.get_by_label("Profile name").fill("E2E Speaker")
            page.get_by_role("button", name="Save Voice Profile").click()
            page.get_by_text("Confirm speaker authorization before saving a voice profile.").wait_for()

            page.locator("#voice-consent").check()
            page.get_by_label("Consent note (optional)").fill("Playwright verified consent")
            page.get_by_role("button", name="Save Voice Profile").click()
            page.get_by_role("heading", name="E2E Speaker").wait_for()
            page.screenshot(path=str(SCREENS_DIR / "S02.png"), full_page=True)

            # 3. Studio synthesis flow (S01)
            page.get_by_role("button", name="Use in Studio").click()
            page.get_by_role("heading", name="Speech Studio").wait_for()
            page.get_by_role("button", name="Sample 1").click()
            page.get_by_role("button", name="Generate Speech").click()
            page.get_by_text("Ready (").wait_for(timeout=15000)
            page.get_by_role("link", name="Download WAV").wait_for(timeout=10000)
            page.screenshot(path=str(SCREENS_DIR / "S01.png"), full_page=True)

            # 4. History search & filter (S03)
            page.get_by_role("link", name="History", exact=True).click()
            page.get_by_role("heading", name="Generation History").wait_for()
            page.get_by_text("E2E Speaker").wait_for()
            page.screenshot(path=str(SCREENS_DIR / "S03.png"), full_page=True)

            page.get_by_role("button", name="Hindi", exact=True).click()
            page.get_by_text("No generated clips match your search or language filter.").wait_for()
            page.get_by_role("button", name="All languages").click()
            page.get_by_text("E2E Speaker").wait_for()

            # 5. Engine & Storage Dashboard (S04)
            page.get_by_role("link", name="Engine & Storage", exact=True).click()
            page.get_by_role("heading", name="Engine & Storage Dashboard").wait_for()
            page.get_by_text("t3_mtl23ls_v2.safetensors", exact=True).wait_for()
            page.get_by_text("Stage 1 Measured Benchmark Summary").wait_for()
            page.screenshot(path=str(SCREENS_DIR / "S04.png"), full_page=True)

            # 6. Mobile viewport check (390px)
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(BASE_URL, wait_until="networkidle")
            page.get_by_role("heading", name="Speech Studio").wait_for()
            overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
            page.screenshot(path=str(SCREENS_DIR / "S01_mobile.png"), full_page=True)

            browser.close()

        assert not console_errors, f"Console errors: {console_errors}"
        assert not page_errors, f"Page errors: {page_errors}"
        assert not server_5xx, f"5xx responses: {server_5xx}"
        assert not overflow, "Horizontal overflow detected at 390px mobile width!"
        print("E2E_ALL_PASSED desktop=1440x900 mobile=390x844 overflow=False")
    finally:
        srv_proc.terminate()
        try:
            srv_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv_proc.kill()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

