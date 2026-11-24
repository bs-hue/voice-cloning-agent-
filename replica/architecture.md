# Architecture: SwaraLab (clean-room local rebuild of VoiceStudio's core cloning features)

## Stack

| layer | choice | why |
| --- | --- | --- |
| web | React + TypeScript + Vite + Tailwind CSS | Clean local browser UI (`localhost:5173`) with zero Electron packaging overhead on `E:` |
| api & engine | Python 3.11 + FastAPI + Uvicorn | Native runtime for PyTorch `2.6.0` and `chatterbox-tts==0.1.7` |
| voice model | `ResembleAI/chatterbox` (MIT, rev `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`) | Candidate engine pending Stage 1 feasibility test on RTX 3050 4 GB and CPU |
| audio tools | `ffmpeg` + `soundfile` / `torchaudio` | Audio validation and normalization to WAV |
| database | SQLite (`data/studio.db`) | Zero-setup local persistence for profiles, consent, and history |
| jobs | Single-worker queue + SSE stream | Prevents concurrent synthesis jobs and streams progress cleanly |
| storage | Local `E:` drive (`data/voices`, `data/outputs`, `data/cache`) | Directs configurable files/caches to `E:` (`C:` has only 8.64 GB free) |

## Schema

Tables: 3 (`voice_profiles`, `generation_jobs`, `app_settings`). Access rules: local single-user process bound to `127.0.0.1`. See `replica/schema.sql`.

## API

| method path | does | who | input | output | flow |
| --- | --- | --- | --- | --- | --- |
| GET /api/health | Server, GPU/CUDA, model, and E: disk status | UI | none | JSON status | F04 |
| GET /api/settings | Reads device preference and chunk settings | UI | none | settings JSON | F04 |
| PUT /api/settings | Updates `preferred_device` (`auto`/`cuda`/`cpu`) and `chunk_max_chars` | UI | JSON | updated settings | F04 |
| GET /api/voices | Lists saved voice profiles | UI | none | array of profiles | F01, F02 |
| POST /api/voices | Validates & normalizes reference audio, verifies consent, saves profile | UI | multipart form | created profile | F01 |
| GET /api/voices/{id}/audio | Streams reference WAV for preview | UI | path id | `audio/wav` | F01 |
| DELETE /api/voices/{id} | Deletes voice profile and reference file (keeps history snapshot) | UI | path id | `{ok: true}` | F02 |
| POST /api/generations | Queues a real synthesis job with supported params (`exaggeration`, `cfg_weight`, `temperature`) | UI | JSON | queued job | F01, F02 |
| GET /api/generations/{id}/events | SSE stream of synthesis progress | UI | path id | SSE progress/completion | F01, F02 |
| GET /api/generations | Lists generation history (optional search & language filter) | UI | query `q`, `language` | array of jobs | F03 |
| GET /api/generations/{id}/audio | Streams or downloads generated WAV (`?download=1`) | UI | path id | `audio/wav` | F01, F03 |
| DELETE /api/generations/{id} | Removes history entry and output WAV | UI | path id | `{ok: true}` | F03 |

## The parts that bite

- **4 GB VRAM ceiling:** `ChatterboxMultilingualTTS` loads `float32` by default; precision reduction is not officially supported via a parameter and must be tested in Stage 1.
- **C: drive space (8.64 GB free):** Direct `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `NPM_CONFIG_CACHE`, `PLAYWRIGHT_BROWSERS_PATH`, `TEMP`, and `TMP` to `E:`; disclose unavoidable OS/driver caches.
- **Supported controls only:** `ChatterboxMultilingualTTS.generate()` has no speed/pitch/seed parameters, so the UI exposes only supported parameters.

## Build order

1. Stage 1 Feasibility Gate: Isolated Python `E:` environment + English & Hindi GPU/CPU benchmark + measured report.
2. Stage 2 Vertical Slice & v1 UI: FastAPI + SQLite + React screens S01–S04.
3. Stage 3 Verification: End-to-end tests, `parity.py`, `contrast.py`, and `sweep.py`.
