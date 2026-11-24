# SwaraLab - Voice Cloning Agent

A clean-room, privacy-first local voice cloning and speech synthesis studio built with FastAPI, PyTorch, React, and Chatterbox Multilingual TTS.

## Features

- **Local Voice Cloning**: Clone voices from reference WAV samples with strict consent workflows.
- **Multilingual Support**: High-fidelity English and Hindi voice synthesis with phonetic Devanagari transliteration.
- **Parameter Controls**: Fine-tune prosody and similarity with `exaggeration`, `cfg_weight`, and `temperature`.
- **Modern Web Studio**: React + Vite + Tailwind CSS frontend interface with real-time SSE progress streaming.
- **SQLite Persistence**: Local storage for voice profiles, history, and generation metadata.

## Project Structure

```
├── apps/
│   ├── api/          # FastAPI backend, TTS engine & audio utilities
│   └── web/          # React + Vite studio interface
├── data/
│   ├── feasibility/  # Verification scripts & benchmarks
│   ├── voices/       # Reference voice profiles
│   └── outputs/      # Generated audio files
├── replica/          # Architecture, specs & design tokens
└── start_swaralab.ps1 # Quick launcher script
```

## Quick Start

### 1. Web Frontend
```bash
cd apps/web
npm install
npm run build
```

### 2. Backend & Engine
```bash
pip install -r requirements.txt # or install fastapi uvicorn soundfile torch
python apps/api/server.py
```

### 3. Launch with Script
```powershell
.\start_swaralab.ps1 -OpenBrowser
```
Dashboard will be available at `http://127.0.0.1:8000`.
