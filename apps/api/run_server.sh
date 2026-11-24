#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/Antigravity project/voice cloning agent"
cd "$ROOT"

export HF_HOME="$ROOT/data/cache/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT/data/cache/hf/hub"
export TORCH_HOME="$ROOT/data/cache/torch"
export TMPDIR="$ROOT/data/cache/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

exec /opt/swaralab/.venv/bin/uvicorn apps.api.server:app --host 127.0.0.1 --port 8000

