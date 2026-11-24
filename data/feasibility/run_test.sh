#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/Antigravity project/voice cloning agent"
export HF_HOME="$ROOT/data/cache/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT/data/cache/hf/hub"
export TORCH_HOME="$ROOT/data/cache/torch"
export TMPDIR="$ROOT/data/cache/tmp"
export PYTHONUNBUFFERED=1

/opt/swaralab/.venv/bin/python "$ROOT/apps/api/measure_bench_clean.py"
