#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/Antigravity project/voice cloning agent"
CACHE="$ROOT/data/cache"
mkdir -p "$CACHE/uv-bin" "$CACHE/uv-python" "$CACHE/uv" "$CACHE/hf" "$CACHE/torch" "$CACHE/tmp" /opt/swaralab

export UV_INSTALL_DIR="$CACHE/uv-bin"
export UV_PYTHON_INSTALL_DIR="$CACHE/uv-python"
export UV_CACHE_DIR="$CACHE/uv"
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=2
export HF_HOME="$CACHE/hf"
export TORCH_HOME="$CACHE/torch"
export TMPDIR="$CACHE/tmp"
export PATH="$CACHE/uv-bin:$PATH"

if [ ! -x "$CACHE/uv-bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -x /opt/swaralab/.venv/bin/python ]; then
  uv python install 3.11
  uv venv --python 3.11 /opt/swaralab/.venv
fi

uv pip install --python /opt/swaralab/.venv/bin/python torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python /opt/swaralab/.venv/bin/python "git+https://github.com/resemble-ai/chatterbox.git@ab55923c27757c90cfcc407baa57dc25a7d25217"

/opt/swaralab/.venv/bin/python -c '
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
print("IMPORT OK:", "torch=", torch.__version__, "cuda=", torch.cuda.is_available(), "gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
'
