#!/usr/bin/env bash
# Serve the GPU playground (8B `production` model as default) on the H100.
# This is the GPU counterpart to `make serve`; see docs/RUNNING_ON_GPU.md.
#
# Usage:
#   scripts/serve_gpu.sh                 # foreground, port 8000
#   PORT=8080 scripts/serve_gpu.sh       # different port
#   tmux new-session -d -s freight 'scripts/serve_gpu.sh'   # durable (survives logout)
set -euo pipefail
cd "$(dirname "$0")/.."

# GPU path is opt-in; these do not affect the CPU `make demo` defaults.
export FREIGHT_GPU=1                 # bf16 CUDA inference
export TORCHDYNAMO_DISABLE=1         # force eager (skips the triton/Python.h build)
export FREIGHT_DEFAULT_MODEL=production   # serve the 8B by default
export FREIGHT_MAX_NEW_TOKENS=512    # headroom for long multi-leg confirmations
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

exec uv run uvicorn serving.app:app --host 0.0.0.0 --port "${PORT:-8000}"
