#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Serve from the dedicated `tongguocr-vllm2` env (python 3.12 + vllm 0.32.0).
# vllm 0.18 — formerly installed in the unified env; fresh builds exclude it — has a GlmOcr text-stack
# numeric bug: logits diverge from transformers by multiple nats on uncertain
# positions, generations fork mid-text and degenerate into repetition loops
# (see docs/vllm.md). vllm 0.32 fixes it and
# reproduces the transformers backend byte-exactly. 0.32 also needs torch 2.13,
# which the mmcv-locked unified env (torch 2.10) cannot host.
_CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda")}"
_VLLM_ENV_PY="$_CONDA_BASE/envs/tongguocr-vllm2/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$_VLLM_ENV_PY" ]]; then
    PYTHON_BIN="$_VLLM_ENV_PY"
  else
    # Fail fast instead of silently falling back to some arbitrary python3
    # on PATH (it may carry an incompatible vllm — or none at all).
    echo "!! tongguocr-vllm2 env python not found at: $_VLLM_ENV_PY" >&2
    echo "   Run tools/setup_vllm_env.sh first, or set CONDA_BASE/PYTHON_BIN." >&2
    exit 1
  fi
fi
MODEL_PATH="${MODEL_PATH:-$ROOT/weights/ocr}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-tonggu-ocr}"
GPU_IDS="${GPU_IDS:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
# Default 0.55, not 0.9 (2026-10-27): the inference pipeline runs layout on
# the SAME card (LAYOUT_DEVICE=cuda:0); MTH dewarp peaks 1-1.9GB/page and a
# 0.7 util server already OOM'd 4 layout pages on a 24G card .
# 0.55 leaves ample same-card headroom on both 24G and 48G GPUs. Override to
# 0.9 only for standalone serving with no same-card layout work.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
# --- Alignment with the transformers backend ---
# The unified inference profile is: bf16, max_tokens 8192, pixel budget
# [1024, 1048576], greedy decoding. See tongguocr/config.py
# (ocr.inference) — keep these defaults in sync.
# dtype bfloat16 matches the transformers backend load
# (torch_dtype=torch.bfloat16); "auto" would pick fp32
# from config.json and diverge from the transformers backend.
DTYPE="${DTYPE:-bfloat16}"
# 8192 generation tokens + ~1400 image tokens (1M pixel budget) + margin.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SEED="${SEED:-42}"
# Server-side pixel budget as a backstop; the client (vllm_backend.py)
# pre-resizes to the same budget before upload, so both layers agree.
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"image_max_pixels\": 1048576, \"image_min_pixels\": 1024}}"
# vllm 0.18's torch.compile path was incompatible with transformers 5.x
# (standalone_compile/FakeTensorMode). Kept eager by default on 0.32 as well:
# outputs are verified byte-exact vs the transformers backend in this mode;
# feel free to try ENFORCE_EAGER=0 for throughput after re-validating.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

EXTRA_ARGS=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DISABLE_VERSION_CHECK=1

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
  "${EXTRA_ARGS[@]}" \
  --trust-remote-code
