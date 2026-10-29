#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Run the OCR server in the dedicated vLLM environment.
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
# A fixed KV-cache budget avoids reserving a fraction of a large GPU.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2147483648}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
DTYPE="${DTYPE:-bfloat16}"
# Up to 8192 output tokens plus the shared image/prompt budget.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
SEED="${SEED:-42}"
# Server-side pixel budget as a backstop; the client (vllm_backend.py)
# pre-resizes to the same budget before upload, so both layers agree.
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"image_max_pixels\": 1048576, \"image_min_pixels\": 1024}}"
# Eager execution is the validated default.
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
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
  "${EXTRA_ARGS[@]}" \
  --trust-remote-code
