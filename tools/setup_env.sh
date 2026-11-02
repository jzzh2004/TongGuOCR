#!/usr/bin/env bash
# One-command setup of the `tongguocr` environment (layout + transformers OCR).
#
# The default pipeline needs only this environment (see README.md):
#   - THIS script builds `tongguocr`: python 3.10 + torch 2.10.0 (PyPI wheel
#     bundles the cu128 runtime) + transformers 5.3.0 (glm_ocr native support)
#     + mmcv-full 1.7.2 (built against torch 2.10).
#   - Optional setup_vllm_env.sh builds `tongguocr-vllm2` for vLLM serving, which
#     needs torch 2.13 and cannot coexist with the mmcv-locked torch 2.10.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-tongguocr}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda")}"
# Prefix-based env creation: robust even if an env with the same NAME already
# exists elsewhere in conda's envs_dirs. Override ENV_PREFIX to build elsewhere.
ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/$ENV_NAME}"
# Keep pip caches and build temp local to the checkout by default.
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.pip_cache}"
PIP_TMPDIR="${PIP_TMPDIR:-$ROOT/.pip_tmp}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# GPU arch for the mmcv source build. Empty = auto-detect from the installed
# torch + local GPU at build time. Only used when no prebuilt wheel matches.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
# Official package index by default; PIP_INDEX may select a Python package mirror.
# If Hugging Face is unreachable, use tools/download_weights.sh modelscope.
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple}"
# mmcv-full 1.7.2 has no PyPI wheel for torch 2.10: it is built from source
# here (5-60 min) and cached under wheels/. Drop a matching prebuilt wheel
# (linux x86_64, CPython 3.10, sm_86) into $WHEELS_DIR to skip the build.
WHEELS_DIR="${WHEELS_DIR:-$ROOT/wheels}"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -p "$ENV_PREFIX" python=3.10 -y
fi
conda activate "$ENV_PREFIX"
PIP="$ENV_PREFIX/bin/pip"
mkdir -p "$PIP_CACHE_DIR" "$PIP_TMPDIR" "$WHEELS_DIR"
export TMPDIR="$PIP_TMPDIR"
export PIP_CACHE_DIR

echo "==> [1/6] torch 2.10.0 (PyPI wheel bundles cu128 runtime; no vllm needed here)"
"$PIP" install -i "$PIP_INDEX" torch==2.10.0

echo "==> [2/6] transformers 5.3.0 (glm_ocr native support; >=5.3.0 required)"
"$PIP" install -i "$PIP_INDEX" transformers==5.3.0

echo "==> [3/6] mmcv-full 1.7.2"
# Newer setuptools removed pkg_resources, which mmcv 1.7.2's setup.py needs.
"$PIP" install -i "$PIP_INDEX" "setuptools==80.10.2" "Cython==3.0.12"
ARCH="${TORCH_CUDA_ARCH_LIST:-}"
if [[ -z "$ARCH" ]]; then
  ARCH="$("$ENV_PREFIX/bin/python" - <<'PYARCH'
import torch
try:
    cap = torch.cuda.get_device_capability(0)
    print(f"{cap[0]}.{cap[1]}")
except Exception:
    print("")
PYARCH
)"
fi
WHEEL="$(ls "$WHEELS_DIR"/mmcv_full-1.7.2-*.whl 2>/dev/null | head -n1 || true)"
# A prebuilt wheel found in $WHEELS_DIR is used only when it matches
# CPython 3.10 + sm_86 (Ampere); otherwise build from source (needs the CUDA
# toolkit at CUDA_HOME; ~5-60 min, cached into $WHEELS_DIR).
PYTAG="$("$ENV_PREFIX/bin/python" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
if [[ -n "$WHEEL" && "$ARCH" == "8.6" && "$PYTAG" == "cp310" ]]; then
  echo "    using prebuilt sm_86/cp310 wheel: $WHEEL"
else
  if [[ -z "$ARCH" ]]; then ARCH="8.6"; echo "    (no GPU visible; defaulting TORCH_CUDA_ARCH_LIST=8.6 — override if building for another arch)"; fi
  echo "    arch=$ARCH does not match the bundled sm_86/cp310 wheel; building from source..."
  (cd "$WHEELS_DIR" && \
   CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST="$ARCH" \
   MMCV_WITH_OPS=1 PIP_NO_BUILD_ISOLATION=1 MAX_JOBS="${MAX_JOBS:-16}" \
   "$PIP" wheel --no-build-isolation --no-deps -i "$PIP_INDEX" -w "$WHEELS_DIR" mmcv-full==1.7.2)
  WHEEL="$(ls "$WHEELS_DIR"/mmcv_full-1.7.2-*.whl | tail -n1)"
fi
"$PIP" install -i "$PIP_INDEX" "$WHEEL"

echo "==> [4/6] remaining pinned deps"
"$PIP" install -i "$PIP_INDEX" -r "$ROOT/requirements/inference.txt"

echo "==> [5/6] OpenMMLab compatibility patches (site-packages)"
"$ENV_PREFIX/bin/python" "$ROOT/tools/patch_openmmlab.py"

echo "==> [6/6] environment verification"
"$ENV_PREFIX/bin/python" "$ROOT/tools/verify_env.py"

echo "Done. Activate with: conda activate $ENV_PREFIX"
