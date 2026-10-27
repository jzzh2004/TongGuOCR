#!/usr/bin/env bash
# Create the vLLM serving env `tongguocr-vllm2`.
#
# Why a separate env: mmcv-full 1.7.2 (layout stack) locks torch 2.10, while
# vLLM 0.32 needs torch 2.13 — the two cannot coexist in one environment.
# Why vLLM 0.32 and not 0.18: vllm 0.18.0 has a GlmOcr text-stack numeric bug
# (generations diverge from the transformers backend and can degenerate into
# repetition loops; details in README.md "Backends"). vLLM 0.32 outputs were
# verified byte-exact against the transformers backend.
#
# Two install routes, selected automatically from the NVIDIA driver version:
#   - driver >= 580: plain `pip install vllm==0.32.0` (PyPI wheel, CUDA 13).
#   - driver 575.x (CUDA 12.9, e.g. no-sudo containers): the PyPI vllm/torch
#     wheels are CUDA-13-linked and cannot initialize CUDA or even import.
#     Preinstall torch 2.13.0+cu129 directly from download.pytorch.org and the
#     official vllm-0.32.0+cu129 wheel from wheels.vllm.ai (cached in wheels/).
# Override with VLLM_ROUTE=pypi|cu129.
#
# start_vllm_server.sh auto-selects this env's python when present.
set -euo pipefail

ENV_NAME="${ENV_NAME:-tongguocr-vllm2}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda")}"
# Prefix-based (mirrors setup_env.sh): robust to same-name envs elsewhere.
ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/$ENV_NAME}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
VLLM_INDEX="${VLLM_INDEX:-https://wheels.vllm.ai/0.32.0/cu129}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="${WHEELS_DIR:-$ROOT/wheels}"
PIP_TMPDIR="${PIP_TMPDIR:-$ROOT/.pip_tmp}"
mkdir -p "$PIP_TMPDIR" "$WHEELS_DIR"
export TMPDIR="$PIP_TMPDIR"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -p "$ENV_PREFIX" python=3.12 -y
fi
conda activate "$ENV_PREFIX"
PIP="$ENV_PREFIX/bin/pip"

ROUTE="${VLLM_ROUTE:-auto}"
if [[ "$ROUTE" == "auto" ]]; then
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1)"
  DRIVER_MAJOR="${DRIVER_MAJOR:-0}"
  if (( DRIVER_MAJOR >= 580 )); then ROUTE="pypi"; else ROUTE="cu129"; fi
  echo "driver=${DRIVER_MAJOR}.x -> route=$ROUTE"
fi

if [[ "$ROUTE" == "pypi" ]]; then
  echo "==> [1/2] vllm 0.32.0 + transformers 5.19.0 (PyPI, CUDA 13 wheels)"
  "$PIP" install -i "$PIP_INDEX" "vllm==0.32.0" "transformers==5.19.0"
else
  echo "==> [1/2] torch 2.13.0+cu129 trio via direct wheel URLs"
  # Resolve exact cp312 wheel filenames from the cu129 simple index, then pass
  # them as direct URL requirements so only $PIP_INDEX is consulted for deps.
  TRIO=""
  for pkg in torch torchvision torchaudio; do
    case "$pkg" in
      torch)       ver="2.13.0" ;;
      torchvision) ver="0.28.0" ;;
      torchaudio)  ver="2.11.0" ;;
    esac
    url_path="$(curl -fsSL "$TORCH_INDEX/$pkg/" | \
      grep -o "href=\"[^\"]*${ver}%2Bcu129-cp312-cp312-manylinux[^\"]*x86_64[^\"]*\.whl[^\"]*\"" | \
      head -n1 | sed 's/^href="//; s/"$//')"
    [[ -n "$url_path" ]] || { echo "!! could not find ${pkg} ${ver}+cu129 cp312 wheel in $TORCH_INDEX"; exit 1; }
    case "$url_path" in
      https://*) TRIO+=" $url_path" ;;
      *)         TRIO+=" $TORCH_INDEX/$url_path" ;;
    esac
    echo "    $pkg -> $url_path"
  done
  # shellcheck disable=SC2086
  "$PIP" install -i "$PIP_INDEX" $TRIO

  echo "==> [1b/2] vllm 0.32.0+cu129 + transformers 5.19.0"
  # The PyPI vllm 0.32.0 wheel is CUDA-13-linked; use the official +cu129
  # wheel from the vllm wheel index instead; cache it in wheels/.
  VLLM_WHEEL="$(ls "$WHEELS_DIR"/vllm-0.32.0+cu129-*.whl 2>/dev/null | head -n1 || true)"
  if [[ -z "$VLLM_WHEEL" ]]; then
    # Index pages nest: /0.32.0/cu129/ holds only `vllm/`; the wheel hrefs there
    # are relative ../../../<hash>/vllm-...whl resolving to the SITE ROOT.
    # Join by hand.
    url_path="$(curl -fsSL "$VLLM_INDEX/vllm/" | \
      grep -o 'href="[^"]*%2Bcu129-cp38-abi3-manylinux[^"]*x86_64\.whl"' | head -n1 | \
      sed 's/^href="//; s/"$//')"
    [[ -n "$url_path" ]] || { echo "!! no cu129 x86_64 vllm wheel listed at $VLLM_INDEX/vllm/"; exit 1; }
    case "$url_path" in
      https://*) VLLM_WHEEL_URL="$url_path" ;;
      *)         VLLM_WHEEL_URL="https://wheels.vllm.ai/${url_path#\.\./\.\./\.\./}" ;;
    esac
    echo "    downloading $VLLM_WHEEL_URL (~560MB)"
    curl -fL --retry 3 -C - -o "$WHEELS_DIR/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" "$VLLM_WHEEL_URL"
    VLLM_WHEEL="$(ls "$WHEELS_DIR"/vllm-0.32.0+cu129-*.whl | head -n1)"
  fi
  echo "    using wheel: $VLLM_WHEEL"
  "$PIP" install -i "$PIP_INDEX" "$VLLM_WHEEL" transformers==5.19.0

  echo "==> [1c/2] torchcodec cu13-soname shims"
  # vllm requires torchcodec>=0.14, but the cu129 channel tops out at 0.10 —
  # PyPI's torchcodec 0.17.0 needs libnvrtc.so.13 / libcudart.so.13. vllm
  # imports torchcodec at startup even though the OCR path never executes it.
  # Install the real cu13 libs + symlinks inside torchcodec.libs/, whose RPATH
  # resolves them — no LD_LIBRARY_PATH plumbing needed.
  "$PIP" install -i "$PIP_INDEX" "nvidia-cuda-nvrtc==13.4.92" "nvidia-cuda-runtime==13.4.92"
  SP="$ENV_PREFIX/lib/python3.12/site-packages"
  ln -sfn ../nvidia/cu13/lib/libnvrtc.so.13  "$SP/torchcodec.libs/libnvrtc.so.13"
  ln -sfn ../nvidia/cu13/lib/libcudart.so.13 "$SP/torchcodec.libs/libcudart.so.13"
fi

echo "==> [2/2] sanity: torch variant, CUDA on this driver, extension linkage, GlmOcr arch"
"$ENV_PREFIX/bin/python" - <<'EOFPY'
import torch
import vllm
print(f"torch {torch.__version__} vllm {vllm.__version__}")
assert torch.cuda.is_available(), "CUDA unavailable — wrong torch variant for this driver?"
import vllm._C_stable_libtorch  # noqa: F401  (catch linkage failure at setup time)
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "GlmOcrForConditionalGeneration" in archs, f"GlmOcr missing: {archs[:5]}..."
print(f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}; GlmOcrForConditionalGeneration registered OK")
EOFPY

echo "Done. Env prefix: $ENV_PREFIX"
echo "Start the server with: GPU_IDS=0 bash $ROOT/start_vllm_server.sh"
