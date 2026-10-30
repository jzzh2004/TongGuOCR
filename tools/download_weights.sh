#!/usr/bin/env bash
# Download TongGuOCR model weights from Hugging Face Hub into weights/.
#
#   weights/ocr/                     full OCR model dir (model.safetensors +
#                                    configs + tokenizer)
#   weights/layout/finetune_infer.pth   layout detector checkpoint
#                                    (hdc_speed.py ships with the repo)
#
# Override the repository id with HF_REPO_ID if you use a mirror/another
# account. If huggingface.co is unreachable, set HF_ENDPOINT (e.g.
# https://hf-mirror.com). Alternatively, place a local copy of the weights
# at the same paths and skip this script (see README.md).
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-jzzh2004/TongGuOCR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" - "$ROOT" "$HF_REPO_ID" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    raise SystemExit("huggingface_hub is required; activate the environment created by tools/setup_env.sh.")

root = Path(sys.argv[1])
repo_id = sys.argv[2]
print(f"Downloading OCR model from {repo_id} -> {root / 'weights/ocr'}", flush=True)
snapshot_download(repo_id, local_dir=root / "weights/ocr", ignore_patterns=["layout/*"])

# The checkpoint lives at layout/finetune_infer.pth inside the repo; download
# into weights/ (not weights/layout/) so it lands at weights/layout/finetune_infer.pth.
print(f"Downloading layout checkpoint -> {root / 'weights/layout'}", flush=True)
hf_hub_download(repo_id, filename="layout/finetune_infer.pth", local_dir=root / "weights")

print(f"Done. Weights ready under {root / 'weights'}/{{ocr,layout}}.")
PY
