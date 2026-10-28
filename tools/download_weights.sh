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
# at the same paths and skip this script (see README FAQ).
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-jzzh2004/TongGuOCR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v huggingface-cli >/dev/null 2>&1 || {
  echo "!! huggingface-cli not found. Install it first:  pip install 'huggingface_hub[cli]'"
  exit 1
}


echo "==> downloading OCR model (full snapshot) from $HF_REPO_ID -> $ROOT/weights/ocr"
huggingface-cli download "$HF_REPO_ID" \
  --local-dir "$ROOT/weights/ocr" --local-dir-use-symlinks False

# The checkpoint lives at layout/finetune_infer.pth inside the repo; download
# into weights/ (not weights/layout/) so it lands at weights/layout/finetune_infer.pth.
echo "==> downloading layout checkpoint from ${HF_REPO_ID}/layout -> $ROOT/weights/layout"
huggingface-cli download "$HF_REPO_ID" \
  layout/finetune_infer.pth \
  --local-dir "$ROOT/weights" --local-dir-use-symlinks False

echo "Done. Weights ready under $ROOT/weights/{ocr,layout}."
