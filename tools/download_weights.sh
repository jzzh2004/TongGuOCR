#!/usr/bin/env bash
# Download TongGuOCR model weights from Hugging Face Hub into weights/.
#
# The HF repository is not published yet; set HF_REPO_ID below (or via env)
# once it is live. Requires: pip install "huggingface_hub[cli]".
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-jzzh2004/TongGuOCR}"   # placeholder until published
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_DIR="$ROOT/weights/ocr"
LAYOUT_DIR="$ROOT/weights/layout"

command -v huggingface-cli >/dev/null 2>&1 || {
  echo "!! huggingface-cli not found. Install it first:  pip install 'huggingface_hub[cli]'"
  exit 1
}


echo "==> downloading OCR model (full snapshot) from $HF_REPO_ID -> $OCR_DIR"
huggingface-cli download "$HF_REPO_ID" \
  --local-dir "$OCR_DIR" --local-dir-use-symlinks False

echo "==> downloading layout checkpoint from ${HF_REPO_ID}/layout -> $LAYOUT_DIR"
huggingface-cli download "$HF_REPO_ID" \
  layout/finetune_infer.pth \
  --local-dir "$LAYOUT_DIR" --local-dir-use-symlinks False
# hdc_speed.py ships with the repository (it is code, not a weight).

echo "Done. Weights ready under $ROOT/weights/{ocr,layout}."
