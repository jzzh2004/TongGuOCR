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

# sha256 of the released artifacts (from the training-side model manifest).
declare -A OCR_SHA=(
  [model.safetensors]=5c7ddcebdd801c4c8db022b9791a85ef69f1db2e0ed673fce3d5e7f5f71ee3e5
  [config.json]=54daa949d855602da0aff992a566bc71c15a9bf890b2ce3bf617f70cdd9e95ee
  [tokenizer.json]=1b650d9e39f2a6a3fc97aec9f2be5f2e9b0571b38f4adf573f6d43e95ba981f6
  [tokenizer_config.json]=1b38c3d4d8b627fecb2820893eca17d9a1b59d1a1afbb52abbbbb507d18de11d
  [generation_config.json]=c508f7cd4fd470213212f0e2408e023984f9ddff23b118bbd79f0056bac6b6d3
  [processor_config.json]=f1a065a3eaba45a8b9a9ec1eecbfaba5e60993484edd15818c63c74c1cdbde00
  [chat_template.jinja]=062e7ee4cc8defa88a5938b5d456dc60366ffd80647f918946ee747bf09ddc7c
)
LAYOUT_PTH_SHA=08415f322550027500606a1be9b6c9fe0b220d8812affe2faecaf0c9d6d36e9e

echo "==> downloading OCR model (full snapshot) from $HF_REPO_ID -> $OCR_DIR"
huggingface-cli download "$HF_REPO_ID" \
  --local-dir "$OCR_DIR" --local-dir-use-symlinks False

echo "==> downloading layout checkpoint from ${HF_REPO_ID}/layout -> $LAYOUT_DIR"
huggingface-cli download "$HF_REPO_ID" \
  layout/finetune_infer.pth \
  --local-dir "$LAYOUT_DIR" --local-dir-use-symlinks False
# hdc_speed.py ships with the repository (it is code, not a weight).

echo "==> verifying sha256"
fail=0
for f in "${!OCR_SHA[@]}"; do
  p="$OCR_DIR/$f"
  if [[ ! -f "$p" ]]; then echo "!! MISSING $p"; fail=1; continue; fi
  got="$(sha256sum "$p" | cut -d' ' -f1)"
  if [[ "$got" != "${OCR_SHA[$f]}" ]]; then echo "!! SHA MISMATCH $f: $got"; fail=1; else echo "   ok  $f"; fi
done
p="$LAYOUT_DIR/finetune_infer.pth"
if [[ -f "$p" ]]; then
  got="$(sha256sum "$p" | cut -d' ' -f1)"
  if [[ "$got" != "$LAYOUT_PTH_SHA" ]]; then echo "!! SHA MISMATCH finetune_infer.pth: $got"; fail=1; else echo "   ok  layout/finetune_infer.pth"; fi
else
  # tolerate an alternate flat layout: snapshot may place it at $LAYOUT_DIR/layout/finetune_infer.pth
  alt="$LAYOUT_DIR/layout/finetune_infer.pth"
  if [[ -f "$alt" ]] && ! cmp -s "$alt" "$p" 2>/dev/null; then mv "$alt" "$p" 2>/dev/null || true; fi
  [[ -f "$p" ]] && echo "   ok  layout/finetune_infer.pth" || { echo "!! MISSING $p"; fail=1; }
fi

if [[ "$fail" != 0 ]]; then
  echo "!! verification FAILED"
  exit 1
fi
echo "Done. Weights ready under $ROOT/weights/{ocr,layout}."
