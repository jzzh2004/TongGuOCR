#!/usr/bin/env bash
# Download TongGuOCR model weights from Hugging Face or ModelScope into weights/.
#
#   weights/ocr/                     full OCR model dir (model.safetensors +
#                                    configs + tokenizer)
#   weights/layout/finetune_infer.pth   layout detector checkpoint
#                                    (hdc_speed.py ships with the repo)
#
# Usage: bash tools/download_weights.sh [huggingface|modelscope]
# If Hugging Face is unreachable, select ModelScope International.
# Private repositories require HF_TOKEN or MODELSCOPE_API_TOKEN respectively.
# Alternatively, place a local copy at the same paths (see README.md).
set -euo pipefail

SOURCE="${1:-huggingface}"
if [[ "$#" -gt 1 || ( "$SOURCE" != "huggingface" && "$SOURCE" != "modelscope" ) ]]; then
  echo "Usage: bash tools/download_weights.sh [huggingface|modelscope]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" - "$ROOT" "$SOURCE" <<'PY'
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = sys.argv[2]
ocr_files = [
    "model.safetensors", "config.json", "generation_config.json",
    "processor_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja",
]
layout_file = "layout/finetune_infer.pth"
files = ocr_files + [layout_file]

if source == "huggingface":
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Activate the environment created by tools/setup_env.sh; huggingface_hub is required.")
    repo_id = os.environ.get("HF_REPO_ID", "jzzh2004/TongGuOCR")
    print(f"Downloading from Hugging Face: {repo_id}", flush=True)
    snapshot = Path(snapshot_download(repo_id, allow_patterns=files, token=os.environ.get("HF_TOKEN")))
else:
    # This release is hosted on the international site, not modelscope.cn.
    os.environ["MODELSCOPE_DOMAIN"] = "modelscope.ai"
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise SystemExit("ModelScope download requires: python -m pip install modelscope")
    repo_id = os.environ.get("MODELSCOPE_REPO_ID", "jzzh2004/TongGuOCR")
    print(f"Downloading from ModelScope International: {repo_id}", flush=True)
    snapshot = Path(snapshot_download(repo_id, allow_patterns=files,
                                     token=os.environ.get("MODELSCOPE_API_TOKEN")))

missing = [name for name in files if not (snapshot / name).is_file()]
if missing:
    raise SystemExit("Downloaded snapshot is incomplete: " + ", ".join(missing))
ocr_dir = root / "weights/ocr"
layout_dir = root / "weights/layout"
ocr_dir.mkdir(parents=True, exist_ok=True)
layout_dir.mkdir(parents=True, exist_ok=True)
for name in ocr_files:
    shutil.copy2(snapshot / name, ocr_dir / name)
shutil.copy2(snapshot / layout_file, layout_dir / "finetune_infer.pth")

print(f"Done. Weights ready under {root / 'weights'}/{{ocr,layout}}.")
PY
