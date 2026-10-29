#!/usr/bin/env bash
# Unified TongGuOCR evaluation. All metrics run in one Python environment.
# Usage: evaluation/run_eval.sh --dataset m5|mthv2 --run-dir OUTPUT \
#          --dataset-root DATASET --output-dir EVAL [--skip-ro-ed] [--match-workers N] [--limit N]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DATASET="" RUN_DIR="" DATASET_ROOT="" OUTPUT_DIR=""
SKIP_RO_ED=0 MATCH_WORKERS=12 LIMIT=""
die() { echo "run_eval.sh: error: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --skip-ro-ed) SKIP_RO_ED=1; shift ;;
    --match-workers) MATCH_WORKERS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$DATASET" == m5 || "$DATASET" == mthv2 ]] || die "--dataset must be m5 or mthv2"
[[ -n "$RUN_DIR" && -n "$DATASET_ROOT" && -n "$OUTPUT_DIR" ]] || die "--run-dir/--dataset-root/--output-dir are required"
[[ -d "$RUN_DIR/texts" ]] || die "texts dir not found: $RUN_DIR/texts"
mkdir -p "$OUTPUT_DIR"
ROOT_FLAG=--m5-root; [[ "$DATASET" == mthv2 ]] && ROOT_FLAG=--mthv2-root
LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")

echo '== 1/3 Build character-metric predictions'
"$PYTHON" "$SCRIPT_DIR/make_eval_jsonl.py" --dataset "$DATASET" \
  --texts-dir "$RUN_DIR/texts" "$ROOT_FLAG" "$DATASET_ROOT" \
  --output "$OUTPUT_DIR/predict.jsonl" "${LIMIT_ARGS[@]}"
RO_ARGS=()
if [[ "$SKIP_RO_ED" -eq 0 ]]; then
  if [[ -d "$RUN_DIR/texts_tokens" ]]; then
    # Refuse incomplete token directories instead of silently evaluating a subset.
    "$PYTHON" - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
plain = {p.name for p in (root / 'texts').glob('*.txt')}
tokens = {p.name for p in (root / 'texts_tokens').glob('*.txt')}
if plain != tokens:
    raise SystemExit(f'incomplete token-form outputs: {len(plain - tokens)} missing, {len(tokens - plain)} extra')
PY
    "$PYTHON" "$SCRIPT_DIR/make_eval_jsonl.py" --dataset "$DATASET" \
      --texts-dir "$RUN_DIR/texts_tokens" --keep-relation-tokens "$ROOT_FLAG" "$DATASET_ROOT" \
      --output "$OUTPUT_DIR/predict_tokens.jsonl" "${LIMIT_ARGS[@]}"
    RO_INPUT="$OUTPUT_DIR/predict_tokens.jsonl"
  else
    echo 'NOTE: token-form directory absent; plain-form RO-ED does not reproduce paper values.' >&2
    RO_INPUT="$OUTPUT_DIR/predict.jsonl"
  fi
  RO_ARGS=(--ro-ed-input "$RO_INPUT" --ro-ed-output "$OUTPUT_DIR/ro_ed" --match-workers "$MATCH_WORKERS")
fi

echo '== 2/3 Build historical character-metric input'
"$PYTHON" "$SCRIPT_DIR/transform.py" --input_path "$OUTPUT_DIR/predict.jsonl" --output_path "$OUTPUT_DIR/metric.json"
echo '== 3/3 Calculate AR / CR / NED / F1 / BLEU and standalone RO-ED'
"$PYTHON" "$SCRIPT_DIR/calculate_metrics.py" --input_path "$OUTPUT_DIR/metric.json" \
  --output_path "$OUTPUT_DIR/res.txt" --summary-json "$OUTPUT_DIR/summary.json" "${RO_ARGS[@]}"
cat "$OUTPUT_DIR/res.txt"
echo "Combined summary: $OUTPUT_DIR/summary.json"
