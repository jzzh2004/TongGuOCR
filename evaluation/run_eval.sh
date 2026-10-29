#!/usr/bin/env bash
# Usage: evaluation/run_eval.sh --dataset m5|mthv2 --run-dir OUTPUT \
#          --dataset-root DATASET --output-dir EVAL [--match-workers N] [--limit N]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DATASET="" RUN_DIR="" DATASET_ROOT="" OUTPUT_DIR=""
MATCH_WORKERS=12 LIMIT=""
METRIC_ARGS=()
die() { echo "run_eval.sh: error: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --skip-ro-ed) METRIC_ARGS+=(--skip-ro-ed); shift ;;
    --match-workers) MATCH_WORKERS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$DATASET" == m5 || "$DATASET" == mthv2 ]] || die "--dataset must be m5 or mthv2"
[[ -n "$RUN_DIR" && -n "$DATASET_ROOT" && -n "$OUTPUT_DIR" ]] || die "--run-dir/--dataset-root/--output-dir are required"
[[ -d "$RUN_DIR/texts" ]] || die "texts dir not found: $RUN_DIR/texts"
mkdir -p "$OUTPUT_DIR"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
ROOT_FLAG=--m5-root; [[ "$DATASET" == mthv2 ]] && ROOT_FLAG=--mthv2-root
LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
"$PYTHON" "$SCRIPT_DIR/make_eval_jsonl.py" --dataset "$DATASET" \
  --texts-dir "$RUN_DIR/texts" "$ROOT_FLAG" "$DATASET_ROOT" \
  --output "$WORK_DIR/predict.jsonl" "${LIMIT_ARGS[@]}"
"$PYTHON" "$SCRIPT_DIR/calculate_metrics.py" --input_path "$WORK_DIR/predict.jsonl" \
  --output_path "$OUTPUT_DIR/metrics.json" --match-workers "$MATCH_WORKERS" "${METRIC_ARGS[@]}"
echo "Metrics: $OUTPUT_DIR/metrics.json"
cat "$OUTPUT_DIR/metrics.json"
