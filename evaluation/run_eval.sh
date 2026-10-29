#!/usr/bin/env bash
# Single entry point for TongGuOCR evaluation.
#
#   AR / CR / NED / F1 / BLEU : transform.py + calculate_metrics.py
#                               (needs the inference env, e.g. `conda activate tongguocr`)
#   RO-ED                     : evaluation/ro_ed reading-order protocol
#                               (needs the deps in evaluation/requirements_eval.txt;
#                               works in a torch-free environment)
#
# Usage:
#   ./evaluation/run_eval.sh --dataset m5|mthv2 \
#       --run-dir <pipeline output dir containing texts/> \
#       --dataset-root <path to M5 or MTHv2 dataset root> \
#       --output-dir <dir for predict.jsonl / metric.json / res.txt / summary.json> \
#       [--skip-ro-ed] [--match-workers N] [--limit N]
#
# Reading order enters the evaluation through the newline structure of the
# pipeline texts/ files: the model's relation tokens are converted to
# newlines at inference time (see README.md), so plain texts/ is
# all this script needs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
# RO-ED runs in its own torch-free environment (requirements_eval.txt).
# Defaults to $PYTHON; override when your inference env lacks the pins:
#   PYTHON=<inference python> RO_ED_PYTHON=<eval python> ./evaluation/run_eval.sh ...
RO_ED_PYTHON="${RO_ED_PYTHON:-$PYTHON}"

usage() { grep '^#' "$0" | sed -n '3,20p' | sed 's/^# \{0,1\}//'; }
die() { echo "run_eval.sh: error: $*" >&2; exit 1; }

DATASET="" RUN_DIR="" DATASET_ROOT="" OUTPUT_DIR=""
SKIP_RO_ED=0 MATCH_WORKERS=12 LIMIT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --skip-ro-ed) SKIP_RO_ED=1; shift ;;
    --match-workers) MATCH_WORKERS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$DATASET" == "m5" || "$DATASET" == "mthv2" ]] || die "--dataset must be m5 or mthv2"
[[ -n "$RUN_DIR" && -n "$DATASET_ROOT" && -n "$OUTPUT_DIR" ]] || die "--run-dir/--dataset-root/--output-dir are required"
TEXTS_DIR="$RUN_DIR/texts"
[[ -d "$TEXTS_DIR" ]] || die "texts dir not found: $TEXTS_DIR"

OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
ROOT_FLAG="--m5-root"; [[ "$DATASET" == "mthv2" ]] && ROOT_FLAG="--mthv2-root"

PREDICT="$OUTPUT_DIR/predict.jsonl"
METRIC="$OUTPUT_DIR/metric.json"
RES="$OUTPUT_DIR/res.txt"
SUMMARY="$OUTPUT_DIR/summary.json"

LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")

echo "== 1/4 make_eval_jsonl -> predict.jsonl"
"$PYTHON" "$SCRIPT_DIR/make_eval_jsonl.py" \
  --dataset "$DATASET" \
  --texts-dir "$TEXTS_DIR" \
  "$ROOT_FLAG" "$DATASET_ROOT" \
  --output "$PREDICT" \
  "${LIMIT_ARGS[@]}"

echo "== 2/4 transform -> metric.json"
"$PYTHON" "$SCRIPT_DIR/transform.py" --input_path "$PREDICT" --output_path "$METRIC"

echo "== 3/4 calculate_metrics -> res.txt"
"$PYTHON" "$SCRIPT_DIR/calculate_metrics.py" --input_path "$METRIC" --output_path "$RES"

RO_ED_SUMMARY=""
if [[ $SKIP_RO_ED -eq 0 ]]; then
  echo "== 4/4 RO-ED (reading-order protocol)"
  [[ -d "$SCRIPT_DIR/ro_ed" ]] || die "evaluation/ro_ed not found (incomplete checkout?)"
  ODB_ROOT="$REPO_ROOT/evaluation/third_party/OmniDocBench"
  if [[ ! -d "$ODB_ROOT" ]]; then
    die "vendored OmniDocBench not found at evaluation/third_party/OmniDocBench"
  fi
  if ! "$RO_ED_PYTHON" -c "import Levenshtein, apted, nltk" 2>/dev/null; then
    echo "run_eval.sh: RO-ED python deps missing; install them with:" >&2
    echo "  ${PYTHON} -m pip install -r evaluation/requirements_eval.txt" >&2
    exit 1
  fi
  # The paper's RO-ED protocol is defined on the token-form predictions
  # (texts_tokens/: relation tokens kept). Prefer them when present.
  TOKENS_DIR="$RUN_DIR/texts_tokens"
  RO_ED_INPUT="$PREDICT"
  if [[ -d "$TOKENS_DIR" ]]; then
    RO_ED_INPUT="$OUTPUT_DIR/predict_tokens.jsonl"
    echo "  building token-form predictions from $TOKENS_DIR"
    "$PYTHON" "$SCRIPT_DIR/make_eval_jsonl.py" \
      --dataset "$DATASET" \
      --texts-dir "$TOKENS_DIR" \
      --keep-relation-tokens \
      "$ROOT_FLAG" "$DATASET_ROOT" \
      --output "$RO_ED_INPUT" \
      "${LIMIT_ARGS[@]}"
  else
    echo "  NOTE: $TOKENS_DIR not found — falling back to plain texts/ form." >&2
    echo "  RO-ED on the plain form does NOT reproduce the paper numbers" >&2
    echo "  (the protocol expects the model's relation-token boundaries)." >&2
  fi
  RO_ED_DIR="$OUTPUT_DIR/ro_ed"
  (cd "$REPO_ROOT" && "$RO_ED_PYTHON" -m evaluation.ro_ed.ro_ed \
    --predict-jsonl "$RO_ED_INPUT" \
    --work-dir "$RO_ED_DIR" \
    --omnidocbench-root "$ODB_ROOT" \
    --workers "$MATCH_WORKERS")
  RO_ED_SUMMARY="$RO_ED_DIR/summary.json"
  [[ -f "$RO_ED_SUMMARY" ]] || die "RO-ED did not produce $RO_ED_SUMMARY"
else
  echo "== 4/4 RO-ED skipped (--skip-ro-ed)"
fi

if [[ -n "$RO_ED_SUMMARY" ]]; then
  "$PYTHON" "$SCRIPT_DIR/eval_summary.py" --res-txt "$RES" --ro-ed-summary "$RO_ED_SUMMARY" --output "$SUMMARY"
else
  "$PYTHON" "$SCRIPT_DIR/eval_summary.py" --res-txt "$RES" --output "$SUMMARY"
fi
echo "Evaluation report: $RES"
[[ -n "$RO_ED_SUMMARY" ]] && echo "RO-ED summary:    $RO_ED_SUMMARY"
echo "Combined summary: $SUMMARY"
