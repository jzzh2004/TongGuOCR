# Evaluation

End-to-end evaluation of TongGuOCR: text-recognition metrics (AR / CR / NED /
F1 / BLEU) plus the RO-ED reading-order metric, from one entry script.

| File | Role |
| --- | --- |
| `run_eval.sh` | single entry point; runs all steps below |
| `make_eval_jsonl.py` | joins pipeline `texts/` with dataset GT into `predict.jsonl` |
| `transform.py` | converts predict.jsonl into the metric input format (`metric.json`) |
| `calculate_metrics.py` | computes all metrics, including standalone RO-ED; writes `res.txt` and `summary.json` |
| `eval_summary.py` | compatibility utility for collecting historical reports |
| `ro_ed/` | local text-line matching and RO-ED helpers (see `ro_ed/README.md`) |
| `tests/` | unit tests for the RO-ED protocol adapter |
| `requirements_eval.txt` | small CPU-only dependencies for all metrics |

## Metrics

AR / CR / Edit Distance / F1 / Precision / Recall / BLEU are computed by
`calculate_metrics.py` over character-level comparisons of full-page text
(all newline/spacing already normalized away by `transform.py`):

- **AR** (accuracy rate) and **CR** (character recognition rate) come from a
  Levenshtein backtrace between GT and prediction:
  `AR = (N - S - D - I) / N`, `CR = (N - S - D) / N` with N = GT length.
- **NED** is the `Edit Distance` line of `res.txt`: the mean over pages of
  `Levenshtein.distance(pred, gt) / max(len(pred), len(gt), 1)`. It is an
  independent metric, **not** `1 - CR`.
- **RO-ED** is the reading-order edit distance from `ro_ed/`, computed on
  the **token form** of the predictions (`texts_tokens/`, relation tokens
  kept) — the input form the paper's RO-ED values were defined on. The
  protocol splits predictions on newlines and the 24 relation tokens;
  `run_eval.sh` builds `predict_tokens.jsonl` from `texts_tokens/`
  automatically (and warns + falls back to the plain `texts/` form only if
  that directory is absent — plain-form RO-ED does NOT reproduce the paper
  numbers).

## Dataset formats

Both datasets are used in-place from their original roots (nothing is
redistributed here):

- **M5**: `<m5_root>/txt_line/<page_id>.txt`; each line
  `x1,y1,...,x4,y4,text` (coordinates first, text last).
- **MTHv2** (TKHMTH2200): `<mthv2_root>/<subset>/label_textline/<name>.txt`;
  each line `text,x1,y1,...,x4,y4` (text first). Staging page ids are
  `SUBSET__NAME` (e.g. `MTH1000__06-V005P0405`).

## Usage

```bash
# The inference env can run all metrics, including RO-ED.
conda activate tongguocr
python -m pip install -r evaluation/requirements_eval.txt

./evaluation/run_eval.sh \
  --dataset m5 \
  --run-dir  <pipeline output containing texts/ and texts_tokens/> \
  --dataset-root /path/to/M5 \
  --output-dir runs/eval/m5

# One Python entry point, useful for existing metric.json + raw predictions:
python evaluation/calculate_metrics.py \
  --input_path runs/eval/m5/metric.json \
  --ro-ed-input runs/eval/m5/predict_tokens.jsonl \
  --output_path runs/eval/m5/res.txt \
  --summary-json runs/eval/m5/summary.json \
  --ro-ed-output runs/eval/m5/ro_ed

# Add --skip-ro-ed to run_eval.sh for character metrics only.
# Add --ro-ed-only to calculate_metrics.py to score reading order only.
# Neither torch nor transformers is required for evaluation. A CPU-only env
# with requirements_eval.txt also works; no second environment is needed.
```

RO-ED needs no extra pipeline outputs: the pipeline already writes the
token form (`texts_tokens/`) next to `texts/`, so running on a plain
pipeline output directory reproduces the paper protocol.

## Expected values (paper runs)

Full-set values of the archived evaluation runs this toolchain reproduces:

| Dataset | Backend | AR | CR | NED |
| --- | --- | --- | --- | --- |
| M5 (1000 pages) | HF transformers | 93.74 | 94.45 | 6.17 |
| M5 (1000 pages) | vLLM | 93.80 | 94.49 | 6.12 |
| MTHv2 (800 pages) | HF transformers | 97.93 | 98.32 | 2.05 |
| MTHv2 (800 pages) | vLLM | 97.93 | 98.32 | 2.05 |

RO-ED on the archived paper runs: **0.0349** (M5) and **0.0151** (MTHv2).

Notes:

- The paper's M5 row (93.76 / 94.46 / 6.15) sits within 0.04 pt of each
  backend row above and within 0.01 pt of their exact means
  (93.77 / 94.47 / 6.145); it is not an arithmetic mean of any pair.
- RO-ED exact recomputation of the archived runs gives 0.0348767895… / 
  0.0150837504…; the paper reports them rounded as 3.49 / 1.51. Fresh
  end-to-end runs may deviate slightly (different generation runs); see
  `ro_ed/README.md` for the exact-reproduction procedure.
- Fresh-clone reproduction check (2026-10-28, different GPU, rebuilt envs):
  AR/CR/NED within 0.01 pt of the table above (M5 CR 94.44; MTHv2 identical
  except Precision 98.07), RO-ED 0.03463 / 0.01511 — i.e. 3.46 / 1.51.


## Standalone RO-ED validation

The OmniDocBench framework has been removed from this repository. Only its
text-matching decisions, normalization, and order-pairing logic were adapted
into local helpers, retaining their Apache-2.0 attribution and license.
`calculate_metrics.py` exposes `cal_ro_ed()` and writes RO-ED in the same
report as AR/CR/NED/F1/BLEU. Line-sensitive input remains separate from the
character-metric transform, which removes newlines by design.

The standalone implementation was checked on all 1800 archived paper pages:
GT order IDs, predicted order IDs, and per-page edit distances are identical
to the previous scorer. Aggregate values are exactly 0.0348767895414561
(M5) and 0.015083750445055579 (MTHv2), with no timeout fallbacks. These
checks also pass with the inference environment's newer NumPy/SciPy stack.
The old NumPy/Pandas pins and the extra evaluation environment are unnecessary.
