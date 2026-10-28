# Evaluation

End-to-end evaluation of TongGuOCR: text-recognition metrics (AR / CR / NED /
F1 / BLEU) plus the RO-ED reading-order metric, from one entry script.

| File | Role |
| --- | --- |
| `run_eval.sh` | single entry point; runs all steps below |
| `make_eval_jsonl.py` | joins pipeline `texts/` with dataset GT into `predict.jsonl` |
| `transform.py` | converts predict.jsonl into the metric input format (`metric.json`) |
| `calculate_metrics.py` | computes the metrics; writes the `res.txt` report |
| `eval_summary.py` | collects `res.txt` + RO-ED result into `summary.json` |
| `ro_ed/` | RO-ED reading-order evaluation (see `ro_ed/README.md`) |
| `third_party/OmniDocBench/` | vendored OmniDocBench scorer used by RO-ED |
| `tests/` | unit tests for the RO-ED protocol adapter |
| `requirements_eval.txt` | python deps for the RO-ED part (torch-free) |

## Metrics

AR / CR / Edit Distance / F1 / Precision / Recall / BLEU are computed by
`calculate_metrics.py` over character-level comparisons of full-page text
(all newline/spacing already normalized away by `transform.py`):

- **AR** (accuracy rate) and **CR** (character recognition rate) come from a
  Levenshtein backtrace between GT and prediction:
  `AR = (N - S - D - I) / N`, `CR = (N - S - D) / N` with N = GT length.
- **NED** is the `Edit Distance` line of `res.txt`: the mean over pages of
  `nltk.edit_distance(pred, gt) / max(len(pred), len(gt))`. It is an
  independent metric, **not** `1 - CR`.
- **RO-ED** is the reading-order edit distance from `ro_ed/`, computed on
  the line structure of the predictions: the model emits 24 reading-order
  relation tokens (e.g. `<left|up>`), which the inference pipeline converts
  to newlines, so the newlines in `texts/` encode reading order.

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
# AR/CR/NED/F1/BLEU need the inference env (calculate_metrics imports torch):
conda activate tongguocr

./evaluation/run_eval.sh \
  --dataset m5 \
  --run-dir  <pipeline output containing texts/> \
  --dataset-root /path/to/M5 \
  --output-dir runs/eval/m5

# RO-ED step: torch-free deps in a SEPARATE env (its pinned numpy/pandas
# must not downgrade the inference stack), selected via RO_ED_PYTHON:
conda create -n tongguocr-eval python=3.10 -y
conda run -n tongguocr-eval pip install -r evaluation/requirements_eval.txt
RO_ED_PYTHON="$(conda info --base)/envs/tongguocr-eval/bin/python" \
./evaluation/run_eval.sh ...          # add --skip-ro-ed to skip step 4
```

RO-ED needs no extra pipeline outputs: the relation tokens are already
converted to newlines inside `texts/` (see README.md), so running
on a plain pipeline output directory reproduces the paper protocol.

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
