# Evaluation

[简体中文](README.zh-CN.md) · [Main guide](../README.md)

```bash
python -m pip install -r evaluation/requirements_eval.txt
bash evaluation/run_eval.sh \
  --dataset m5 \
  --run-dir output/ \
  --dataset-root /path/to/M5 \
  --output-dir eval_out/m5
```

Use `--dataset mthv2` for MTHv2. All metrics are written to one file:
**`eval_out/m5/metrics.json`**. Temporary evaluation inputs are removed after
completion. Evaluation uses `texts/*.txt`; each of the model's 24 relation
tokens is a newline. No additional prediction form is needed.
The inference environment or a CPU environment with `requirements_eval.txt`
is sufficient.

## Metrics

| Key | Definition | Better |
| --- | --- | --- |
| `AR` | Mean page accuracy: `(N − S − D − I) / N` | Higher |
| `CR` | Mean page correctness: `(N − S − D) / N` | Higher |
| `NED` | Mean `distance(pred, gt) / max(len(pred), len(gt), 1)` | Lower |
| `F1`, `Precision`, `Recall` | Mean page scores using character sets | Higher |
| `BLEU` | Mean character BLEU with smoothing | Higher |
| `RO-ED` | Mean normalized edit distance between matched line-order sequences | Lower |

`N` is GT length; `S`, `D`, and `I` count substitutions, deletions and
insertions using the historical edit-path tie-break. Character metrics
ignore spaces and newlines. RO-ED preserves line boundaries and matches
lines by content before comparing their order. A missing GT line adds a
deletion; an unmatched prediction has no GT order index. Empty order pairs
are excluded from the RO-ED mean. Details are in [the RO-ED note](ro_ed/README.md).

AR/CR/NED/F1/Precision/Recall/BLEU use percentage units in the JSON.
RO-ED is stored as a fraction; multiply it by 100 to compare with the paper.
The archived paper predictions give RO-ED 0.0348767895414561 for M5 and
0.015083750445055579 for MTHv2. Fresh inference can differ slightly.

## Dataset layout

- M5: `<root>/txt_line/<page_id>.txt`, one line per text line:
  `x1,y1,x2,y2,x3,y3,x4,y4,text`.
- MTHv2: `<root>/<subset>/label_textline/<name>.txt`, one line per text line:
  `text,x1,y1,x2,y2,x3,y3,x4,y4`. Prediction IDs use `<subset>__<name>`.

Missing GT causes an error. The datasets are not redistributed.
`--limit N` selects the first N pages; `--match-workers N` sets CPU workers.

## Existing predictions

For JSONL records `{page_id, predict, label}` with line breaks preserved:

```bash
python evaluation/calculate_metrics.py \
  --input_path predictions.jsonl --output_path metrics.json
```

The same command accepts legacy `metric.json` records with
`Answer_Simplified` and `Response`. RO-ED needs the original line boundaries;
line breaks already removed from an older input cannot be recovered.
