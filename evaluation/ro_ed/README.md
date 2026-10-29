# RO-ED: Unified Reading-Order Edit Distance

## Scope

TongGuOCR computes RO-ED with local text-only matching and order-pairing
helpers. They are adapted from the Apache-2.0 OmniDocBench implementation;
see `../../THIRD_PARTY_NOTICES.md` and `LICENSE`. The OmniDocBench checkout,
dataset registry, general metric engine, Markdown parser, formula/table
scorers, and their dependencies are not used.

`../calculate_metrics.py` provides the public `cal_ro_ed()` function and
computes this metric alongside the text-recognition metrics.
The reported value is `reading_order / all / Edit_dist / ALL_page_avg`
(lower is better). The paper reports it multiplied by 100 (3.49 for M5,
1.51 for MTHv2).

## Text units

The reading-order unit in this project is a text line.

GT is split on one or more consecutive newline characters. Predictions are
split on one or more newlines, the 24 relation tokens, block-level structural
tags, Markdown fence lines, and every `</mm:think>` occurrence. Consecutive
boundaries are equivalent to one boundary because empty units are discarded.
The 24 relation tokens comprise eight single-axis tokens and sixteen
`<horizontal|vertical>` combinations.

ASCII space U+0020 and full-width space U+3000 are removed from both GT and
prediction. Unit edges are stripped and empty units are discarded. Real OCR
characters, including `#` and `|`, are retained.

Block tags such as `br`, `p`, `div`, `table`, `tr`, `td`, and `th` become line
boundaries. Inline tags such as `sup`, `underline`, `b`, and `span` are removed
while retaining their content. HTML comments and `<|ref|>/<|det|>` position
metadata are deleted without creating an extra line. Unknown angle tokens are
retained as text and counted in diagnostics. `</mm:think>` becomes a boundary
while all recognition content before and after it is retained.

The exact machine-readable rules live in `protocol.py`; each run directory
records `protocol.json` (spec + hash). The protocol hash (`ff195d9b…`) is recorded in each summary and pinned by
a regression test against the paper protocol.

## Content matching

For every page, each GT line is constructed as an OmniDocBench `text_block`
with an integer `order` starting at 1. Every normalized prediction line is
directly constructed as a plain `text_all`/`text_block` item in file order.
The adapter therefore bypasses `md_tex_filter()` semantics while retaining the
archived text-only matching rules.

OmniDocBench normalizes line content with `normalized_text()`, which converts
text and keeps word characters (Chinese characters, letters, digits, and other
regex word characters). The retained `quick_match` then associates GT and
prediction lines by normalized content. The matcher supports merged/truncated
line associations and uses edit-distance costs with Hungarian/fuzzy matching.
Timeout fallback counts are recorded in every `summary.json`.

## Reading-order score

After content matching, OmniDocBench represents every matched prediction line
by the order index or indices of its matched GT line(s). Sorting these
associations by prediction position gives the predicted GT-index sequence `P`.
Sorting all GT indices gives the reference sequence `G`.

For page `i`, the score is:

```text
RO_ED_i = LevenshteinDistance(G_i, P_i) / max(len(G_i), len(P_i))
```

`ALL_page_avg` is the unweighted arithmetic mean of `RO_ED_i` over evaluated
pages, so every page has equal weight. `edit_whole` is the length-weighted
aggregate (sum edit operations / sum upper lengths), and `edit_sample_avg` is
a sample-level mean retained as a diagnostic.

An unmatched GT line appears in `G` but not `P`, so it contributes a deletion.
An unmatched prediction has no GT order index and therefore does not directly
add an insertion to `P`, although it can affect content matching. This is a
property of the official OmniDocBench reading-order implementation and is an
important limitation when interpreting the metric as a pure ordering measure.

## Usage

Input is the token-form JSONL from `../make_eval_jsonl.py` (rows
`{page_id, predict, label}`). `../run_eval.sh` builds `predict_tokens.jsonl`
from the pipeline's `texts_tokens/` directory with `--keep-relation-tokens`;
the protocol then splits on both relation tokens and newlines. Usually
invoked through `../run_eval.sh`; directly:

```bash
python evaluation/calculate_metrics.py \
    --ro-ed-only \
    --ro-ed-input runs/eval/m5/predict_tokens.jsonl \
    --ro-ed-output runs/eval/m5/ro_ed \
    --match-workers 12
```

Install `../requirements_eval.txt` into the same environment used for the
other metrics. No torch, transformers, Pandas, or OmniDocBench is required.
`summary.json` stores the aggregate score and protocol hash; `per_page.json`
stores the matched GT/predicted sequences and score of every page. The old
`python -m evaluation.ro_ed.ro_ed` command remains a compatibility wrapper.

## Reproducing the paper numbers

The paper's RO-ED values were computed on the archived full-set prediction
runs: exact recomputation gives **0.0348767895414561** (M5, 1000 pages) and
**0.015083750445055579** (MTHv2, 800 pages). Re-running the whole pipeline
end-to-end and evaluating with this module reproduces those values up to the
generation nondeterminism of a fresh inference run. Relation tokens are
preserved in `texts_tokens/` and the evaluation input; this module converts
them to line boundaries. The layout-based line splitting in plain `texts/`
defines different units and does not reproduce the paper's RO-ED numbers.

## Validation

`../tests/test_protocol.py` covers repeated newlines, relation tokens, block
and inline tags, repeated `</mm:think>`, `<|ref|>/<|det|>` metadata, comments,
unknown tags, real `|/#` characters, space removal, Markdown-link unwrapping,
the literal-`||` regression (no Markdown table object is ever created), and
the protocol hash pin.

All 1800 archived M5/MTHv2 pages were checked against the previous framework:
GT order sequences, predicted order sequences, per-page scores, and both
aggregate paper values are identical (zero timeout fallbacks).
