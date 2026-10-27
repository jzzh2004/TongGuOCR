# Vendored OmniDocBench

This directory contains a vendored subset of
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) (Apache-2.0, see
`LICENSE`), used by TongGuOCR's RO-ED reading-order evaluation
(`../../ro_ed/`).

- Upstream: https://github.com/opendatalab/OmniDocBench
- Pinned revision: `be7a44d371e43801fabd2e31e02976f106b6aa16`
  ("sync: align codebase with local v1.6", 2026-04-10)
- Vendored content: `src/`, `LICENSE`, `pyproject.toml`, `README.md` —
  byte-identical to the upstream revision; nothing else from the upstream
  repository (datasets, configs, demo data, results) is included.
- No modifications: TongGuOCR adapts inputs *around* the scorer
  (`src.dataset.end2end_dataset.md_tex_filter` is monkeypatched at runtime by
  `evaluation/ro_ed/ro_ed.py`); the vendored source itself is unchanged.

Runtime dependencies of the scorer are pinned in
`../../requirements_eval.txt`, taken from the environment of the paper's
archived RO-ED runs (python 3.10, no torch): Levenshtein 0.25.1, nltk 3.9.1,
apted 1.0.3, numpy 1.24.4, pandas 2.0.3, scipy 1.10.1, PyYAML 6.0.2,
evaluate 0.4.3, func-timeout 4.3.5, beautifulsoup4 4.11.1, Pillow 10.4.0,
pylatexenc 2.10.

If you use this evaluation, please cite OmniDocBench (see its README).
