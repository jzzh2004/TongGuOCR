# RO-ED

[简体中文](README.zh-CN.md) · [Evaluation](../README.md)

RO-ED matches recognized text lines to GT lines by content, then compares
their order. Predictions are split at newlines and the 24 relation tokens;
empty lines and ASCII/full-width spaces are removed. Structural tags are
handled by `protocol.py` without discarding recognized text.

Each prediction receives the order indices of its matched GT lines. Let
`P` be those indices in prediction order and `G` the sorted reference indices:

```text
page_RO_ED = LevenshteinDistance(G, P) / max(len(G), len(P))
```

The result is the unweighted page mean, excluding empty sequence pairs.
A missing GT line contributes a deletion. An unmatched prediction has no GT
order index and does not directly add an insertion; this metric should be
read alongside character-recognition metrics.

`calculate_metrics.py` computes RO-ED together with the other metrics and
writes one `metrics.json`. Line-preserving text and raw relation-token text
produce the same reading-order units. All 1800 archived paper pages were
checked: their matched order sequences and per-page scores are unchanged.

The matcher implementation and its license are retained locally; attribution
is in [third-party notices](../../THIRD_PARTY_NOTICES.md).
