"""Reading-order relation tokens of the TongGuOCR recognition model.

The final_ocr tokenizer carries 24 special relation tokens (ids 59282-59305):
eight single-axis tokens plus sixteen ``<horizontal|vertical>`` combinations.
During decoding the model emits them between text lines to steer generation
along its predicted reading path.

The released inference pipeline decodes with ``skip_special_tokens=True``
(both backends), so these tokens are stripped from the raw output; the line
structure of ``texts/`` is then recovered by the pipeline's line-count
re-split (one output line per detected layout text line). This is exactly
the semantics the released benchmark numbers were produced under. The
RO-ED evaluation protocol additionally accepts raw token streams: it splits
predictions on newlines *or* any of the tokens below, treating the two forms
as equivalent (see evaluation/ro_ed/protocol.py).
"""
from __future__ import annotations

import re
from typing import Final

X_RELATIONS = ("left", "right", "far_left", "far_right")
Y_RELATIONS = ("up", "down", "far_up", "far_down")
RELATION_TOKENS: Final = tuple(
    [f"<{label}>" for label in (*X_RELATIONS, *Y_RELATIONS)]
    + [f"<{x}|{y}>" for x in X_RELATIONS for y in Y_RELATIONS]
)
# longest first so that "<left|up>" is never matched as "<left>" + "|up>"
RELATION_TOKEN_RE: Final = re.compile(
    "|".join(re.escape(token) for token in sorted(RELATION_TOKENS, key=len, reverse=True))
)


def relation_tokens_to_newlines(text: str) -> str:
    """Replace every relation token with a newline (longest match first).

    Not used by the released pipeline (which strips the tokens before this
    stage); kept as the canonical mapping for consumers of raw token streams.
    """
    return RELATION_TOKEN_RE.sub("\n", text)
