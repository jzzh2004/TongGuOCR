"""Reading-order relation tokens of the TongGuOCR recognition model.

The final_ocr tokenizer carries 24 special relation tokens (ids 59282-59305):
eight single-axis tokens plus sixteen ``<horizontal|vertical>`` combinations.
During decoding the model emits them between text lines to steer generation
along its predicted reading path.

The pipeline writes TWO output forms per page:

- ``texts_tokens/<page>.txt`` — the token form: raw decode with control
  special tokens removed but the relation tokens kept. Its line boundaries
  are the model's own reading-order predictions; this is the input form the
  paper's RO-ED protocol was defined on (evaluation consumes this one).
- ``texts/<page>.txt`` — the plain form: relation tokens removed (byte-equal
  to a skip_special_tokens=True decode), then deterministically re-split to
  one output line per detected layout text line. AR/CR/NED/F1/BLEU are
  insensitive to this distinction (newlines are stripped by the metric
  chain); the archived benchmark texts/ files use this form.
"""
from __future__ import annotations

import re
from typing import Final, Iterable

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

# The remaining special tokens of the final_ocr tokenizer (ids 59246-59281):
# control/chat tokens that must never appear in recognition output. The
# backends decode with skip_special_tokens=False (the relation tokens must
# survive), so these are removed textually afterwards.
CONTROL_SPECIAL_TOKENS: Final = (
    "<|endoftext|>",
    "[MASK]",
    "[gMASK]",
    "[sMASK]",
    "<sop>",
    "<eop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|end_of_video|>",
    "<|begin_of_audio|>",
    "<|end_of_audio|>",
    "<|begin_of_transcription|>",
    "<|end_of_transcription|>",
    "<|code_prefix|>",
    "<|code_middle|>",
    "<|code_suffix|>",
    "/nothink",
)
_CONTROL_TOKEN_RE: Final = re.compile(
    "|".join(re.escape(token) for token in sorted(CONTROL_SPECIAL_TOKENS, key=len, reverse=True))
)


def strip_control_tokens(text: str, extra_special_tokens: Iterable[str] = ()) -> str:
    """Remove control/chat special tokens, KEEPING the relation tokens.

    This yields the token-form output (`texts_tokens/`) whose line boundaries
    are the model's own reading-order predictions — the input form the paper's
    RO-ED protocol was defined on.
    """
    text = _CONTROL_TOKEN_RE.sub("", text)
    for token in extra_special_tokens:
        if token and token not in RELATION_TOKENS:
            text = text.replace(token, "")
    return text


def remove_relation_tokens(text: str) -> str:
    """Delete every relation token (no separator inserted).

    Applied to the token-form text this reproduces, byte for byte, what a
    skip_special_tokens=True decode yields — the plain form (`texts/`) the
    released benchmark AR/CR/NED numbers were computed on.
    """
    return RELATION_TOKEN_RE.sub("", text)
