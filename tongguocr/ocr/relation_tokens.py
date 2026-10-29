"""Reading-order relation tokens of the TongGuOCR recognition model.

The final_ocr tokenizer carries 24 special relation tokens (ids 59282-59305):
eight single-axis tokens plus sixteen ``<horizontal|vertical>`` combinations.
During decoding the model emits them between text lines to steer generation
along its predicted reading path.

The pipeline replaces each relation token with a newline in ``texts/``.
Line boundaries come from the recognition model. Raw relation tokens are
retained in the per-block JSON record for inspection.
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
    """Remove control/chat special tokens, keeping relation tokens."""
    text = _CONTROL_TOKEN_RE.sub("", text)
    for token in extra_special_tokens:
        if token and token not in RELATION_TOKENS:
            text = text.replace(token, "")
    return text


def remove_relation_tokens(text: str) -> str:
    """Remove relation tokens for whitespace-insensitive character scoring."""
    return RELATION_TOKEN_RE.sub("", text)


def relation_tokens_to_newlines(text: str) -> str:
    """Replace exactly the model's 24 relation tokens with line boundaries."""
    return RELATION_TOKEN_RE.sub("\n", text.replace("\r\n", "\n").replace("\r", "\n"))
