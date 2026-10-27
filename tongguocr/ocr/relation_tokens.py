"""Reading-order relation tokens of the TongGuOCR recognition model.

The final_ocr tokenizer carries 24 special relation tokens (ids 59282-59305):
eight single-axis tokens plus sixteen ``<horizontal|vertical>`` combinations.
They encode the model's predicted reading order between text lines and are
converted to newlines in every inference output, so the per-page text files
are naturally line-broken in reading order (the same unit boundaries the
RO-ED evaluation protocol derives from them).
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
# control/chat tokens that must never appear in recognition output. Decoding
# with skip_special_tokens=False (needed to see the relation tokens) lets them
# through, so finalize_ocr_text() removes them explicitly.
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


def relation_tokens_to_newlines(text: str) -> str:
    """Replace every relation token with a newline (longest match first)."""
    return RELATION_TOKEN_RE.sub("\n", text)


def finalize_ocr_text(text: str, extra_special_tokens: Iterable[str] = ()) -> str:
    """Postprocess raw decoded output: relation tokens -> newlines, control
    special tokens removed.

    ``extra_special_tokens`` lets callers pass ``tokenizer.all_special_tokens``
    (or the equivalent from the serving API) so any tokenizer-level additions
    are covered in addition to the frozen list above.
    """
    text = relation_tokens_to_newlines(text)
    text = _CONTROL_TOKEN_RE.sub("", text)
    for token in extra_special_tokens:
        if token and token not in RELATION_TOKENS:
            text = text.replace(token, "")
    return text
