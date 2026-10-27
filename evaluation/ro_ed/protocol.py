from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


PROTOCOL_ID = "plain_text_lines_v1"
PROTOCOL_REVISION = 2
BOUNDARY = "\x1e"
SPACE_CHARS = " \u3000"

X_RELATIONS = ("left", "right", "far_left", "far_right")
Y_RELATIONS = ("up", "down", "far_up", "far_down")
RELATION_TOKENS = tuple(
    [f"<{label}>" for label in (*X_RELATIONS, *Y_RELATIONS)]
    + [f"<{x}|{y}>" for x in X_RELATIONS for y in Y_RELATIONS]
)

RELATION_RE = re.compile(
    "|".join(re.escape(token) for token in sorted(RELATION_TOKENS, key=len, reverse=True))
)
NEWLINE_RE = re.compile(r"(?:\r\n|\r|\n)+")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
THINK_END_RE = re.compile(r"</mm:think>", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")
FENCE_LINE_RE = re.compile(r"(?m)^\s*```[^\n]*$")

BLOCK_TAG_NAMES = (
    "br",
    "n",
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "li",
    "ul",
    "ol",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "caption",
    "colgroup",
    "col",
)
INLINE_TAG_NAMES = (
    "sup",
    "sub",
    "underline",
    "u",
    "b",
    "strong",
    "i",
    "em",
    "small",
    "span",
    "g",
    "t",
    "html",
    "body",
)


def _tag_re(names: tuple[str, ...]) -> re.Pattern[str]:
    joined = "|".join(re.escape(name) for name in names)
    return re.compile(rf"<\s*/?\s*(?:{joined})\b[^>]*>", re.IGNORECASE)


BLOCK_TAG_RE = _tag_re(BLOCK_TAG_NAMES)
INLINE_TAG_RE = _tag_re(INLINE_TAG_NAMES)
MALFORMED_EMPTY_TAG_RE = re.compile(r"<\s*/\s*>")
UNKNOWN_ANGLE_RE = re.compile(r"<[^<>\n]{1,100}>")

REF_BLOCK_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|>", re.DOTALL | re.IGNORECASE)
DET_BLOCK_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL | re.IGNORECASE)
UNCLOSED_DET_RE = re.compile(r"<\|det\|>.*?(?=<\|ref\|>|$)", re.DOTALL | re.IGNORECASE)
PROVIDER_TOKEN_RE = re.compile(r"<\|/?(?:ref|det)\|>", re.IGNORECASE)
BEGIN_SENTENCE_TOKEN_RE = re.compile(r"<｜begin▁of▁sentence｜>")


@dataclass
class PreprocessStats:
    raw_chars: int = 0
    normalized_chars: int = 0
    unit_count: int = 0
    removed_spaces: int = 0
    relation_boundaries: int = 0
    newline_boundaries: int = 0
    block_tag_boundaries: int = 0
    inline_tags_removed: int = 0
    think_boundaries: int = 0
    comments_removed: int = 0
    ref_blocks_removed: int = 0
    det_blocks_removed: int = 0
    provider_tokens_removed: int = 0
    begin_tokens_removed: int = 0
    markdown_links_unwrapped: int = 0
    fence_lines_removed: int = 0
    unknown_angle_tokens: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["unknown_angle_tokens"] = dict(sorted(self.unknown_angle_tokens.items()))
        result["unknown_angle_token_count"] = sum(self.unknown_angle_tokens.values())
        return result


def protocol_spec() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "gt_split": "one-or-more newline characters",
        "prediction_boundaries": {
            "newlines": True,
            "relation_tokens": list(RELATION_TOKENS),
            "block_tags": list(BLOCK_TAG_NAMES),
            "mm_think_end": "boundary; preserve content on both sides",
        },
        "spaces_removed": ["ASCII space U+0020", "full-width space U+3000"],
        "literal_characters_preserved": ["#", "|", "Markdown punctuation"],
        "tag_policy": {
            "block_tag_names": list(BLOCK_TAG_NAMES),
            "inline_tag_names": list(INLINE_TAG_NAMES),
            "html_comments": "remove without adding a boundary",
            "deepseek_ref_det": "remove metadata without adding a boundary",
            "provider_begin_token": "remove without adding a boundary",
            "markdown_links": "keep visible label and remove target syntax",
            "markdown_fence_lines": "remove as a line boundary",
        },
        "prediction_parser": "plain text blocks; bypass OmniDocBench md_tex_filter",
        "unknown_angle_policy": "preserve and report",
        "missing_page_policy": "evaluate existing pages only",
    }


def protocol_hash() -> str:
    encoded = json.dumps(protocol_spec(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _remove_spaces(text: str) -> tuple[str, int]:
    removed = 0
    for char in SPACE_CHARS:
        count = text.count(char)
        if count:
            removed += count
            text = text.replace(char, "")
    return text, removed


def preprocess_gt(text: Any) -> tuple[list[str], PreprocessStats]:
    raw = str(text or "")
    if BOUNDARY in raw:
        raise ValueError("GT contains reserved boundary character U+001E")
    stats = PreprocessStats(raw_chars=len(raw))
    stats.newline_boundaries = len(NEWLINE_RE.findall(raw))
    units: list[str] = []
    for part in NEWLINE_RE.split(raw):
        cleaned, removed = _remove_spaces(part)
        stats.removed_spaces += removed
        cleaned = cleaned.strip()
        if cleaned:
            units.append(cleaned)
    stats.unit_count = len(units)
    stats.normalized_chars = sum(len(unit) for unit in units)
    return units, stats


def _sub_count(pattern: re.Pattern[str], repl: str, text: str) -> tuple[str, int]:
    return pattern.subn(repl, text)


def preprocess_prediction(text: Any) -> tuple[list[str], PreprocessStats]:
    raw = str(text or "")
    if BOUNDARY in raw:
        raise ValueError("Prediction contains reserved boundary character U+001E")
    stats = PreprocessStats(raw_chars=len(raw))
    working = raw

    working, stats.comments_removed = _sub_count(HTML_COMMENT_RE, "", working)
    working, stats.think_boundaries = _sub_count(THINK_END_RE, BOUNDARY, working)
    working, stats.ref_blocks_removed = _sub_count(REF_BLOCK_RE, "", working)
    working, stats.det_blocks_removed = _sub_count(DET_BLOCK_RE, "", working)
    working, unclosed_det = _sub_count(UNCLOSED_DET_RE, "", working)
    stats.det_blocks_removed += unclosed_det
    working, stats.provider_tokens_removed = _sub_count(PROVIDER_TOKEN_RE, "", working)
    working, stats.begin_tokens_removed = _sub_count(BEGIN_SENTENCE_TOKEN_RE, "", working)

    stats.relation_boundaries = len(RELATION_RE.findall(working))
    working = RELATION_RE.sub(BOUNDARY, working)
    stats.newline_boundaries = len(NEWLINE_RE.findall(working))
    working = NEWLINE_RE.sub(BOUNDARY, working)

    working, stats.block_tag_boundaries = _sub_count(BLOCK_TAG_RE, BOUNDARY, working)
    working, stats.inline_tags_removed = _sub_count(INLINE_TAG_RE, "", working)
    working = MALFORMED_EMPTY_TAG_RE.sub(BOUNDARY, working)
    working, stats.fence_lines_removed = _sub_count(FENCE_LINE_RE, BOUNDARY, working)

    def unwrap_link(match: re.Match[str]) -> str:
        stats.markdown_links_unwrapped += 1
        return match.group(1)

    working = MARKDOWN_LINK_RE.sub(unwrap_link, working)
    stats.unknown_angle_tokens.update(UNKNOWN_ANGLE_RE.findall(working))

    units: list[str] = []
    for part in working.split(BOUNDARY):
        cleaned, removed = _remove_spaces(part)
        stats.removed_spaces += removed
        cleaned = cleaned.strip()
        if cleaned:
            units.append(cleaned)
    stats.unit_count = len(units)
    stats.normalized_chars = sum(len(unit) for unit in units)
    return units, stats
