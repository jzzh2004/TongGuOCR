from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from evaluation.ro_ed.protocol import (
    preprocess_gt,
    preprocess_prediction,
    protocol_hash,
)
from evaluation.ro_ed.ro_ed import EXPECTED_PROTOCOL_HASH, plain_text_filter

# byte-level pin of the protocol implementation used by the paper's runs
PROTOCOL_FILE = Path(__file__).resolve().parents[1] / "ro_ed" / "protocol.py"
PROTOCOL_FILE_SHA256 = "369032913f9592d0a0c15fc0b92919f879536294092072aa6b04bc4b49726bf1"


class ProtocolTests(unittest.TestCase):
    def test_gt_preserves_real_symbols_and_removes_spaces(self):
        units, stats = preprocess_gt(" A|# B\n\nC　D ")
        self.assertEqual(units, ["A|#B", "CD"])
        self.assertEqual(stats.removed_spaces, 4)

    def test_relation_and_repeated_newline_are_boundaries(self):
        units, _ = preprocess_prediction("甲<left|up>乙\n\n\n丙")
        self.assertEqual(units, ["甲", "乙", "丙"])

    def test_think_tag_preserves_both_sides(self):
        units, stats = preprocess_prediction("前一</mm:think>後一</mm:think>後二")
        self.assertEqual(units, ["前一", "後一", "後二"])
        self.assertEqual(stats.think_boundaries, 2)

    def test_block_and_inline_tags(self):
        units, _ = preprocess_prediction("甲<sup>上</sup><br/>乙<td>丙</td>丁")
        self.assertEqual(units, ["甲上", "乙", "丙", "丁"])

    def test_deepseek_metadata_is_removed(self):
        units, stats = preprocess_prediction(
            "正文<|ref|>text<|/ref|><|det|>[[1,2,3,4]]"
        )
        self.assertEqual(units, ["正文"])
        self.assertEqual(stats.ref_blocks_removed, 1)
        self.assertEqual(stats.det_blocks_removed, 1)

    def test_removed_metadata_does_not_create_a_line_boundary(self):
        units, stats = preprocess_prediction(
            "甲<!-- hidden -->乙<|ref|>text<|/ref|>丙<｜begin▁of▁sentence｜>丁"
        )
        self.assertEqual(units, ["甲乙丙丁"])
        self.assertEqual(stats.comments_removed, 1)
        self.assertEqual(stats.ref_blocks_removed, 1)
        self.assertEqual(stats.begin_tokens_removed, 1)

    def test_unknown_angle_text_is_preserved(self):
        units, stats = preprocess_prediction("甲<形象>乙")
        self.assertEqual(units, ["甲<形象>乙"])
        self.assertEqual(stats.unknown_angle_tokens["<形象>"], 1)

    def test_literal_markdown_symbols_are_preserved(self):
        units, _ = preprocess_prediction("春江壯||延\n#正文")
        self.assertEqual(units, ["春江壯||延", "#正文"])

    def test_markdown_link_keeps_visible_text(self):
        units, _ = preprocess_prediction("甲[可見文字](https://example.com)乙")
        self.assertEqual(units, ["甲可見文字乙"])

    def test_double_pipe_lines_bypass_markdown_table_parsing(self):
        # OCR pages regularly contain literal "||" and "|" characters; the
        # plain-text adapter must keep them as text lines and never produce a
        # Markdown table object (in-repo regression for the case verified on
        # real pages during the paper runs).
        source = "春江壯||延<left|up>#正文\na|b||c|d"
        units, _ = preprocess_prediction(source)
        self.assertEqual(len(units), 3)
        self.assertEqual(sum(unit.count("|") for unit in units), 6)
        parsed = plain_text_filter("\n".join(units))
        self.assertEqual(len(parsed["text_all"]), 3)
        self.assertEqual(parsed["html_table"], [])

    def test_protocol_hash_pinned(self):
        self.assertEqual(protocol_hash(), EXPECTED_PROTOCOL_HASH)
        digest = hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()
        self.assertEqual(digest, PROTOCOL_FILE_SHA256)


if __name__ == "__main__":
    unittest.main()
