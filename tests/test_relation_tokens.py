import unittest
from tongguocr.ocr.relation_tokens import RELATION_TOKENS, relation_tokens_to_newlines, strip_control_tokens
from evaluation.ro_ed.protocol import preprocess_prediction


class RelationTokenTests(unittest.TestCase):
    def test_all_24_tokens_are_line_boundaries(self):
        self.assertEqual(len(set(RELATION_TOKENS)), 24)
        for token in RELATION_TOKENS:
            with self.subTest(token=token):
                self.assertEqual(relation_tokens_to_newlines('甲'+token+'乙'), '甲\n乙')

    def test_control_tokens_removed_unknown_text_preserved(self):
        text = strip_control_tokens('<|assistant|>甲<left>乙<形象><|endoftext|>')
        self.assertEqual(relation_tokens_to_newlines(text), '甲\n乙<形象>')

    def test_newline_output_preserves_reading_order_units(self):
        source = '甲<left|far_up>乙\n丙<far_right|far_down>丁'
        self.assertEqual(preprocess_prediction(source)[0], preprocess_prediction(relation_tokens_to_newlines(source))[0])


if __name__ == '__main__':
    unittest.main()
