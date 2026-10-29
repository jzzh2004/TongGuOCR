import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from evaluation.calculate_metrics import cal_ar_cr, cal_edit_distance, cal_ro_ed, read_records
from evaluation.ro_ed.reading_order import calculate_reading_order, score_page


class ReadingOrderTests(unittest.TestCase):
    def test_character_metric_ties_and_insertions(self):
        cases = [('甲乙', '乙甲', (0.0, 0.0)),
                 ('甲乙', '甲乙丙', (0.5, 1.0)),
                 ('𰻞甲', '𰻞', (0.5, 0.5)),
                 ('甲乙', '', (0.0, 0.0))]
        with contextlib.redirect_stdout(io.StringIO()):
            for gt, pred, expected in cases:
                self.assertEqual(cal_ar_cr(1, [dict(Answer_Simplified=gt, Response=pred)]), expected)
            self.assertEqual(cal_edit_distance(1, [dict(Answer_Simplified='', Response='')]), 0)

    def test_identity_and_permutation(self):
        gt = '甲乙\n丙丁\n戊己'
        identity = score_page(dict(page_id='same', label=gt, predict='甲乙<left|up>丙丁\n戊己'))
        swapped = score_page(dict(page_id='swap', label=gt, predict='丙丁<left|up>甲乙<left|up>戊己'))
        self.assertEqual(identity['edit'], 0)
        self.assertEqual(swapped['gt'], [1, 2, 3])
        self.assertEqual(swapped['pred'], [2, 1, 3])
        self.assertAlmostEqual(swapped['edit'], 2 / 3)

    def test_merged_prediction_retains_gt_order(self):
        page = score_page(dict(page_id='merge', label='甲乙丙丁\n戊己庚辛', predict='甲乙丙丁戊己庚辛'))
        self.assertEqual(page['edit'], 0)

    def test_missing_prediction_is_deletion(self):
        page = score_page(dict(page_id='missing', label='甲乙\n丙丁', predict=''))
        self.assertEqual(page['edit'], 1)

    def test_empty_pair_excluded_from_page_mean(self):
        summary = calculate_reading_order([
            dict(page_id='empty', label='', predict=''),
            dict(page_id='missing', label='甲乙', predict=''),
        ], workers=1)
        self.assertEqual(summary['record_count'], 2)
        self.assertEqual(summary['official_page_count'], 1)
        self.assertEqual(summary['ALL_page_avg'], 1)

    def test_duplicate_page_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            calculate_reading_order([dict(page_id='x'), dict(page_id='x')], workers=1)

    def test_integrated_entry_records_source_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'predict_tokens.jsonl'
            source.write_text(json.dumps(dict(page_id='one', label='甲乙\n丙丁', predict='甲乙<left>丙丁')) + '\n')
            summary = cal_ro_ed(1, read_records(source), workers=1, source_path=source, output_dir=root / 'ro')
            self.assertEqual(summary['ALL_page_avg'], 0)
            self.assertEqual(summary['source_path'], str(source.resolve()))
            self.assertTrue((root / 'ro/per_page.json').is_file())
            self.assertEqual(summary['protocol_sha256'], 'ff195d9bb8574edf69dac04decfd7facd8cd5c2a19e2bd871bf0285ab0ff4623')


if __name__ == '__main__':
    unittest.main()
