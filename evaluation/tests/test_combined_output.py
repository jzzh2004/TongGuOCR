import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


class CombinedOutputTests(unittest.TestCase):
    def test_eval_entry_writes_all_metrics_to_one_file(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'run/texts').mkdir(parents=True)
            (root/'data/txt_line').mkdir(parents=True)
            (root/'run/texts/one.txt').write_text('丙丁\n甲乙')
            (root/'data/txt_line/one.txt').write_text('0,0,1,0,1,1,0,1,甲乙\n0,0,1,0,1,1,0,1,丙丁\n')
            subprocess.run(['bash',str(repo/'evaluation/run_eval.sh'),'--dataset','m5',
                            '--run-dir',str(root/'run'),'--dataset-root',str(root/'data'),
                            '--output-dir',str(root/'eval'),'--match-workers','1'],
                           env={**os.environ,'PYTHON':sys.executable}, check=True, capture_output=True)
            self.assertEqual([p.name for p in (root/'eval').iterdir()], ['metrics.json'])
            metrics = json.loads((root/'eval/metrics.json').read_text())
            self.assertTrue({'AR','CR','NED','F1','BLEU','RO-ED'} <= metrics.keys())
            self.assertEqual(metrics['RO-ED'], 1.0)


if __name__ == '__main__':
    unittest.main()
