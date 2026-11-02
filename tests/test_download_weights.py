import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/download_weights.sh'
OCR_FILES = {
    'model.safetensors', 'config.json', 'generation_config.json',
    'processor_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'chat_template.jinja',
}
FILES = OCR_FILES | {'layout/finetune_infer.pth'}


class DownloadWeightsTests(unittest.TestCase):
    def run_download(self, backend, incomplete=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'tools').mkdir()
            shutil.copy2(SCRIPT, root / 'tools/download_weights.sh')
            snapshot = root / 'snapshot'
            for name in FILES - ({'tokenizer.json'} if incomplete else set()):
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            (snapshot / 'README.md').write_text('Remote model card')
            local_readme = root / 'weights/ocr/README.md'
            local_readme.parent.mkdir(parents=True)
            local_readme.write_text('Local instructions')
            module = root / 'modules'
            module.mkdir()
            stub = '''import json, os
from pathlib import Path
def snapshot_download(repo_id, **kwargs):
    Path(os.environ['CALL_RECORD']).write_text(json.dumps({
        'repo_id': repo_id, 'kwargs': kwargs,
        'domain': os.environ.get('MODELSCOPE_DOMAIN'),
    }))
    return os.environ['SNAPSHOT_DIR']
'''
            module_name = 'huggingface_hub' if backend == 'huggingface' else 'modelscope'
            (module / (module_name + '.py')).write_text(stub)
            record = root / 'call.json'
            env = {
                **os.environ, 'PYTHON': sys.executable, 'PYTHONPATH': str(module),
                'SNAPSHOT_DIR': str(snapshot), 'CALL_RECORD': str(record),
                'HF_REPO_ID': 'test/hf', 'MODELSCOPE_REPO_ID': 'test/ms',
                'HF_TOKEN': 'test-hf-token', 'MODELSCOPE_API_TOKEN': 'test-ms-token',
                'MODELSCOPE_DOMAIN': 'modelscope.cn',
            }
            result = subprocess.run(['bash', str(root / 'tools/download_weights.sh'), backend],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(local_readme.read_text(), 'Local instructions')
            if incomplete:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('tokenizer.json', result.stderr)
                self.assertFalse((root / 'weights/ocr/model.safetensors').exists())
                return
            self.assertEqual(result.returncode, 0, result.stderr)
            call = json.loads(record.read_text())
            self.assertEqual(set(call['kwargs']['allow_patterns']), FILES)
            self.assertEqual(call['kwargs']['token'], 'test-hf-token' if backend == 'huggingface' else 'test-ms-token')
            self.assertEqual(call['repo_id'], 'test/hf' if backend == 'huggingface' else 'test/ms')
            if backend == 'modelscope':
                self.assertEqual(call['domain'], 'modelscope.ai')
            for name in OCR_FILES:
                self.assertEqual((root / 'weights/ocr' / name).read_text(), name)
            self.assertEqual((root / 'weights/layout/finetune_infer.pth').read_text(), 'layout/finetune_infer.pth')
            self.assertFalse((root / 'weights/ocr/layout').exists())

    def test_huggingface_download_places_only_required_files(self):
        self.run_download('huggingface')

    def test_modelscope_download_uses_international_site(self):
        self.run_download('modelscope')

    def test_incomplete_snapshot_does_not_install_partial_weights(self):
        for backend in ('huggingface', 'modelscope'):
            with self.subTest(backend=backend):
                self.run_download(backend, incomplete=True)

    def test_invalid_backend_is_rejected(self):
        result = subprocess.run(['bash', str(SCRIPT), 'invalid'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Usage:', result.stderr)


if __name__ == '__main__':
    unittest.main()
