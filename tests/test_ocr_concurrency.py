import threading
import unittest
from unittest.mock import patch

from tongguocr.config import load_config
from tongguocr.models import OcrBlockResult
from tongguocr.ocr import recognize_blocks


class CropConcurrencyTests(unittest.TestCase):
    def test_same_page_crops_overlap_and_keep_input_order(self):
        config = load_config()
        config['ocr']['backend'] = 'vllm'
        config['ocr']['vllm']['concurrency'] = 3
        barrier = threading.Barrier(3, timeout=5)
        seen = []
        def recognize(**kwargs):
            seen.append(kwargs['block_id'])
            barrier.wait()
            return OcrBlockResult(ocr_text=str(kwargs['block_id']), backend='vllm', **{k: kwargs[k] for k in ('block_id', 'crop_path', 'masked_crop_path')})
        images = [dict(block_id=i, crop_path=f'{i}.png') for i in [3, 1, 2]]
        with patch('tongguocr.ocr.recognize_block_image_vllm', side_effect=recognize):
            result = recognize_blocks(images, config)
        self.assertCountEqual(seen, [3, 1, 2])
        self.assertEqual([r.block_id for r in result], [3, 1, 2])

    def test_failed_crop_is_not_silently_dropped(self):
        config = load_config()
        config['ocr']['backend'] = 'vllm'
        with patch('tongguocr.ocr.recognize_block_image_vllm', side_effect=RuntimeError('HTTP failure')):
            with self.assertRaisesRegex(RuntimeError, 'HTTP failure'):
                recognize_blocks([dict(block_id=i, crop_path=f'{i}.png') for i in range(2)], config)

    def test_serial_setting_remains_available(self):
        config = load_config()
        config['ocr']['backend'] = 'vllm'
        config['ocr']['vllm']['concurrency'] = 1
        seen = []
        def recognize(**kwargs):
            seen.append(kwargs['block_id'])
            return kwargs['block_id']
        with patch('tongguocr.ocr.recognize_block_image_vllm', side_effect=recognize):
            result = recognize_blocks([dict(block_id=i, crop_path=f'{i}.png') for i in range(3)], config)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(result, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
