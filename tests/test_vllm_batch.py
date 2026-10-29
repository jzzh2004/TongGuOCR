"""Batch submission, crop identity, and server response validation."""
import unittest
from unittest.mock import Mock, patch

import requests

from tongguocr.config import load_config
from tongguocr.ocr import recognize_blocks


class PageBatchTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config['ocr']['backend'] = 'vllm'

    def response(self, indices):
        result = Mock()
        result.json.return_value = {'choices': [dict(index=i, message=dict(content=f'甲{i}<left|up>乙')) for i in indices]}
        return result

    def test_whole_page_one_request_independent_images_and_stable_order(self):
        images = [dict(block_id=i, crop_path=f'{i}.png') for i in range(10, -1, -1)]
        session = Mock()
        session.post.return_value = self.response(list(range(10, -1, -1)))
        with patch('tongguocr.ocr.vllm_backend._THREAD_LOCAL', session=session), \
             patch('tongguocr.ocr.vllm_backend.encode_prepared_image_to_data_url', side_effect=lambda path, **kw: str(path)):
            results = recognize_blocks(images, self.config)
        session.post.assert_called_once()
        args, kwargs = session.post.call_args
        self.assertTrue(args[0].endswith('/chat/completions/batch'))
        payload = kwargs['json']
        self.assertEqual(len(payload['messages']), 11)
        self.assertFalse(payload['skip_special_tokens'])
        self.assertEqual([m[0]['content'][0]['image_url']['url'] for m in payload['messages']], [i['crop_path'] for i in images])
        self.assertEqual([r.block_id for r in results], [i['block_id'] for i in images])
        self.assertEqual(results[0].ocr_text, '甲0\n乙')
        self.assertEqual(results[0].ocr_text_tokens, '甲0<left|up>乙')

    def test_incomplete_or_duplicate_batch_fails(self):
        images = [dict(block_id=i, crop_path=f'{i}.png') for i in range(2)]
        for indices in ([0], [0, 0], [0, 2]):
            session = Mock()
            session.post.return_value = self.response(indices)
            with self.subTest(indices=indices), \
                 patch('tongguocr.ocr.vllm_backend._THREAD_LOCAL', session=session), \
                 patch('tongguocr.ocr.vllm_backend._crop_messages', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'batch indices'):
                    recognize_blocks(images, self.config)

    def test_http_failure_propagates(self):
        session = Mock()
        session.post.return_value.raise_for_status.side_effect = requests.HTTPError('HTTP failure')
        with patch('tongguocr.ocr.vllm_backend._THREAD_LOCAL', session=session), \
             patch('tongguocr.ocr.vllm_backend._crop_messages', return_value=[]):
            with self.assertRaises(requests.HTTPError):
                recognize_blocks([dict(block_id=0, crop_path='0.png')], self.config)

    def test_serial_limit_and_empty_page(self):
        self.config['ocr']['vllm']['batch_size'] = 1
        session = Mock()
        session.post.return_value = self.response([0])
        with patch('tongguocr.ocr.vllm_backend._THREAD_LOCAL', session=session), \
             patch('tongguocr.ocr.vllm_backend._crop_messages', return_value=[]):
            self.assertEqual(recognize_blocks([], self.config), [])
            results = recognize_blocks([dict(block_id=i, crop_path=f'{i}.png') for i in range(3)], self.config)
        self.assertEqual([r.block_id for r in results], [0, 1, 2])
        self.assertEqual(session.post.call_count, 3)
        self.assertTrue(all(call.args[0].endswith('/chat/completions') for call in session.post.call_args_list))


if __name__ == '__main__':
    unittest.main()
