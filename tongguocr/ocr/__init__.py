"""OCR backends for tongguocr.

Both are in-process: ``transformers`` (default, local GPU) and ``vllm``
(OpenAI-compatible HTTP client pointed at a running vLLM server; see
docs/vllm.md).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..models import OcrBlockResult
from .transformers_backend import recognize_block_image_transformers
from .vllm_backend import recognize_block_image_vllm

__all__ = ["recognize_blocks"]


def recognize_blocks(block_images: list[dict], config: dict) -> list[OcrBlockResult]:
    backend = config["ocr"]["backend"]

    def recognize(item):
        kwargs = {
            "block_id": item["block_id"],
            "crop_path": item["crop_path"],
            "masked_crop_path": item.get("masked_crop_path"),
            "config": config,
        }
        if backend == "vllm":
            return recognize_block_image_vllm(**kwargs)
        elif backend == "transformers":
            return recognize_block_image_transformers(**kwargs)
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")

    workers = int(config["ocr"].get("vllm", {}).get("concurrency", 4)) if backend == "vllm" else 1
    if workers < 1:
        raise ValueError("ocr.vllm.concurrency must be positive")
    if workers == 1 or len(block_images) < 2:
        return [recognize(item) for item in block_images]
    # Each crop keeps its own prompt/context. Concurrent independent requests
    # let vLLM batch decoding; map preserves the original block order.
    with ThreadPoolExecutor(max_workers=min(workers, len(block_images))) as pool:
        return list(pool.map(recognize, block_images))
