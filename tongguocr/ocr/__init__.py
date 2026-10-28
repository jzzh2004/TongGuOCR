"""OCR backends for tongguocr.

Both are in-process: ``transformers`` (default, local GPU) and ``vllm``
(OpenAI-compatible HTTP client pointed at a running vLLM server; see
docs/vllm.md).
"""

from __future__ import annotations

from ..models import OcrBlockResult
from .transformers_backend import recognize_block_image_transformers
from .vllm_backend import recognize_block_image_vllm

__all__ = ["recognize_blocks"]


def recognize_blocks(block_images: list[dict], config: dict) -> list[OcrBlockResult]:
    backend = config["ocr"]["backend"]
    results: list[OcrBlockResult] = []
    for item in block_images:
        kwargs = {
            "block_id": item["block_id"],
            "crop_path": item["crop_path"],
            "masked_crop_path": item.get("masked_crop_path"),
            "config": config,
        }
        if backend == "vllm":
            results.append(recognize_block_image_vllm(**kwargs))
        elif backend == "transformers":
            results.append(recognize_block_image_transformers(**kwargs))
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")
    return results
