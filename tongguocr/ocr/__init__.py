"""Local Transformers OCR and page-batched vLLM OCR."""
from __future__ import annotations

from ..models import OcrBlockResult
from .transformers_backend import recognize_block_image_transformers
from .vllm_backend import recognize_block_images_vllm

__all__ = ["recognize_blocks"]


def recognize_blocks(block_images: list[dict], config: dict) -> list[OcrBlockResult]:
    backend = config["ocr"]["backend"]
    if backend == "vllm":
        return recognize_block_images_vllm(block_images, config)
    if backend != "transformers":
        raise ValueError(f"Unsupported OCR backend: {backend}")
    return [recognize_block_image_transformers(
        block_id=item["block_id"], crop_path=item["crop_path"],
        masked_crop_path=item.get("masked_crop_path"), config=config,
    ) for item in block_images]
