"""OCR backends for tongguocr.

In-process backends (``transformers``, ``vllm`` HTTP client) are dispatched
here; the heavy GLMOCR helper scripts (``glmocr_block_ocr.py``,
``glmocr_direct_ocr.py``, ``glmocr_vllm_block_ocr.py``) run as subprocesses
orchestrated by :mod:`tongguocr.pipeline`.
"""

from __future__ import annotations

from ..models import OcrBlockResult
from .transformers_backend import recognize_block_image_transformers
from .vllm_backend import recognize_block_image_vllm

__all__ = ["recognize_blocks"]


def recognize_blocks(block_images: list[dict], config: dict) -> list[OcrBlockResult]:
    backend = config["ocr"]["backend"]
    results: list[OcrBlockResult] = []
    if backend == "glmocr_hf":
        raise ValueError(
            "OCR backend 'glmocr_hf' must be run via the pipeline so it can "
            "prepare page stubs and invoke the external GLMOCR helper environment."
        )
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
