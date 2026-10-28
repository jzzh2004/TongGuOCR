"""In-process transformers OCR backend (default).

Mirrors the inference semantics validated for the released weights:
bfloat16, greedy decoding, max 8192 new tokens, and the shared pixel
budget (image_max_pixels / image_min_pixels) applied both on the PIL
image and on the processor, so the same pixel matrix reaches the model
as in the archived benchmark runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..models import OcrBlockResult
from .preprocess import prepare_pil_image
from .relation_tokens import finalize_ocr_text


_TRANSFORMERS_CACHE: dict[str, tuple[object, object]] = {}


def _load_transformers_runtime(model_path: str, device: str):
    cache = _TRANSFORMERS_CACHE.get(model_path)
    if cache is not None:
        return cache

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device else "auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    _TRANSFORMERS_CACHE[model_path] = (processor, model)
    return processor, model


def recognize_block_image_transformers(
    *,
    block_id: int,
    crop_path: str | Path,
    masked_crop_path: str | Path | None,
    config: dict,
) -> OcrBlockResult:
    backend_cfg = config["ocr"]["transformers"]
    inference_cfg = config["ocr"]["inference"]
    model_path = backend_cfg["model_path"]
    processor, model = _load_transformers_runtime(model_path, backend_cfg.get("device", "cuda:0"))

    # Same pixel budget as every other backend: pre-resize the PIL image ...
    image = prepare_pil_image(
        Path(masked_crop_path or crop_path),
        image_max_pixels=int(inference_cfg["image_max_pixels"]),
        image_min_pixels=int(inference_cfg["image_min_pixels"]),
    )
    # ... and mirror it on the processor so its own resizing agrees.
    processor.image_max_pixels = int(inference_cfg["image_max_pixels"])
    processor.image_min_pixels = int(inference_cfg["image_min_pixels"])
    processor.image_do_pan_and_scan = False
    processor.crop_to_patches = False

    prompt = config["ocr"]["prompt"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    started = time.perf_counter()
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(inference_cfg["max_tokens"]),
            do_sample=False,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    # Relation tokens are special tokens: decode with them kept, then turn
    # them into newlines (reading-order line breaks) and strip other specials.
    decoded = processor.decode(generated_ids, skip_special_tokens=False)
    decoded = finalize_ocr_text(
        decoded, getattr(processor, "tokenizer", processor).all_special_tokens
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return OcrBlockResult(
        block_id=block_id,
        crop_path=str(crop_path),
        masked_crop_path=str(masked_crop_path) if masked_crop_path else None,
        ocr_text=decoded,
        backend="transformers",
        latency_ms=latency_ms,
    )
