from __future__ import annotations

import time
from pathlib import Path

from ..models import OcrBlockResult
from .relation_tokens import finalize_ocr_text


_TRANSFORMERS_CACHE: dict[str, tuple[object, object]] = {}


def _load_transformers_runtime(model_path: str):
    cache = _TRANSFORMERS_CACHE.get(model_path)
    if cache is not None:
        return cache

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    _TRANSFORMERS_CACHE[model_path] = (processor, model)
    return processor, model


def recognize_block_image_transformers(
    *,
    block_id: int,
    crop_path: str | Path,
    masked_crop_path: str | Path | None,
    config: dict,
) -> OcrBlockResult:
    from PIL import Image

    model_path = config["ocr"]["transformers"]["model_path"]
    processor, model = _load_transformers_runtime(model_path)
    image = Image.open(masked_crop_path or crop_path).convert("RGB")
    prompt = config["ocr"]["prompt"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    start = time.perf_counter()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=config["ocr"]["transformers"]["max_tokens"],
        do_sample=config["ocr"]["transformers"]["temperature"] > 0,
        temperature=max(config["ocr"]["transformers"]["temperature"], 1e-5),
    )
    trimmed = []
    input_ids = inputs.get("input_ids")
    for idx, output_ids in enumerate(generated_ids):
        if input_ids is not None:
            trimmed.append(output_ids[input_ids[idx].shape[0] :])
        else:
            trimmed.append(output_ids)
    # skip_special_tokens=False keeps the relation tokens in the decoded text;
    # finalize_ocr_text() turns them into newlines (reading-order line breaks)
    # and strips the remaining control special tokens.
    decoded = processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded = finalize_ocr_text(decoded, tokenizer.all_special_tokens)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return OcrBlockResult(
        block_id=block_id,
        crop_path=str(crop_path),
        masked_crop_path=str(masked_crop_path) if masked_crop_path else None,
        ocr_text=decoded,
        backend="transformers",
        latency_ms=latency_ms,
    )
