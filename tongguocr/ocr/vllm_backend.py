from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
import threading
import time

import requests

from ..models import OcrBlockResult
from .preprocess import prepare_pil_image
from .relation_tokens import relation_tokens_to_newlines, strip_control_tokens

_THREAD_LOCAL = threading.local()


def encode_image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('utf-8')}"


def encode_prepared_image_to_data_url(
    image_path: str | Path, *, image_max_pixels: int, image_min_pixels: int,
) -> str:
    """Apply the shared pixel budget and encode the crop losslessly."""
    image = prepare_pil_image(Path(image_path), image_max_pixels=image_max_pixels,
                              image_min_pixels=image_min_pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _crop_messages(item: dict, config: dict) -> list[dict]:
    backend = config["ocr"]["vllm"]
    inference = config["ocr"]["inference"]
    source = item.get("masked_crop_path") or item["crop_path"]
    if backend.get("pre_resize", True):
        url = encode_prepared_image_to_data_url(
            source, image_max_pixels=int(inference["image_max_pixels"]),
            image_min_pixels=int(inference["image_min_pixels"]),
        )
    else:
        url = encode_image_to_data_url(source)
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": backend.get("prompt", config["ocr"]["prompt"])},
    ]}]


def recognize_block_images_vllm(block_images: list[dict], config: dict) -> list[OcrBlockResult]:
    """Send one page as independent conversations in one batch HTTP request.

    vLLM schedules each crop independently; response indices map results back
    to input crops. A positive batch_size splits the page into smaller batches.
    """
    backend = config["ocr"]["vllm"]
    inference = config["ocr"]["inference"]
    batch_size = int(backend.get("batch_size", 0))
    if batch_size < 0:
        raise ValueError("ocr.vllm.batch_size must be nonnegative")
    if not block_images:
        return []
    batch_size = batch_size or len(block_images)
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    results = []
    for offset in range(0, len(block_images), batch_size):
        batch = block_images[offset:offset + batch_size]
        start = time.perf_counter()
        messages = [_crop_messages(item, config) for item in batch]
        # The single-conversation endpoint also works with older servers when
        # explicitly selecting batch_size=1. Whole-page batches require 0.32.
        endpoint = "chat/completions" if len(batch) == 1 else "chat/completions/batch"
        payload = dict(model=backend["model_name"],
                       messages=messages[0] if len(batch) == 1 else messages,
                       max_tokens=int(inference["max_tokens"]),
                       temperature=float(inference["temperature"]),
                       skip_special_tokens=False)
        response = session.post(f"{backend['api_base'].rstrip('/')}/{endpoint}",
                                json=payload, timeout=backend["timeout"])
        response.raise_for_status()
        choices = response.json()["choices"]
        indices = [choice["index"] for choice in choices]
        if len(indices) != len(batch) or set(indices) != set(range(len(batch))):
            raise ValueError(f"vLLM returned incomplete or duplicate batch indices: {indices}")
        choices = sorted(choices, key=lambda choice: choice["index"])
        latency_ms = (time.perf_counter() - start) * 1000.0
        for item, choice in zip(batch, choices):
            raw = choice["message"]["content"]
            if not isinstance(raw, str):
                raise ValueError("vLLM returned a non-text recognition result")
            token_text = strip_control_tokens(raw)
            results.append(OcrBlockResult(
                block_id=item["block_id"], crop_path=str(item["crop_path"]),
                masked_crop_path=str(item["masked_crop_path"]) if item.get("masked_crop_path") else None,
                ocr_text=relation_tokens_to_newlines(token_text), backend="vllm",
                latency_ms=latency_ms, ocr_text_tokens=token_text,
            ))
    return results


def recognize_block_image_vllm(*, block_id: int, crop_path: str | Path,
                               masked_crop_path: str | Path | None, config: dict) -> OcrBlockResult:
    return recognize_block_images_vllm([dict(block_id=block_id, crop_path=crop_path,
                                            masked_crop_path=masked_crop_path)], config)[0]
