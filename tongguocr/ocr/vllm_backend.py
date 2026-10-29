from __future__ import annotations

import base64
import io
import mimetypes
import time
from pathlib import Path
import threading

import requests

from ..models import OcrBlockResult
from .preprocess import prepare_pil_image
from .relation_tokens import remove_relation_tokens, strip_control_tokens


_THREAD_LOCAL = threading.local()


def encode_image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/jpeg"
    with image_path.open("rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def encode_prepared_image_to_data_url(
    image_path: str | Path,
    *,
    image_max_pixels: int,
    image_min_pixels: int,
) -> str:
    """Pre-resize locally, then re-encode losslessly (PNG) as a data URL."""
    image = prepare_pil_image(
        Path(image_path),
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def recognize_block_image_vllm(
    *,
    block_id: int,
    crop_path: str | Path,
    masked_crop_path: str | Path | None,
    config: dict,
) -> OcrBlockResult:
    backend_cfg = config["ocr"]["vllm"]
    # Shared inference profile (ocr.inference) is the single source of truth
    # for max_tokens / pixel budget / temperature — the same keys the
    # transformers backend reads — so the two backends cannot drift apart.
    inference_cfg = config["ocr"]["inference"]
    source_path = Path(masked_crop_path or crop_path)
    if backend_cfg.get("pre_resize", True):
        image_url = encode_prepared_image_to_data_url(
            source_path,
            image_max_pixels=int(inference_cfg["image_max_pixels"]),
            image_min_pixels=int(inference_cfg["image_min_pixels"]),
        )
    else:
        image_url = encode_image_to_data_url(source_path)
    payload = {
        "model": backend_cfg["model_name"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": backend_cfg.get("prompt", config["ocr"]["prompt"])},
                ],
            }
        ],
        "max_tokens": int(inference_cfg["max_tokens"]),
        "temperature": float(inference_cfg["temperature"]),
        # Keep special tokens server-side so the relation tokens survive;
        # both output forms are derived client-side below, mirroring the
        # transformers backend exactly.
        "skip_special_tokens": False,
    }
    start = time.perf_counter()
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    response = session.post(
        f"{backend_cfg['api_base'].rstrip('/')}/chat/completions",
        json=payload,
        timeout=backend_cfg["timeout"],
    )
    response.raise_for_status()
    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    token_text = strip_control_tokens(raw)
    plain_text = remove_relation_tokens(token_text)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return OcrBlockResult(
        block_id=block_id,
        crop_path=str(crop_path),
        masked_crop_path=str(masked_crop_path) if masked_crop_path else None,
        ocr_text=plain_text,
        backend="vllm",
        latency_ms=latency_ms,
        ocr_text_tokens=token_text,
    )
