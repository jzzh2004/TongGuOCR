from __future__ import annotations

import base64
import io
import mimetypes
import time
from pathlib import Path
import threading

import requests

from ..models import OcrBlockResult
from .relation_tokens import finalize_ocr_text


_THREAD_LOCAL = threading.local()


def encode_image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/jpeg"
    with image_path.open("rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _prepare_pil_image(
    image_path: Path,
    image_max_pixels: int,
    image_min_pixels: int,
):
    """Mirror glmocr_block_ocr._prepare_pil_image (resize_mode=image_max_pixels).

    Keeping this bit-identical (same int rounding, same PIL default resample,
    same RGB conversion) guarantees the vLLM server receives exactly the same
    pixel matrix the transformers helper would feed its local processor.
    """
    from PIL import Image

    image = Image.open(image_path)
    total_pixels = image.width * image.height
    if total_pixels > image_max_pixels:
        resize_factor = (image_max_pixels / total_pixels) ** 0.5
        width = max(1, int(image.width * resize_factor))
        height = max(1, int(image.height * resize_factor))
        image = image.resize((width, height))

    total_pixels = image.width * image.height
    if total_pixels < image_min_pixels:
        resize_factor = (image_min_pixels / total_pixels) ** 0.5
        width = max(1, int(image.width * resize_factor))
        height = max(1, int(image.height * resize_factor))
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def encode_prepared_image_to_data_url(
    image_path: str | Path,
    *,
    image_max_pixels: int,
    image_min_pixels: int,
) -> str:
    """Pre-resize locally, then re-encode losslessly (PNG) as a data URL."""
    image = _prepare_pil_image(Path(image_path), image_max_pixels, image_min_pixels)
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
    source_path = Path(masked_crop_path or crop_path)
    if backend_cfg.get("pre_resize", True):
        image_max_pixels = int(backend_cfg.get("image_max_pixels", 1048576))
        image_min_pixels = int(backend_cfg.get("image_min_pixels", 1024))
        image_url = encode_prepared_image_to_data_url(
            source_path,
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
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
        "max_tokens": backend_cfg["max_tokens"],
        "temperature": backend_cfg["temperature"],
        # keep special tokens so the relation tokens survive decoding;
        # finalize_ocr_text() below turns them into newlines
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
    text = finalize_ocr_text(data["choices"][0]["message"]["content"])
    latency_ms = (time.perf_counter() - start) * 1000.0
    return OcrBlockResult(
        block_id=block_id,
        crop_path=str(crop_path),
        masked_crop_path=str(masked_crop_path) if masked_crop_path else None,
        ocr_text=text,
        backend="vllm",
        latency_ms=latency_ms,
    )
