"""Shared OCR image preprocessing.

This is the single source of the pixel-budget resize applied to every block
crop before recognition. It must stay bit-identical across backends (same
int rounding, same PIL default resample, same RGB conversion) so the
transformers and vLLM backends feed the model the same pixel matrix as the
archived benchmark runs.
"""

from __future__ import annotations

from pathlib import Path


def prepare_pil_image(
    image_path: str | Path,
    *,
    image_max_pixels: int,
    image_min_pixels: int,
):
    """Open an image and clamp its total pixel count into the budget."""
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
