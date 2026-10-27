from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import numpy as np

from .blocking.models import PageResult, TextLine


@dataclass
class SavedBlockCrop:
    block_id: int
    path: Path
    background_fill_color: list[int]
    crop_canvas_size: list[int]
    sparse_canvas_applied: bool


def _clip_crop_box(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    left = max(0, int(round(x1)) - padding)
    top = max(0, int(round(y1)) - padding)
    right = min(width, int(round(x2)) + padding)
    bottom = min(height, int(round(y2)) + padding)
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def _intersect_boxes(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> tuple[int, int, int, int] | None:
    left = max(int(round(box_a[0])), int(round(box_b[0])))
    top = max(int(round(box_a[1])), int(round(box_b[1])))
    right = min(int(round(box_a[2])), int(round(box_b[2])))
    bottom = min(int(round(box_a[3])), int(round(box_b[3])))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _translate_to_crop(
    box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        box[0] - crop_box[0],
        box[1] - crop_box[1],
        box[2] - crop_box[0],
        box[3] - crop_box[1],
    )


def _paint_rect(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        return
    mask[top:bottom, left:right] = True


def _compute_background_fill_color(
    crop_pixels: np.ndarray,
    crop_box: tuple[int, int, int, int],
    line_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    crop_height, crop_width = crop_pixels.shape[:2]
    text_mask = np.zeros((crop_height, crop_width), dtype=bool)
    for line_box in line_boxes:
        intersection = _intersect_boxes(line_box, crop_box)
        if intersection is None:
            continue
        _paint_rect(text_mask, _translate_to_crop(intersection, crop_box))

    background_mask = ~text_mask
    if background_mask.any():
        return crop_pixels[background_mask].mean(axis=0)
    return crop_pixels.reshape(-1, crop_pixels.shape[-1]).mean(axis=0)


def _mask_overlapping_foreign_regions(
    crop_pixels: np.ndarray,
    crop_box: tuple[int, int, int, int],
    current_line_boxes: list[tuple[float, float, float, float]],
    other_line_boxes: list[tuple[float, float, float, float]],
    fill_color: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return crop_pixels
    crop_height, crop_width = crop_pixels.shape[:2]
    own_line_mask = np.zeros((crop_height, crop_width), dtype=bool)
    for line_box in current_line_boxes:
        intersection = _intersect_boxes(line_box, crop_box)
        if intersection is None:
            continue
        _paint_rect(own_line_mask, _translate_to_crop(intersection, crop_box))

    overlap_regions: list[tuple[int, int, int, int]] = []
    foreign_mask = np.zeros((crop_height, crop_width), dtype=bool)
    for other_line_box in other_line_boxes:
        crop_overlap = _intersect_boxes(other_line_box, crop_box)
        if crop_overlap is None:
            continue
        translated = _translate_to_crop(crop_overlap, crop_box)
        overlap_regions.append(translated)
        _paint_rect(foreign_mask, translated)

    foreign_mask &= ~own_line_mask
    if not foreign_mask.any():
        return crop_pixels

    masked_pixels = crop_pixels.copy()
    masked_pixels[foreign_mask] = np.asarray(fill_color, dtype=np.uint8)
    return masked_pixels


def _compute_tight_text_bbox_area_ratio(
    crop_box: tuple[int, int, int, int],
    current_line_boxes: list[tuple[float, float, float, float]],
) -> float:
    intersections: list[tuple[int, int, int, int]] = []
    for line_box in current_line_boxes:
        intersection = _intersect_boxes(line_box, crop_box)
        if intersection is not None:
            intersections.append(intersection)
    if not intersections:
        return 0.0

    left = min(box[0] for box in intersections)
    top = min(box[1] for box in intersections)
    right = max(box[2] for box in intersections)
    bottom = max(box[3] for box in intersections)
    text_bbox_area = max(0, right - left) * max(0, bottom - top)
    crop_area = max(1, (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]))
    return float(text_bbox_area) / float(crop_area)


def _square_pad_crop(
    crop_pixels: np.ndarray,
    fill_color: np.ndarray,
    square_side: int,
) -> np.ndarray:
    crop_height, crop_width = crop_pixels.shape[:2]
    square_side = max(square_side, crop_height, crop_width, 1)
    canvas = np.full(
        (square_side, square_side, crop_pixels.shape[-1]),
        np.asarray(fill_color, dtype=np.uint8),
        dtype=np.uint8,
    )
    x_offset = (square_side - crop_width) // 2
    y_offset = (square_side - crop_height) // 2
    canvas[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width] = crop_pixels
    return canvas


def save_block_crops(
    image_path: str | Path,
    page_result: PageResult,
    lines: list[TextLine],
    output_dir: str | Path,
    padding: int = 8,
    base_scale: float = 1.25,
    sparse_scale: float = 1.5,
    sparse_line_threshold: int = 2,
    sparse_text_bbox_area_ratio: float = 0.55,
    crop_mode: str = "padded_square",
    enable_foreign_mask: bool = True,
    selected_block_ids: set[int] | None = None,
) -> list[SavedBlockCrop]:
    """
    Save OCR input crops for each block.

    crop_mode:
    - padded_square: current stable behavior. Expand the block bbox by `padding`,
      mask foreign text lines, then place it on a larger square canvas.
    - raw_bbox: crop the block bbox exactly, mask foreign text lines, and keep the
      original aspect ratio.
    - square_no_padding: crop the block bbox exactly, mask foreign text lines,
      then pad only the short side to make a 1:1 canvas. The long side touches
      the final image boundary.
    """
    crop_mode = crop_mode.strip().lower()
    if crop_mode in {"default", "current"}:
        crop_mode = "padded_square"
    valid_modes = {"padded_square", "raw_bbox", "square_no_padding"}
    if crop_mode not in valid_modes:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}; expected one of {sorted(valid_modes)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_crops: list[SavedBlockCrop] = []
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
        for block in page_result.blocks:
            if selected_block_ids is not None and block.block_id not in selected_block_ids:
                continue
            crop_padding = padding if crop_mode == "padded_square" else 0
            crop_box = _clip_crop_box(block.bbox, canvas.size, padding=crop_padding)
            cropped = canvas.crop(crop_box)
            crop_pixels = np.array(cropped, dtype=np.uint8)
            current_line_boxes = [lines[line_id].bbox for line_id in block.line_ids if 0 <= line_id < len(lines)]
            all_line_boxes = [line.bbox for line in lines]
            other_line_boxes = [
                lines[line_id].bbox
                for other_block in page_result.blocks
                if other_block.block_id != block.block_id
                for line_id in other_block.line_ids
                if 0 <= line_id < len(lines)
            ]
            fill_color = _compute_background_fill_color(crop_pixels, crop_box, all_line_boxes)
            crop_pixels = _mask_overlapping_foreign_regions(
                crop_pixels,
                crop_box,
                current_line_boxes,
                other_line_boxes,
                fill_color,
                enabled=enable_foreign_mask,
            )
            line_count = len(block.line_ids)
            text_bbox_area_ratio = _compute_tight_text_bbox_area_ratio(crop_box, current_line_boxes)
            sparse_canvas_applied = (
                line_count <= sparse_line_threshold
                or text_bbox_area_ratio >= sparse_text_bbox_area_ratio
            )
            crop_height, crop_width = crop_pixels.shape[:2]
            if crop_mode == "raw_bbox":
                square_side = None
                crop_canvas_size = [int(crop_width), int(crop_height)]
                sparse_canvas_applied = False
            else:
                if crop_mode == "square_no_padding":
                    scale = 1.0
                    sparse_canvas_applied = False
                else:
                    scale = sparse_scale if sparse_canvas_applied else base_scale
                square_side = int(np.ceil(scale * max(crop_width, crop_height)))
                crop_pixels = _square_pad_crop(crop_pixels, fill_color, square_side)
                crop_canvas_size = [square_side, square_side]
            out_path = output_dir / (
                f"block_{block.block_id:03d}_{block.line_range[0]}-{block.line_range[1]}.jpg"
            )
            Image.fromarray(crop_pixels).save(out_path)
            fill_color_list = [int(round(channel)) for channel in np.asarray(fill_color).tolist()]
            saved_crops.append(
                SavedBlockCrop(
                    block_id=block.block_id,
                    path=out_path,
                    background_fill_color=fill_color_list,
                    crop_canvas_size=crop_canvas_size,
                    sparse_canvas_applied=sparse_canvas_applied,
                )
            )
    return saved_crops
