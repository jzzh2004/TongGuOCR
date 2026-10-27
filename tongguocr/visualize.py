from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .blocking.models import PageResult, TextLine


LINE_COLOR = (0, 170, 255)
TEXT_BG = (255, 255, 255)
BLOCK_PALETTE = [
    (230, 57, 70),
    (244, 162, 97),
    (233, 196, 106),
    (42, 157, 143),
    (69, 123, 157),
    (69, 80, 180),
    (126, 87, 194),
]


def draw_pipeline_visualization(
    image_path: str | Path,
    lines: list[TextLine],
    page_result: PageResult,
    output_path: str | Path,
    *,
    show_debug_metrics: bool = False,
) -> Path:
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for line in lines:
        draw.rectangle(line.bbox, outline=LINE_COLOR, width=1)

    for block in page_result.blocks:
        block_color = BLOCK_PALETTE[block.block_id % len(BLOCK_PALETTE)]
        draw.rectangle(block.bbox, outline=block_color, width=4)
        label = f"B{block.block_id} {block.line_range[0]}-{block.line_range[1]}"
        if show_debug_metrics:
            label += (
                f" n={block.stats['line_count']}"
                f" s={block.stats.get('vertical_strip_ratio', 0.0):.1f}"
            )
        text_bbox = draw.textbbox((0, 0), label, font=font)
        x1, y1, _, _ = block.bbox
        text_x = int(x1)
        text_y = max(0, int(y1) - (text_bbox[3] - text_bbox[1]) - 2)
        draw.rectangle(
            (
                text_x,
                text_y,
                text_x + (text_bbox[2] - text_bbox[0]) + 4,
                text_y + (text_bbox[3] - text_bbox[1]) + 4,
            ),
            fill=TEXT_BG,
        )
        draw.text((text_x + 2, text_y + 2), label, font=font, fill=block_color)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path

