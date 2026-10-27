from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class SimpleLine:
    line_id: int
    bbox: tuple[float, float, float, float]
    text: str = ""


@dataclass
class SimpleLayoutResult:
    page_id: str
    image_path: str
    image_size: tuple[int, int]
    lines: list[SimpleLine]


def detect_text_lines_simple(image_path: str | Path) -> SimpleLayoutResult:
    """Simple text line detection using OpenCV."""
    image_path = Path(image_path)
    page_id = image_path.stem

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    line_id = 0
    min_height = 10
    max_height = h // 2

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch < min_height or ch > max_height or cw < 20:
            continue

        lines.append(SimpleLine(
            line_id=line_id,
            text="",
            bbox=(float(x), float(y), float(x + cw), float(y + ch))
        ))
        line_id += 1

    lines.sort(key=lambda l: (l.bbox[1], l.bbox[0]))

    for i, line in enumerate(lines):
        line.line_id = i

    return SimpleLayoutResult(
        page_id=page_id,
        image_path=str(image_path),
        image_size=(w, h),
        lines=lines
    )


def extract_line_bboxes_simple(image_path: str | Path, config: dict[str, Any]) -> SimpleLayoutResult:
    """Entry point compatible with the pipeline."""
    return detect_text_lines_simple(image_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python simple_layout.py <image_path>")
        sys.exit(1)

    result = detect_text_lines_simple(sys.argv[1])
    print(f"Found {len(result.lines)} lines:")
    for line in result.lines:
        print(f"  Line {line.line_id}: bbox={line.bbox}")
