"""Render the TongGuOCR paper's figures and tables into web-ready assets.

Usage:
  python scripts/extract-paper-assets.py "path/to/TongguOCR_journal (5).pdf"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium


SCALE = 4

# Crop boxes are expressed in PDF points: (left, top, right, bottom).
ASSETS = {
    "figures/figure-1.webp": (2, (39, 54, 556, 524), "WEBP"),
    "figures/figure-2.png": (5, (58, 55, 538, 350), "PNG"),
    "figures/figure-3.png": (7, (58, 54, 538, 334), "PNG"),
    "figures/figure-4.webp": (15, (42, 72, 553, 631), "WEBP"),
    "tables/table-1.png": (9, (64, 67, 532, 151), "PNG"),
    "tables/table-2.png": (11, (64, 68, 532, 191), "PNG"),
    "tables/table-3.png": (12, (72, 67, 524, 505), "PNG"),
    "tables/table-4.png": (13, (64, 68, 532, 526), "PNG"),
    "tables/table-5.png": (16, (72, 68, 524, 210), "PNG"),
    "tables/table-6.png": (16, (72, 211, 524, 409), "PNG"),
    "tables/table-7.png": (17, (64, 68, 532, 217), "PNG"),
    "tables/table-8.png": (17, (64, 214, 532, 376), "PNG"),
    "tables/table-9.png": (18, (72, 68, 524, 212), "PNG"),
}


def px_box(points: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(round(value * SCALE) for value in points)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Pass the source PDF path as the only argument.")

    source = Path(sys.argv[1]).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"PDF not found: {source}")

    output_root = Path(__file__).resolve().parents[1] / "public" / "paper-assets"
    document = pdfium.PdfDocument(str(source))

    for relative_path, (page_number, crop_box, image_format) in ASSETS.items():
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        page = document[page_number - 1]
        rendered = page.render(scale=SCALE).to_pil().convert("RGB")
        cropped = rendered.crop(px_box(crop_box))

        save_options = {"optimize": True}
        if image_format == "WEBP":
            save_options.update({"quality": 92, "method": 6})
        cropped.save(output_path, format=image_format, **save_options)
        print(f"{relative_path}: {cropped.width}x{cropped.height}")


if __name__ == "__main__":
    main()
