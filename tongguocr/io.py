from __future__ import annotations

import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    intermediate = root / "intermediate"
    paths = {
        "root": root,
        "pages": root / "pages",
        "texts": root / "texts",
        "block_crops": root / "block_crops",
        "visuals": root / "visuals",
        "logs": root / "logs",
        "intermediate": intermediate,
        "intermediate_layout": intermediate / "layout",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def iter_input_images(
    image_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    *,
    limit: int | None = None,
) -> list[Path]:
    if bool(image_path) == bool(input_dir):
        raise ValueError("Provide exactly one of image_path or input_dir")
    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        return [path]
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing input dir: {root}")
    images = [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if limit is not None:
        images = images[:limit]
    if not images:
        LOGGER.warning("No input images found in %s", root)
    return images


def write_page_result(output_dir: str | Path, page_result: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{page_result['page_id']}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(page_result, handle, ensure_ascii=False, indent=2)
    return out_path


def write_text_result(output_dir: str | Path, page_id: str, text: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{page_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_summary(output_dir: str | Path, summary: dict) -> Path:
    out_path = Path(output_dir) / "summary.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return out_path


def write_layout_result(output_dirs: dict[str, Path], page_id: str, layout_data: dict) -> Path:
    out_path = output_dirs["intermediate_layout"] / f"{page_id}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(layout_data, handle, ensure_ascii=False, indent=2)
    return out_path


def read_layout_result(output_dirs: dict[str, Path], page_id: str) -> dict:
    path = output_dirs["intermediate_layout"] / f"{page_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
