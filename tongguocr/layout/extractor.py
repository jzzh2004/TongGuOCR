from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class LayoutLine:
    line_id: int
    text: str
    bbox: tuple[float, float, float, float]
    polygon: tuple[float, ...] = ()


@dataclass(frozen=True)
class LayoutPageResult:
    page_id: str
    image_path: str
    image_size: tuple[int, int]
    lines: list[LayoutLine]

    def to_csv_rows(self) -> list[list[str | float]]:
        rows: list[list[str | float]] = []
        for line in self.lines:
            x1, y1, x2, y2 = line.bbox
            rows.append([x1, y1, x2, y2, line.text])
        return rows


def layout_page_result_to_dict(page_result: LayoutPageResult) -> dict[str, Any]:
    return {
        "page_id": page_result.page_id,
        "image_path": page_result.image_path,
        "image_size": list(page_result.image_size),
        "lines": [
            {
                "line_id": line.line_id,
                "text": line.text,
                "bbox": list(line.bbox),
                "polygon": list(line.polygon),
            }
            for line in page_result.lines
        ],
    }


def layout_page_result_from_dict(payload: dict[str, Any]) -> LayoutPageResult:
    return LayoutPageResult(
        page_id=str(payload["page_id"]),
        image_path=str(payload["image_path"]),
        image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
        lines=[
            LayoutLine(
                line_id=int(line["line_id"]),
                text=str(line.get("text") or ""),
                bbox=tuple(float(value) for value in line["bbox"]),
                polygon=tuple(float(value) for value in line.get("polygon", [])),
            )
            for line in payload["lines"]
        ],
    )


_EXTRACTOR_CACHE: dict[str, LineBBoxExtractor] = {}


class LineBBoxExtractor:
    """Adapter around the layout-only detector pipeline."""

    def __init__(self, config: dict[str, Any]):
        from .detector import LayoutInfer

        self.config = config
        self.infer = LayoutInfer(config)

    @staticmethod
    def _to_bbox(box: list[Any]) -> tuple[int, int, int, int] | None:
        coords: list[float] = []
        for value in box[:-1]:
            try:
                coords.append(float(value))
            except (TypeError, ValueError):
                continue
        if len(coords) < 4:
            return None
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return None
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    @staticmethod
    def _fallback_full_image_line(image_size: tuple[int, int]) -> LayoutLine:
        width, height = image_size
        return LayoutLine(
            line_id=0,
            text="",
            bbox=(0, 0, int(width), int(height)),
            polygon=(0.0, 0.0, float(width), 0.0, float(width), float(height), 0.0, float(height)),
        )

    def extract(self, image_path: str | Path) -> LayoutPageResult:
        image_path = str(image_path)
        img, img_info, grid, org_w, org_h = self.infer.dewarp(image_path)
        det_polys, det_segs = self.infer.det_eval(img_info)
        _, _, all_points_batch, width, height = self.infer.det_recg_connect(img, det_polys, det_segs)

        points_batch: list[list[float]] = []
        for points_group in all_points_batch:
            for points in points_group:
                points_batch.append(points)

        points_batch = self.infer.dewarper.poly_recovery(points_batch, grid, org_w, org_h)
        text_batch = [""] * len(points_batch)
        ordered_boxes = self.infer.order_eval(points_batch, text_batch, width, height)

        with Image.open(image_path) as img_handle:
            image_size = img_handle.size

        lines: list[LayoutLine] = []
        for idx, box in enumerate(ordered_boxes):
            bbox = self._to_bbox(box)
            if bbox is None:
                continue
            polygon = tuple(float(value) for value in box[:-1])
            text = str(box[-1]) if box else ""
            lines.append(LayoutLine(line_id=len(lines), text=text, bbox=bbox, polygon=polygon))

        if not lines:
            lines = [self._fallback_full_image_line(image_size)]

        return LayoutPageResult(
            page_id=Path(image_path).stem,
            image_path=image_path,
            image_size=image_size,
            lines=lines,
        )

    @staticmethod
    def write_csv(page_result: LayoutPageResult, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for row in page_result.to_csv_rows():
                writer.writerow(row)
        return output_path


def extract_line_bboxes(image_path: str | Path, config: dict[str, Any]) -> LayoutPageResult:
    cache_key = "|".join(
        [
            str(config["layout_inference"]["detector_config_path"]),
            str(config["layout_inference"]["detector_checkpoint_path"]),
            str(config["layout_inference"]["device"]),
            str(config["layout_inference"]["detector_score_threshold"]),
        ]
    )
    extractor = _EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor = LineBBoxExtractor(config)
        _EXTRACTOR_CACHE[cache_key] = extractor
    return extractor.extract(image_path)


def clear_layout_extractor_cache() -> None:
    _EXTRACTOR_CACHE.clear()


def extract_line_bboxes_to_json(
    image_path: str | Path,
    config: dict[str, Any],
    output_dirs: dict[str, Path],
) -> dict[str, Any]:
    from ..io import write_layout_result

    page_result = extract_line_bboxes(image_path, config)
    payload = layout_page_result_to_dict(page_result)
    write_layout_result(output_dirs, page_result.page_id, payload)
    return payload
