"""TongGuOCR pipeline: layout detection -> block segmentation -> block OCR.

Both OCR backends (``transformers`` in-process, ``vllm`` HTTP client) run
in-process; there is no subprocess orchestration. Per page:

1. detect rotated text lines (``tongguocr.layout``),
2. group them into recognition blocks (``tongguocr.blocking``),
3. crop each block (masking foreign text; ``tongguocr.crop``),
4. recognize each crop with the OCR backend (``tongguocr.ocr``),
5. join block texts with newlines and write per-page JSON + text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .blocking.dp_solver import segment_page
from .blocking.models import TextLine
from .config import load_config
from .io import (
    ensure_output_dirs,
    iter_input_images,
    read_layout_result,
    write_layout_result,
    write_page_result,
    write_summary,
    write_text_result,
)
from .layout import (
    LayoutPageResult,
    extract_line_bboxes,
    layout_page_result_from_dict,
    layout_page_result_to_dict,
)
from .models import OcrBlockResult, PipelinePageResult, PipelineRunSummary
from .ocr import recognize_blocks
from .crop import SavedBlockCrop, save_block_crops
from .visualize import draw_pipeline_visualization


LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedPage:
    page_id: str
    mode: str
    image_path: str
    image_size: tuple[int, int]
    line_bboxes: list[dict[str, Any]]
    line_count: int
    blocks: list[dict[str, Any]]
    block_count: int
    visual_path: str | None
    crop_dir: str | None
    crop_count: int
    debug_payload: dict[str, Any]


def setup_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def _layout_lines_to_block_lines(layout_page: LayoutPageResult) -> list[TextLine]:
    return [
        TextLine(line_id=line.line_id, text=line.text, bbox=line.bbox)
        for line in layout_page.lines
    ]


def _layout_lines_to_dicts(layout_page: LayoutPageResult) -> list[dict[str, Any]]:
    return [
        {
            "line_id": line.line_id,
            "text": line.text,
            "bbox": list(line.bbox),
        }
        for line in layout_page.lines
    ]


def _build_block_entries(
    page_result,
    crop_outputs: list[SavedBlockCrop],
) -> list[dict[str, Any]]:
    crop_map = {item.block_id: item for item in crop_outputs}
    block_entries: list[dict[str, Any]] = []
    for block in page_result.blocks:
        crop_output = crop_map.get(block.block_id)
        block_entries.append(
            {
                "block_id": block.block_id,
                "label": block.label,
                "bbox": list(block.bbox),
                "line_ids": block.line_ids,
                "line_range": block.line_range,
                "crop_path": str(crop_output.path) if crop_output else None,
                "masked_crop_path": str(crop_output.path) if crop_output else None,
                "background_fill_color": crop_output.background_fill_color if crop_output else None,
                "crop_canvas_size": crop_output.crop_canvas_size if crop_output else None,
                "sparse_canvas_applied": crop_output.sparse_canvas_applied if crop_output else False,
                "stats": block.stats,
                "cost": block.cost,
            }
        )
    return block_entries


def _compose_ocr_text(ocr_blocks: list[OcrBlockResult]) -> str:
    ordered_texts = [item.ocr_text.strip() for item in sorted(ocr_blocks, key=lambda item: item.block_id)]
    return "\n".join(text for text in ordered_texts if text)


def _build_page_preview(page_result: PipelinePageResult) -> dict[str, Any]:
    return {
        "page_id": page_result.page_id,
        "mode": page_result.mode,
        "backend": page_result.backend,
        "line_count": page_result.line_count,
        "block_count": page_result.block_count,
        "ocr_blocks": [
            {
                "block_id": block.block_id,
                "ocr_text": block.ocr_text,
            }
            for block in page_result.ocr_blocks
        ],
        "ocr_text": page_result.ocr_text,
    }


def segment_page_from_lines(
    block_lines: list[TextLine],
    image_size: tuple[int, int],
    config: dict[str, Any],
    *,
    page_id: str = "",
    image_path: str = "",
    txt_path: str = "",
    include_debug: bool = False,
):
    return segment_page(
        block_lines,
        image_size,
        config.get("blocking", config),
        page_id=page_id,
        image_path=image_path,
        txt_path=txt_path,
        include_debug=include_debug,
    )


def _process_one_image(
    image_path: str | Path,
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    *,
    debug: bool,
    visualize: bool,
    crop_blocks: bool,
    skip_layout: bool = False,
) -> tuple[PipelinePageResult, dict[str, Any]]:
    prepared = _prepare_one_image(
        image_path,
        config,
        output_dirs,
        debug=debug,
        visualize=visualize,
        crop_blocks=crop_blocks,
        skip_layout=skip_layout,
    )
    ocr_blocks = recognize_blocks(prepared.blocks, config)
    return _finalize_prepared_page(
        prepared,
        ocr_blocks,
        config,
        output_dirs,
        crop_blocks=crop_blocks,
    )


def _prepare_one_image(
    image_path: str | Path,
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    *,
    debug: bool,
    visualize: bool,
    crop_blocks: bool,
    skip_layout: bool = False,
) -> PreparedPage:
    image_path = Path(image_path)
    page_id = image_path.stem
    if skip_layout:
        layout_page = layout_page_result_from_dict(read_layout_result(output_dirs, page_id))
    else:
        layout_page = extract_line_bboxes(image_path, config)
        write_layout_result(output_dirs, page_id, layout_page_result_to_dict(layout_page))

    block_lines = _layout_lines_to_block_lines(layout_page)
    page_result = segment_page_from_lines(
        block_lines,
        layout_page.image_size,
        config,
        page_id=page_id,
        image_path=str(image_path),
        txt_path="",
        include_debug=debug,
    )

    visual_path = None
    if visualize:
        visual_path = draw_pipeline_visualization(
            image_path,
            block_lines,
            page_result,
            output_dirs["visuals"] / f"{page_id}.jpg",
            show_debug_metrics=debug,
        )

    effective_crop_dir = output_dirs["block_crops"] / page_id
    output_config = config["output"]
    crop_outputs = save_block_crops(
        image_path,
        page_result,
        block_lines,
        effective_crop_dir,
        padding=int(output_config["crop_padding"]),
        base_scale=float(output_config.get("crop_canvas", {}).get("base_scale", 1.25)),
        sparse_scale=float(output_config.get("crop_canvas", {}).get("sparse_scale", 1.5)),
        sparse_line_threshold=int(output_config.get("crop_canvas", {}).get("sparse_line_threshold", 2)),
        sparse_text_bbox_area_ratio=float(
            output_config.get("crop_canvas", {}).get("sparse_text_bbox_area_ratio", 0.55)
        ),
        crop_mode=str(output_config.get("crop_canvas", {}).get("mode", "padded_square")),
        enable_foreign_mask=bool(output_config.get("crop_mask", {}).get("enabled", True)),
    )
    block_entries = _build_block_entries(page_result, crop_outputs)
    final_crop_dir = str(effective_crop_dir) if crop_blocks else None
    LOGGER.info(
        "Prepared %s -> %s lines, %s blocks, OCR backend=%s",
        page_id,
        len(layout_page.lines),
        page_result.block_count,
        config["ocr"]["backend"],
    )
    return PreparedPage(
        page_id=page_id,
        mode="reuse-layout" if skip_layout else "full",
        image_path=str(image_path),
        image_size=layout_page.image_size,
        line_bboxes=_layout_lines_to_dicts(layout_page),
        line_count=len(layout_page.lines),
        blocks=block_entries,
        block_count=page_result.block_count,
        visual_path=str(visual_path) if visual_path else None,
        crop_dir=final_crop_dir,
        crop_count=len(crop_outputs),
        debug_payload=page_result.debug if debug else {},
    )


def _finalize_prepared_page(
    prepared: PreparedPage,
    ocr_blocks: list[OcrBlockResult],
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    *,
    crop_blocks: bool,
) -> tuple[PipelinePageResult, dict[str, Any]]:
    page_id = prepared.page_id
    effective_crop_dir = Path(prepared.crop_dir) if prepared.crop_dir else output_dirs["block_crops"] / page_id
    block_entries = prepared.blocks
    ocr_text = _compose_ocr_text(ocr_blocks)
    text_path = write_text_result(output_dirs["texts"], page_id, ocr_text)
    final_crop_dir = prepared.crop_dir
    if not crop_blocks:
        crop_paths = [item["crop_path"] for item in block_entries if item.get("crop_path")]
        for crop_path in crop_paths:
            path = Path(crop_path)
            if path.exists():
                path.unlink()
        if effective_crop_dir.exists():
            effective_crop_dir.rmdir()
        block_entries = [
            {
                **item,
                "crop_path": None,
                "masked_crop_path": None,
                "background_fill_color": None,
                "crop_canvas_size": None,
                "sparse_canvas_applied": False,
            }
            for item in block_entries
        ]
        ocr_blocks = [
            OcrBlockResult(
                block_id=item.block_id,
                crop_path="",
                masked_crop_path=None,
                ocr_text=item.ocr_text,
                backend=item.backend,
                latency_ms=item.latency_ms,
                ocr_text_tokens=item.ocr_text_tokens,
            )
            for item in ocr_blocks
        ]
        final_crop_dir = None

    pipeline_result = PipelinePageResult(
        page_id=page_id,
        mode=prepared.mode,
        image_path=prepared.image_path,
        image_size=prepared.image_size,
        status="ok",
        backend=config["ocr"]["backend"],
        line_bboxes=prepared.line_bboxes,
        line_count=prepared.line_count,
        blocks=block_entries,
        block_count=prepared.block_count,
        ocr_blocks=ocr_blocks,
        ocr_text=ocr_text,
        text_path=str(text_path),
        visual_path=prepared.visual_path,
        crop_dir=final_crop_dir,
        debug=prepared.debug_payload,
    )
    page_json_path = write_page_result(output_dirs["pages"], pipeline_result.to_dict(include_debug=bool(prepared.debug_payload)))

    LOGGER.info(
        "Processed %s -> %s lines, %s blocks, OCR backend=%s",
        prepared.page_id,
        prepared.line_count,
        prepared.block_count,
        config["ocr"]["backend"],
    )
    return pipeline_result, {
        "status": "ok",
        "page_id": prepared.page_id,
        "image_path": prepared.image_path,
        "line_count": prepared.line_count,
        "block_count": prepared.block_count,
        "backend": config["ocr"]["backend"],
        "page_json_path": str(page_json_path),
        "text_path": str(text_path),
        "visual_path": prepared.visual_path,
        "crop_dir": final_crop_dir,
        "crop_count": prepared.crop_count if crop_blocks else 0,
    }


def run_pipeline_on_image(
    image_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    debug: bool = False,
    visualize: bool | None = None,
    crop_blocks: bool | None = None,
    skip_layout: bool = False,
) -> PipelinePageResult:
    output_dirs = ensure_output_dirs(output_dir)
    visualize = config["output"]["visualize"] if visualize is None else visualize
    crop_blocks = config["output"]["crop_blocks"] if crop_blocks is None else crop_blocks
    page_result, _ = _process_one_image(
        image_path,
        config,
        output_dirs,
        debug=debug,
        visualize=visualize,
        crop_blocks=crop_blocks,
        skip_layout=skip_layout,
    )
    return page_result


def run_pipeline_on_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    debug: bool = False,
    visualize: bool | None = None,
    crop_blocks: bool | None = None,
    limit: int | None = None,
    skip_layout: bool = False,
    config_path: str | None = None,
) -> PipelineRunSummary:
    output_dirs = ensure_output_dirs(output_dir)
    image_paths = iter_input_images(input_dir=input_dir, limit=limit)
    visualize = config["output"]["visualize"] if visualize is None else visualize
    crop_blocks = config["output"]["crop_blocks"] if crop_blocks is None else crop_blocks

    pages: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []
    for image_path in image_paths:
        try:
            pages.append(
                _process_one_image(
                    image_path,
                    config,
                    output_dirs,
                    debug=debug,
                    visualize=visualize,
                    crop_blocks=crop_blocks,
                    skip_layout=skip_layout,
                )[1]
            )
        except Exception as exc:  # pragma: no cover - defensive batch safety
            LOGGER.exception("Failed to process %s", image_path)
            failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})

    summary = PipelineRunSummary(
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        config_path=config_path,
        page_count=len(pages),
        failed_page_count=len(failed_pages),
        pages=sorted(pages, key=lambda item: item["page_id"]),
        failed_pages=sorted(failed_pages, key=lambda item: item["page_id"]),
    )
    write_summary(output_dirs["root"], summary.to_dict())
    return summary


def run_from_args(args) -> tuple[dict[str, Any], dict[str, Any] | None]:
    output_dirs = ensure_output_dirs(args.output_dir)
    setup_logging(output_dirs["logs"] / "run.log")
    config = load_config(args.config)
    if args.backend:
        config["ocr"]["backend"] = args.backend
    if args.device:
        config["layout_inference"]["device"] = args.device
        config["ocr"]["transformers"]["device"] = args.device

    visualize = config["output"]["visualize"] if args.visualize is None else args.visualize
    crop_blocks = config["output"]["crop_blocks"]
    config_path = str(args.config) if args.config else None
    LOGGER.info(
        "Backend=%s, layout device=%s",
        config["ocr"]["backend"],
        config["layout_inference"]["device"],
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        summary = run_pipeline_on_dir(
            input_path,
            args.output_dir,
            config,
            visualize=visualize,
            crop_blocks=crop_blocks,
            limit=args.limit,
            skip_layout=args.skip_layout,
            config_path=config_path,
        )
        return summary.to_dict(), None

    page_result = run_pipeline_on_image(
        input_path,
        args.output_dir,
        config,
        visualize=visualize,
        crop_blocks=crop_blocks,
        skip_layout=args.skip_layout,
    )
    page_summary = {
        "status": page_result.status,
        "page_id": page_result.page_id,
        "mode": page_result.mode,
        "image_path": page_result.image_path,
        "line_count": page_result.line_count,
        "block_count": page_result.block_count,
        "backend": page_result.backend,
        "page_json_path": str(output_dirs["pages"] / f"{page_result.page_id}.json"),
        "text_path": page_result.text_path,
        "visual_path": page_result.visual_path,
        "crop_dir": page_result.crop_dir,
        "crop_count": len(page_result.blocks) if page_result.crop_dir else 0,
    }
    summary = PipelineRunSummary(
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        config_path=config_path,
        page_count=1,
        failed_page_count=0,
        pages=[page_summary],
        failed_pages=[],
    )
    summary_dict = summary.to_dict()
    write_summary(output_dirs["root"], summary_dict)
    return summary_dict, _build_page_preview(page_result)
