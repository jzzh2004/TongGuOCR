from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OcrBlockResult:
    block_id: int
    crop_path: str
    masked_crop_path: str | None
    ocr_text: str
    backend: str
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelinePageResult:
    page_id: str
    mode: str
    image_path: str
    image_size: tuple[int, int]
    status: str
    backend: str
    line_bboxes: list[dict[str, Any]]
    line_count: int
    blocks: list[dict[str, Any]]
    block_count: int
    ocr_blocks: list[OcrBlockResult]
    ocr_text: str
    text_path: str | None = None
    visual_path: str | None = None
    crop_dir: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_debug: bool = False) -> dict[str, Any]:
        data = {
            "page_id": self.page_id,
            "mode": self.mode,
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "status": self.status,
            "backend": self.backend,
            "line_bboxes": self.line_bboxes,
            "line_count": self.line_count,
            "blocks": self.blocks,
            "block_count": self.block_count,
            "ocr_blocks": [item.to_dict() for item in self.ocr_blocks],
            "ocr_text": self.ocr_text,
            "text_path": self.text_path,
            "visual_path": self.visual_path,
            "crop_dir": self.crop_dir,
        }
        if include_debug and self.debug:
            data["debug"] = self.debug
        return data


@dataclass
class PipelineRunSummary:
    generated_at: str
    config_path: str
    page_count: int
    failed_page_count: int
    pages: list[dict[str, Any]]
    failed_pages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
