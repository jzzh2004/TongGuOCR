"""TongGuOCR: layout-to-block-to-OCR pipeline for classical Chinese documents."""

from .config import load_config
from .models import OcrBlockResult, PipelinePageResult, PipelineRunSummary
from .ocr import recognize_blocks
from .pipeline import run_pipeline_on_dir, run_pipeline_on_image

__all__ = [
    "load_config",
    "recognize_blocks",
    "run_pipeline_on_dir",
    "run_pipeline_on_image",
    "OcrBlockResult",
    "PipelinePageResult",
    "PipelineRunSummary",
]
