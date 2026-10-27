"""Utilities for aggregating ordered text lines into text blocks."""

from .config import load_config
from .dp_solver import segment_page

__all__ = ["load_config", "segment_page"]
