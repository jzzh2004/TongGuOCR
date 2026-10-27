from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TextLine:
    """One text line from the source txt file."""

    line_id: int
    text: str
    bbox: BBox

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class BlockStats:
    """Aggregated statistics for a candidate block."""

    bbox: BBox
    line_count: int
    sum_line_area: float
    union_area: float
    fill_ratio: float
    area_ratio: float
    aspect_ratio: float
    width: float
    height: float
    median_line_width: float
    median_line_height: float
    maturity_factor: float
    vertical_strip_ratio: float
    width_vs_median_line: float
    cross_axis_coverage_ratio: float
    main_axis_span_ratio: float
    avg_x_gap_norm: float
    max_x_gap_norm: float
    avg_y_overlap_ratio: float
    min_y_overlap_ratio: float
    avg_append_y_overlap_ratio: float
    min_append_y_overlap_ratio: float
    avg_main_axis_growth_ratio: float
    max_main_axis_growth_ratio: float
    avg_fill_ratio_drop: float
    max_fill_ratio_drop: float
    avg_union_area_growth_ratio: float
    max_union_area_growth_ratio: float
    internal_x_gap_spike: float
    mature_low_y_overlap_count: int
    max_mature_low_y_overlap_penalty_signal: float
    y_band_stability_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockCandidate:
    """A legal continuous interval that may become one output block."""

    candidate_id: int
    start_idx: int
    end_idx_exclusive: int
    bbox: BBox
    stats: BlockStats
    cost: float
    cost_breakdown: dict[str, float]

    def line_range_inclusive(self) -> list[int]:
        return [self.start_idx, self.end_idx_exclusive - 1]


@dataclass
class BlockResult:
    """Final block chosen by the DP solver."""

    block_id: int
    label: str
    line_range: list[int]
    line_ids: list[int]
    bbox: BBox
    stats: dict[str, Any]
    cost: float
    cost_breakdown: dict[str, float] | None = None

    def to_dict(self, include_debug: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "block_id": self.block_id,
            "label": self.label,
            "line_range": self.line_range,
            "line_ids": self.line_ids,
            "bbox": list(self.bbox),
            "stats": self.stats,
            "cost": self.cost,
        }
        if include_debug and self.cost_breakdown is not None:
            data["cost_breakdown"] = self.cost_breakdown
        return data


@dataclass
class PageResult:
    """Final output for one page."""

    page_id: str
    image_path: str
    txt_path: str
    image_size: tuple[int, int]
    block_count: int
    total_cost: float
    blocks: list[BlockResult]
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_debug: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "page_id": self.page_id,
            "image_path": self.image_path,
            "txt_path": self.txt_path,
            "image_size": list(self.image_size),
            "block_count": self.block_count,
            "total_cost": self.total_cost,
            "blocks": [block.to_dict(include_debug=include_debug) for block in self.blocks],
        }
        if include_debug and self.debug:
            data["debug"] = self.debug
        return data
