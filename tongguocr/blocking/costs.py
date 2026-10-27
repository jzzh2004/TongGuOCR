from __future__ import annotations

from typing import Any

from .geometry import containment_ratio, fill_ratio, interval_gap, iou, union_bbox, y_band_overlap_ratio
from .models import BlockCandidate, BlockStats


REFERENCE_LAMBDAS = {
    "lambda_blk": 2.0,
    "lambda_adj": 2.0,
    "lambda_page": 1.0,
}


def lambda_weights(config: dict[str, Any]) -> dict[str, float]:
    """Return the public paper-level lambda weights."""

    configured = config.get("lambda_weights", {})
    return {
        name: float(configured.get(name, reference))
        for name, reference in REFERENCE_LAMBDAS.items()
    }


def lambda_scales(config: dict[str, Any]) -> dict[str, float]:
    """Return scales relative to the released 2/2/1 behavior."""

    weights = lambda_weights(config)
    return {
        "block_scale": weights["lambda_blk"] / REFERENCE_LAMBDAS["lambda_blk"],
        "adj_scale": weights["lambda_adj"] / REFERENCE_LAMBDAS["lambda_adj"],
        "page_scale": weights["lambda_page"] / REFERENCE_LAMBDAS["lambda_page"],
    }


def block_cost_scale(config: dict[str, Any]) -> float:
    return lambda_scales(config)["block_scale"]


def quadratic_penalty(value: float, low: float | None = None, high: float | None = None) -> float:
    """Quadratic penalty when a scalar leaves a preferred interval."""

    if low is not None and value < low:
        delta = (low - value) / max(abs(low), 1e-6)
        return delta * delta
    if high is not None and value > high:
        delta = (value - high) / max(abs(high), 1e-6)
        return delta * delta
    return 0.0


def line_count_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    threshold = config["thresholds"]
    penalty = 0.0
    if stats.line_count < threshold["preferred_min_lines"]:
        penalty += quadratic_penalty(
            float(stats.line_count),
            low=float(threshold["preferred_min_lines"]),
        )
    return penalty


def tiny_block_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    thresholds = config["thresholds"]
    penalty = 0.0
    if stats.line_count <= thresholds["min_lines_hard"]:
        distance = float(thresholds["preferred_min_lines"] - stats.line_count + 1)
        penalty += distance
        if stats.line_count <= max(1, thresholds["min_lines_hard"] - 1):
            penalty += distance * distance
    penalty += quadratic_penalty(
        stats.area_ratio,
        low=thresholds["tiny_block_area_ratio_soft"],
    ) * (2.0 if stats.line_count <= 2 else 1.0)
    return penalty


def hard_constraint_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    threshold = config["thresholds"]
    penalty = 0.0
    penalty += quadratic_penalty(float(stats.line_count), low=float(threshold["min_lines_hard"]))
    penalty += quadratic_penalty(stats.area_ratio, high=threshold["block_area_ratio_hard_max"])
    penalty += quadratic_penalty(stats.aspect_ratio, high=threshold["aspect_ratio_hard"])
    penalty += quadratic_penalty(stats.fill_ratio, low=threshold["min_fill_ratio_hard"])
    penalty += quadratic_penalty(stats.vertical_strip_ratio, high=threshold["vertical_strip_ratio_hard"])
    penalty += quadratic_penalty(stats.width_vs_median_line, low=threshold["width_vs_median_line_hard"])
    penalty += quadratic_penalty(
        stats.cross_axis_coverage_ratio,
        low=threshold["cross_axis_coverage_ratio_hard_min"],
    )
    penalty += quadratic_penalty(stats.main_axis_span_ratio, high=threshold["main_axis_span_ratio_hard"])
    penalty += quadratic_penalty(stats.internal_x_gap_spike, high=threshold["internal_x_gap_spike_hard"])
    return penalty


def shape_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    return quadratic_penalty(
        stats.aspect_ratio,
        high=config["thresholds"]["aspect_ratio_soft"],
    )


def area_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    threshold = config["thresholds"]
    penalty = quadratic_penalty(stats.area_ratio, low=threshold["block_area_ratio_soft_min"])
    penalty += quadratic_penalty(stats.area_ratio, high=threshold["block_area_ratio_soft_max"])
    return penalty


def fill_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    return quadratic_penalty(stats.fill_ratio, low=config["thresholds"]["min_fill_ratio_soft"])


def adjacency_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    threshold = config["thresholds"]
    penalty = quadratic_penalty(stats.avg_x_gap_norm, high=threshold["x_gap_norm_soft"])
    penalty += quadratic_penalty(stats.max_x_gap_norm, high=threshold["x_gap_norm_soft"])
    penalty += quadratic_penalty(stats.internal_x_gap_spike, high=threshold["internal_x_gap_spike_soft"])
    penalty += quadratic_penalty(
        stats.avg_y_overlap_ratio,
        low=threshold["y_overlap_ratio_soft"],
    )
    penalty += quadratic_penalty(
        stats.min_y_overlap_ratio,
        low=threshold["y_overlap_ratio_soft"],
    )
    penalty += quadratic_penalty(
        stats.avg_append_y_overlap_ratio,
        low=threshold["y_overlap_ratio_soft"],
    )
    penalty += quadratic_penalty(
        stats.min_append_y_overlap_ratio,
        low=threshold["y_overlap_ratio_soft"],
    )
    return penalty


def strip_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    threshold = config["thresholds"]
    penalty = quadratic_penalty(stats.vertical_strip_ratio, high=threshold["vertical_strip_ratio_soft"])
    penalty += quadratic_penalty(stats.width_vs_median_line, low=threshold["width_vs_median_line_soft"])
    penalty += quadratic_penalty(
        stats.cross_axis_coverage_ratio,
        low=threshold["cross_axis_coverage_ratio_soft_min"],
    )
    if stats.line_count <= threshold["preferred_min_lines"]:
        penalty *= 1.0 + 0.5 * max(0, threshold["preferred_min_lines"] - stats.line_count)
    return penalty


def mature_y_overlap_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    thresholds = config["thresholds"]
    penalty = float(stats.mature_low_y_overlap_count)
    penalty += stats.max_mature_low_y_overlap_penalty_signal
    if stats.line_count >= thresholds["mature_block_line_count"]:
        penalty += quadratic_penalty(
            stats.y_band_stability_score,
            low=thresholds["y_band_stability_soft"],
        ) * 0.25
    return penalty


def growth_penalty(stats: BlockStats, config: dict[str, Any]) -> float:
    threshold = config["thresholds"]
    penalty = quadratic_penalty(
        stats.avg_fill_ratio_drop,
        high=threshold["fill_drop_soft"],
    )
    penalty += quadratic_penalty(
        stats.max_fill_ratio_drop,
        high=threshold["fill_drop_soft"],
    )
    penalty += quadratic_penalty(
        stats.max_union_area_growth_ratio,
        high=threshold["union_area_growth_ratio_soft"],
    )
    penalty += quadratic_penalty(
        stats.max_main_axis_growth_ratio,
        high=threshold["main_axis_growth_ratio_soft"],
    )
    return penalty * stats.maturity_factor


PENALTY_REGISTRY = {
    "line_count_penalty": line_count_penalty,
    "tiny_block_penalty": tiny_block_penalty,
    "hard_constraint_penalty": hard_constraint_penalty,
    "shape_penalty": shape_penalty,
    "area_penalty": area_penalty,
    "fill_penalty": fill_penalty,
    "adjacency_penalty": adjacency_penalty,
    "strip_penalty": strip_penalty,
    "mature_y_overlap_penalty": mature_y_overlap_penalty,
    "growth_penalty": growth_penalty,
}


def compute_candidate_cost(stats: BlockStats, config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Return total candidate cost and a per-term breakdown."""

    weights = config["weights"]
    breakdown: dict[str, float] = {"open_block_cost": weights["open_block_cost"]}
    total = breakdown["open_block_cost"]
    for name, func in PENALTY_REGISTRY.items():
        raw = func(stats, config)
        weighted = raw * weights.get(name, 1.0)
        breakdown[name] = weighted
        total += weighted
    scale = block_cost_scale(config)
    breakdown["lambda_blk_scale"] = scale
    breakdown["block_group_unscaled_total"] = total
    breakdown["block_group_scaled_total"] = total * scale
    return total * scale, breakdown


def page_block_count_penalty(block_count: int, config: dict[str, Any]) -> float:
    low, high = config["thresholds"]["target_block_count_range"]
    weight = config["weights"]["page_block_count_penalty"]
    penalty = quadratic_penalty(float(block_count), low=float(low), high=float(high))
    if block_count == 1:
        penalty += config["weights"]["single_block_page_penalty"] / max(weight, 1e-6)
    if block_count > high:
        penalty += quadratic_penalty(
            float(block_count),
            high=float(config["dp"]["soft_max_blocks_per_page"]),
        ) * 2.0
    return penalty * weight * lambda_scales(config)["page_scale"]


def compute_transition_cost(
    prev_candidate: BlockCandidate,
    candidate: BlockCandidate,
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Return cost for placing two adjacent blocks next to each other."""

    thresholds = config["thresholds"]
    weights = config["weights"]
    overlap_iou = iou(prev_candidate.bbox, candidate.bbox)
    contain_ratio = containment_ratio(prev_candidate.bbox, candidate.bbox)
    small_line_count = min(prev_candidate.stats.line_count, candidate.stats.line_count)
    merged_bbox = union_bbox(prev_candidate.bbox, candidate.bbox)
    merged_height = max(merged_bbox[3] - merged_bbox[1], 1e-6)
    merged_width = max(merged_bbox[2] - merged_bbox[0], 1e-6)
    merged_vertical_strip_ratio = merged_height / merged_width
    split_strip_improvement = max(
        0.0,
        max(prev_candidate.stats.vertical_strip_ratio, candidate.stats.vertical_strip_ratio) - merged_vertical_strip_ratio,
    )
    y_band_overlap = y_band_overlap_ratio(prev_candidate.bbox, candidate.bbox)
    x_gap_norm = interval_gap(
        prev_candidate.bbox[0],
        prev_candidate.bbox[2],
        candidate.bbox[0],
        candidate.bbox[2],
    ) / max(prev_candidate.stats.median_line_width, 1e-6)
    bad_split_signal = (
        quadratic_penalty(y_band_overlap, low=thresholds["y_band_stability_soft"])
        * 0.0
    )
    if y_band_overlap >= thresholds["y_band_stability_soft"]:
        bad_split_signal = split_strip_improvement * max(
            0.0,
            1.0 - (x_gap_norm / max(thresholds["x_gap_norm_soft"], 1e-6)),
        )
    breakdown = {
        "overlap_iou": overlap_iou,
        "containment_ratio": contain_ratio,
        "y_band_overlap": y_band_overlap,
        "split_strip_improvement": split_strip_improvement,
        "transition_overlap_penalty": quadratic_penalty(
            overlap_iou,
            high=thresholds["transition_overlap_iou_soft"],
        )
        * weights["transition_overlap_penalty"],
        "transition_containment_penalty": quadratic_penalty(
            contain_ratio,
            high=thresholds["transition_containment_soft"],
        )
        * weights["transition_containment_penalty"],
        "tiny_inside_large_penalty": 0.0,
        "transition_hard_penalty": 0.0,
        "bad_split_penalty": bad_split_signal * weights["bad_split_penalty"],
    }
    if small_line_count <= thresholds["min_lines_hard"]:
        breakdown["tiny_inside_large_penalty"] = quadratic_penalty(
            contain_ratio,
            high=thresholds["transition_containment_soft"],
        ) * weights["tiny_inside_large_penalty"]
    breakdown["transition_hard_penalty"] = (
        quadratic_penalty(contain_ratio, high=thresholds["transition_containment_hard"])
        + quadratic_penalty(overlap_iou, high=thresholds["transition_overlap_iou_hard"])
    ) * weights["transition_hard_penalty"]
    total = sum(breakdown.values())
    scale = lambda_scales(config)["adj_scale"]
    breakdown["lambda_adj_scale"] = scale
    breakdown["adj_group_unscaled_total"] = total
    breakdown["adj_group_scaled_total"] = total * scale
    return total * scale, breakdown
