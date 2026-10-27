from __future__ import annotations

from statistics import mean
from typing import Any

from .costs import block_cost_scale, compute_candidate_cost, quadratic_penalty
from .geometry import (
    adjacency_features,
    aspect_ratio,
    bbox_area,
    bbox_height,
    bbox_width,
    fill_ratio,
    median_line_size,
    union_bbox,
    y_band_overlap_ratio,
    y_band_stability_score,
)
from .models import BlockCandidate, BlockStats, TextLine


def _prune_start_bucket(bucket: list[BlockCandidate], limit: int) -> list[BlockCandidate]:
    if len(bucket) <= limit:
        return bucket
    cheap_candidates = sorted(
        bucket,
        key=lambda candidate: (
            candidate.cost / max(candidate.stats.line_count, 1),
            candidate.cost,
            -candidate.stats.line_count,
        ),
    )[: max(1, limit // 2)]
    long_candidates = sorted(
        bucket,
        key=lambda candidate: (
            -(candidate.end_idx_exclusive - candidate.start_idx),
            candidate.cost,
        ),
    )[: max(1, limit // 2)]
    selected: dict[int, BlockCandidate] = {}
    for candidate in cheap_candidates + long_candidates:
        selected[candidate.candidate_id] = candidate
    if len(selected) < limit:
        for candidate in sorted(bucket, key=lambda candidate: (candidate.cost, -candidate.stats.line_count)):
            selected.setdefault(candidate.candidate_id, candidate)
            if len(selected) >= limit:
                break
    return sorted(selected.values(), key=lambda candidate: (candidate.end_idx_exclusive, candidate.cost))


def _should_break(
    x_gap_norm: float,
    union_area_ratio: float,
    current_fill_ratio: float,
    main_axis_growth_ratio: float,
    line_count: int,
    config: dict[str, Any],
) -> bool:
    threshold = config["thresholds"]
    maturity_factor = min(1.0, line_count / max(threshold["preferred_min_lines"], 1))
    return (
        x_gap_norm > threshold["x_gap_norm_hard"]
        or union_area_ratio > threshold["block_area_ratio_hard_max"]
        or current_fill_ratio < threshold["min_fill_ratio_hard"] * max(0.5, maturity_factor)
        or (
            line_count >= threshold["preferred_min_lines"]
            and main_axis_growth_ratio > threshold["main_axis_growth_ratio_hard"]
        )
    )


def _append_y_overlap_signal(
    append_y_overlap_ratio: float,
    prior_line_count: int,
    prior_stability_score: float,
    config: dict[str, Any],
) -> tuple[float, bool]:
    thresholds = config["thresholds"]
    base = quadratic_penalty(append_y_overlap_ratio, low=thresholds["mature_y_overlap_soft"])
    if base <= 0.0:
        return 0.0, False
    mature_line_count = thresholds["mature_block_line_count"]
    stability_ready = prior_stability_score >= thresholds["y_band_stability_soft"]
    if prior_line_count <= 2:
        return base * 0.2, False
    if prior_line_count < mature_line_count:
        return base * 0.7, False
    if stability_ready:
        return base * (2.0 + prior_stability_score), True
    return base * 1.1, False


def _make_stats(
    union: tuple[float, float, float, float],
    sum_line_area: float,
    page_area: float,
    image_size: tuple[int, int],
    median_width: float,
    median_height: float,
    x_gaps: list[float],
    y_overlaps: list[float],
    append_y_overlaps: list[float],
    fill_drops: list[float],
    area_growths: list[float],
    main_axis_growths: list[float],
    line_bboxes: list[tuple[float, float, float, float]],
    mature_low_y_overlap_count: int,
    append_y_overlap_signals: list[float],
    config: dict[str, Any],
) -> BlockStats:
    union_area = bbox_area(union)
    width = bbox_width(union)
    height = bbox_height(union)
    line_count = len(line_bboxes)
    maturity_factor = min(1.0, line_count / max(config["thresholds"]["preferred_min_lines"], 1))
    return BlockStats(
        bbox=union,
        line_count=line_count,
        sum_line_area=sum_line_area,
        union_area=union_area,
        fill_ratio=fill_ratio(sum_line_area, union),
        area_ratio=union_area / max(page_area, 1e-6),
        aspect_ratio=aspect_ratio(union),
        width=width,
        height=height,
        median_line_width=median_width,
        median_line_height=median_height,
        maturity_factor=maturity_factor,
        vertical_strip_ratio=height / max(width, 1e-6),
        width_vs_median_line=width / max(median_width, 1e-6),
        cross_axis_coverage_ratio=width / max(image_size[0], 1e-6),
        main_axis_span_ratio=height / max(image_size[1], 1e-6),
        avg_x_gap_norm=mean(x_gaps) if x_gaps else 0.0,
        max_x_gap_norm=max(x_gaps) if x_gaps else 0.0,
        avg_y_overlap_ratio=mean(y_overlaps) if y_overlaps else 1.0,
        min_y_overlap_ratio=min(y_overlaps) if y_overlaps else 1.0,
        avg_append_y_overlap_ratio=mean(append_y_overlaps) if append_y_overlaps else 1.0,
        min_append_y_overlap_ratio=min(append_y_overlaps) if append_y_overlaps else 1.0,
        avg_main_axis_growth_ratio=mean(main_axis_growths) if main_axis_growths else 0.0,
        max_main_axis_growth_ratio=max(main_axis_growths) if main_axis_growths else 0.0,
        avg_fill_ratio_drop=mean(fill_drops) if fill_drops else 0.0,
        max_fill_ratio_drop=max(fill_drops) if fill_drops else 0.0,
        avg_union_area_growth_ratio=mean(area_growths) if area_growths else 1.0,
        max_union_area_growth_ratio=max(area_growths) if area_growths else 1.0,
        internal_x_gap_spike=max(x_gaps) if x_gaps else 0.0,
        mature_low_y_overlap_count=mature_low_y_overlap_count,
        max_mature_low_y_overlap_penalty_signal=max(append_y_overlap_signals) if append_y_overlap_signals else 0.0,
        y_band_stability_score=y_band_stability_score(line_bboxes, median_height),
    )


def _build_range_stats(
    lines: list[TextLine],
    image_size: tuple[int, int],
    config: dict[str, Any],
    median_width: float,
    median_height: float,
) -> BlockStats:
    page_area = float(image_size[0] * image_size[1])
    union = lines[0].bbox
    sum_line_area = 0.0
    x_gaps: list[float] = []
    y_overlaps: list[float] = []
    append_y_overlaps: list[float] = []
    fill_drops: list[float] = []
    area_growths: list[float] = []
    main_axis_growths: list[float] = []
    append_y_overlap_signals: list[float] = []
    line_bboxes: list[tuple[float, float, float, float]] = []
    mature_low_y_overlap_count = 0
    prev_fill_ratio = 1.0
    prev_union_area = 0.0

    for idx, line in enumerate(lines):
        prev_union = union
        prior_line_count = len(line_bboxes)
        prior_stability_score = y_band_stability_score(line_bboxes, median_height) if line_bboxes else 1.0
        union = line.bbox if idx == 0 else union_bbox(union, line.bbox)
        sum_line_area += line.area
        union_area = bbox_area(union)
        current_fill_ratio = fill_ratio(sum_line_area, union)
        main_axis_growth_ratio = (
            max(0.0, bbox_height(union) - bbox_height(prev_union)) / max(median_height, 1e-6)
            if idx > 0
            else 0.0
        )

        if idx > 0:
            x_gap_norm, y_overlap_ratio = adjacency_features(lines[idx - 1].bbox, line.bbox, median_width)
            append_y_overlap_ratio = y_band_overlap_ratio(prev_union, line.bbox)
            append_signal, mature_hit = _append_y_overlap_signal(
                append_y_overlap_ratio,
                prior_line_count,
                prior_stability_score,
                config,
            )
            x_gaps.append(x_gap_norm)
            y_overlaps.append(y_overlap_ratio)
            append_y_overlaps.append(append_y_overlap_ratio)
            append_y_overlap_signals.append(append_signal)
            if mature_hit:
                mature_low_y_overlap_count += 1

        if prev_union_area > 0:
            fill_drops.append(max(0.0, prev_fill_ratio - current_fill_ratio))
            area_growths.append((union_area - prev_union_area) / max(line.area, 1e-6))
            main_axis_growths.append(main_axis_growth_ratio)
        else:
            fill_drops.append(0.0)
            area_growths.append(1.0)
            main_axis_growths.append(0.0)

        prev_fill_ratio = current_fill_ratio
        prev_union_area = union_area
        line_bboxes.append(line.bbox)

    return _make_stats(
        union=union,
        sum_line_area=sum_line_area,
        page_area=page_area,
        image_size=image_size,
        median_width=median_width,
        median_height=median_height,
        x_gaps=x_gaps,
        y_overlaps=y_overlaps,
        append_y_overlaps=append_y_overlaps,
        fill_drops=fill_drops,
        area_growths=area_growths,
        main_axis_growths=main_axis_growths,
        line_bboxes=line_bboxes,
        mature_low_y_overlap_count=mature_low_y_overlap_count,
        append_y_overlap_signals=append_y_overlap_signals,
        config=config,
    )


def build_block_stats_for_lines(
    lines: list[TextLine],
    image_size: tuple[int, int],
    config: dict[str, Any],
) -> BlockStats:
    """Build aggregate stats for one continuous line slice."""

    median_width, median_height = median_line_size(lines)
    return _build_range_stats(lines, image_size, config, median_width, median_height)


def build_candidates(
    lines: list[TextLine],
    image_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[list[list[BlockCandidate]], list[list[BlockCandidate]], dict[str, Any]]:
    """Enumerate continuous interval candidates and store them by end index."""

    line_count = len(lines)
    page_area = float(image_size[0] * image_size[1])
    candidate_lists_by_end: list[list[BlockCandidate]] = [[] for _ in range(line_count + 1)]
    candidate_lists_by_start: list[list[BlockCandidate]] = [[] for _ in range(line_count + 1)]
    median_width, median_height = median_line_size(lines)
    max_lines = config["candidate_generation"]["max_lines_per_block"]
    max_candidates_per_start = config["candidate_generation"]["max_candidates_per_start"]
    candidate_id = 0

    debug_meta = {
        "median_line_width": median_width,
        "median_line_height": median_height,
        "candidate_count": 0,
    }

    for start_idx in range(line_count):
        union = lines[start_idx].bbox
        sum_line_area = 0.0
        x_gaps: list[float] = []
        y_overlaps: list[float] = []
        append_y_overlaps: list[float] = []
        fill_drops: list[float] = []
        area_growths: list[float] = []
        main_axis_growths: list[float] = []
        append_y_overlap_signals: list[float] = []
        line_bboxes: list[tuple[float, float, float, float]] = []
        mature_low_y_overlap_count = 0
        prev_fill_ratio = 1.0
        prev_union_area = 0.0

        for end_idx in range(start_idx, min(line_count, start_idx + max_lines)):
            line = lines[end_idx]
            prev_union = union
            prior_line_count = len(line_bboxes)
            prior_stability_score = y_band_stability_score(line_bboxes, median_height) if line_bboxes else 1.0
            union = line.bbox if end_idx == start_idx else union_bbox(union, line.bbox)
            sum_line_area += line.area
            union_area = bbox_area(union)
            current_fill_ratio = fill_ratio(sum_line_area, union)
            area_ratio = union_area / max(page_area, 1e-6)
            main_axis_growth_ratio = (
                max(0.0, bbox_height(union) - bbox_height(prev_union)) / max(median_height, 1e-6)
                if end_idx > start_idx
                else 0.0
            )

            if end_idx > start_idx:
                x_gap_norm, y_overlap_ratio = adjacency_features(lines[end_idx - 1].bbox, line.bbox, median_width)
                append_y_overlap_ratio = y_band_overlap_ratio(prev_union, line.bbox)
                append_signal, mature_hit = _append_y_overlap_signal(
                    append_y_overlap_ratio,
                    prior_line_count,
                    prior_stability_score,
                    config,
                )
                x_gaps.append(x_gap_norm)
                y_overlaps.append(y_overlap_ratio)
                append_y_overlaps.append(append_y_overlap_ratio)
                append_y_overlap_signals.append(append_signal)
                if mature_hit:
                    mature_low_y_overlap_count += 1

                if _should_break(
                    x_gap_norm=x_gap_norm,
                    union_area_ratio=area_ratio,
                    current_fill_ratio=current_fill_ratio,
                    main_axis_growth_ratio=main_axis_growth_ratio,
                    line_count=end_idx - start_idx + 1,
                    config=config,
                ):
                    break

            if prev_union_area > 0:
                fill_drops.append(max(0.0, prev_fill_ratio - current_fill_ratio))
                area_growths.append((union_area - prev_union_area) / max(line.area, 1e-6))
                main_axis_growths.append(main_axis_growth_ratio)
            else:
                fill_drops.append(0.0)
                area_growths.append(1.0)
                main_axis_growths.append(0.0)

            prev_fill_ratio = current_fill_ratio
            prev_union_area = union_area
            line_bboxes.append(line.bbox)

            stats = _make_stats(
                union=union,
                sum_line_area=sum_line_area,
                page_area=page_area,
                image_size=image_size,
                median_width=median_width,
                median_height=median_height,
                x_gaps=x_gaps,
                y_overlaps=y_overlaps,
                append_y_overlaps=append_y_overlaps,
                fill_drops=fill_drops,
                area_growths=area_growths,
                main_axis_growths=main_axis_growths,
                line_bboxes=line_bboxes,
                mature_low_y_overlap_count=mature_low_y_overlap_count,
                append_y_overlap_signals=append_y_overlap_signals,
                config=config,
            )

            cost, breakdown = compute_candidate_cost(stats, config)
            candidate = BlockCandidate(
                candidate_id=candidate_id,
                start_idx=start_idx,
                end_idx_exclusive=end_idx + 1,
                bbox=union,
                stats=stats,
                cost=cost,
                cost_breakdown=breakdown,
            )
            candidate_id += 1
            candidate_lists_by_end[end_idx + 1].append(candidate)
            candidate_lists_by_start[start_idx].append(candidate)
            debug_meta["candidate_count"] += 1

    if line_count > 0:
        fallback_stats = _build_range_stats(lines, image_size, config, median_width, median_height)
        fallback_cost, fallback_breakdown = compute_candidate_cost(fallback_stats, config)
        fallback_breakdown["page_fallback_penalty"] = (
            config["weights"]["page_fallback_penalty"] * block_cost_scale(config)
        )
        fallback_cost += fallback_breakdown["page_fallback_penalty"]
        fallback_candidate = BlockCandidate(
            candidate_id=candidate_id,
            start_idx=0,
            end_idx_exclusive=line_count,
            bbox=fallback_stats.bbox,
            stats=fallback_stats,
            cost=fallback_cost,
            cost_breakdown=fallback_breakdown,
        )
        candidate_lists_by_end[line_count].append(fallback_candidate)
        candidate_lists_by_start[0].append(fallback_candidate)
        candidate_id += 1
        debug_meta["candidate_count"] += 1

    pruned_candidate_lists_by_start: list[list[BlockCandidate]] = [[] for _ in range(line_count + 1)]
    pruned_candidate_lists_by_end: list[list[BlockCandidate]] = [[] for _ in range(line_count + 1)]
    selected_candidates = 0
    for start_idx, bucket in enumerate(candidate_lists_by_start):
        pruned_bucket = _prune_start_bucket(bucket, max_candidates_per_start)
        pruned_candidate_lists_by_start[start_idx] = pruned_bucket
        for candidate in pruned_bucket:
            pruned_candidate_lists_by_end[candidate.end_idx_exclusive].append(candidate)
            selected_candidates += 1

    debug_meta["candidate_count_after_prune"] = selected_candidates

    return pruned_candidate_lists_by_end, pruned_candidate_lists_by_start, debug_meta
