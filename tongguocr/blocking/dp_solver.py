from __future__ import annotations

import logging
import math
from typing import Any

from .candidate_builder import build_block_stats_for_lines, build_candidates
from .costs import (
    compute_candidate_cost,
    compute_transition_cost,
    lambda_scales,
    lambda_weights,
    page_block_count_penalty,
)
from .models import BlockCandidate, BlockResult, PageResult, TextLine


LOGGER = logging.getLogger(__name__)


def _backtrack(
    best_candidate_id: int,
    block_count: int,
    candidates_by_id: dict[int, BlockCandidate],
    backpointer: dict[tuple[int, int], tuple[int | None, dict[str, float] | None]],
) -> tuple[list[BlockCandidate], list[dict[str, float]]]:
    picked: list[BlockCandidate] = []
    transitions: list[dict[str, float]] = []
    current_candidate_id: int | None = best_candidate_id
    current_k = block_count
    while current_candidate_id is not None and current_k > 0:
        candidate = candidates_by_id[current_candidate_id]
        if candidate is None:
            raise RuntimeError("DP backtrack failed to find a valid candidate")
        picked.append(candidate)
        previous_candidate_id, transition_breakdown = backpointer[(current_candidate_id, current_k)]
        if transition_breakdown is not None:
            transitions.append(transition_breakdown)
        current_candidate_id = previous_candidate_id
        current_k -= 1
    picked.reverse()
    transitions.reverse()
    return picked, transitions


def _solve_dp(
    candidate_lists_by_end: list[list[BlockCandidate]],
    candidate_lists_by_start: list[list[BlockCandidate]],
    line_count: int,
    config: dict[str, Any],
    max_blocks: int,
) -> tuple[float, int, int | None, dict[int, BlockCandidate], dict[tuple[int, int], tuple[int | None, dict[str, float] | None]]]:
    all_candidates = [candidate for bucket in candidate_lists_by_end for candidate in bucket]
    candidates_by_id = {candidate.candidate_id: candidate for candidate in all_candidates}
    dp_cost: dict[tuple[int, int], float] = {}
    backpointer: dict[tuple[int, int], tuple[int | None, dict[str, float] | None]] = {}

    for candidate in all_candidates:
        if candidate.start_idx == 0:
            dp_cost[(candidate.candidate_id, 1)] = candidate.cost
            backpointer[(candidate.candidate_id, 1)] = (None, None)

    for block_count in range(2, max_blocks + 1):
        for start_idx in range(line_count + 1):
            current_bucket = candidate_lists_by_start[start_idx]
            if not current_bucket:
                continue
            previous_bucket = candidate_lists_by_end[start_idx]
            if not previous_bucket:
                continue
            for candidate in current_bucket:
                best_cost = math.inf
                best_prev_id: int | None = None
                best_transition: dict[str, float] | None = None
                for prev_candidate in previous_bucket:
                    previous_cost = dp_cost.get((prev_candidate.candidate_id, block_count - 1), math.inf)
                    if math.isinf(previous_cost):
                        continue
                    transition_cost, transition_breakdown = compute_transition_cost(
                        prev_candidate,
                        candidate,
                        config,
                    )
                    proposed = previous_cost + candidate.cost + transition_cost
                    if proposed < best_cost:
                        best_cost = proposed
                        best_prev_id = prev_candidate.candidate_id
                        best_transition = transition_breakdown
                if best_prev_id is not None:
                    dp_cost[(candidate.candidate_id, block_count)] = best_cost
                    backpointer[(candidate.candidate_id, block_count)] = (best_prev_id, best_transition)

    best_cost = math.inf
    best_block_count = 0
    best_candidate_id: int | None = None
    for block_count in range(1, max_blocks + 1):
        for candidate in candidate_lists_by_end[line_count]:
            total = dp_cost.get((candidate.candidate_id, block_count), math.inf)
            if math.isinf(total):
                continue
            total += page_block_count_penalty(block_count, config)
            if total < best_cost:
                best_cost = total
                best_block_count = block_count
                best_candidate_id = candidate.candidate_id

    return best_cost, best_block_count, best_candidate_id, candidates_by_id, backpointer


def _make_candidate_for_range(
    lines: list[TextLine],
    image_size: tuple[int, int],
    config: dict[str, Any],
    start_idx: int,
    end_idx_exclusive: int,
) -> BlockCandidate:
    stats = build_block_stats_for_lines(lines[start_idx:end_idx_exclusive], image_size, config)
    cost, breakdown = compute_candidate_cost(stats, config)
    return BlockCandidate(
        candidate_id=-(start_idx * 10000 + end_idx_exclusive),
        start_idx=start_idx,
        end_idx_exclusive=end_idx_exclusive,
        bbox=stats.bbox,
        stats=stats,
        cost=cost,
        cost_breakdown=breakdown,
    )


def _local_score(
    candidates: list[BlockCandidate],
    config: dict[str, Any],
    start_idx: int,
    end_idx: int,
) -> float:
    score = 0.0
    for idx in range(start_idx, end_idx + 1):
        score += candidates[idx].cost
    left_neighbor = start_idx - 1
    right_neighbor = end_idx + 1
    if left_neighbor >= 0:
        score += compute_transition_cost(candidates[left_neighbor], candidates[start_idx], config)[0]
    for idx in range(start_idx, end_idx):
        score += compute_transition_cost(candidates[idx], candidates[idx + 1], config)[0]
    if right_neighbor < len(candidates):
        score += compute_transition_cost(candidates[end_idx], candidates[right_neighbor], config)[0]
    return score


def _merge_tiny_adjacent_blocks(picked, lines, image_size, config):
    candidates = list(picked)
    thresholds = config["thresholds"]
    postprocess = config.get("postprocess", {}).get("merge_tiny_adjacent_blocks", {})
    if not postprocess.get("enabled", True):
        return candidates, 0

    merge_count = 0
    changed = True
    edge_cluster_merge_used = False

    # Treat truly tiny blocks and unusually small-area edge fragments as mergeable.
    small_area_cutoff = thresholds["tiny_block_area_ratio_soft"] * float(
        postprocess.get("small_area_multiplier", 2.5)
    )
    required_small_neighbor = float(postprocess.get("required_improvement_small_neighbor", -2.0))
    required_default = float(postprocess.get("required_improvement_default", 1.0))
    edge_cluster_enabled = bool(postprocess.get("edge_cluster_merge_enabled", True))
    edge_cluster_max_blocks = int(postprocess.get("edge_cluster_max_blocks", 3))
    edge_cluster_max_lines = int(postprocess.get("edge_cluster_max_lines", 4))

    def is_small(candidate: BlockCandidate) -> bool:
        return (
            candidate.stats.line_count <= thresholds["min_lines_hard"]
            or candidate.stats.area_ratio <= small_area_cutoff
        )

    while changed and len(candidates) > 1:
        changed = False
        for idx, candidate in enumerate(candidates):
            if not is_small(candidate):
                continue
            best_option: tuple[float, int, BlockCandidate] | None = None
            for neighbor_idx in (idx - 1, idx + 1):
                if neighbor_idx < 0 or neighbor_idx >= len(candidates):
                    continue
                left_idx = min(idx, neighbor_idx)
                right_idx = max(idx, neighbor_idx)
                merged_candidate = _make_candidate_for_range(
                    lines,
                    image_size,
                    config,
                    candidates[left_idx].start_idx,
                    candidates[right_idx].end_idx_exclusive,
                )
                if merged_candidate.stats.fill_ratio < thresholds["min_fill_ratio_soft"] * 0.75:
                    continue
                if merged_candidate.stats.vertical_strip_ratio > thresholds["vertical_strip_ratio_soft"] * 1.5:
                    continue
                if merged_candidate.stats.aspect_ratio > thresholds["aspect_ratio_soft"] * 1.6:
                    continue
                before = _local_score(candidates, config, left_idx, right_idx)
                after_candidates = candidates[:left_idx] + [merged_candidate] + candidates[right_idx + 1 :]
                after = _local_score(after_candidates, config, left_idx, left_idx)
                improvement = before - after
                required_improvement = required_small_neighbor if is_small(candidates[neighbor_idx]) else required_default
                if improvement > required_improvement and (best_option is None or improvement > best_option[0]):
                    best_option = (improvement, left_idx, merged_candidate)
            if best_option is not None:
                _, left_idx, merged_candidate = best_option
                right_idx = left_idx + 1
                candidates = candidates[:left_idx] + [merged_candidate] + candidates[right_idx + 1 :]
                merge_count += 1
                changed = True
                break
        if changed:
            continue
        run_start = 0
        while run_start < len(candidates):
            if not is_small(candidates[run_start]):
                run_start += 1
                continue
            run_end = run_start
            while run_end + 1 < len(candidates) and is_small(candidates[run_end + 1]):
                run_end += 1
            run_block_count = run_end - run_start + 1
            run_line_count = sum(item.stats.line_count for item in candidates[run_start : run_end + 1])
            if edge_cluster_merge_used or not edge_cluster_enabled or not (
                (run_start == 0 or run_end == len(candidates) - 1 or run_line_count <= edge_cluster_max_lines)
                and run_block_count <= edge_cluster_max_blocks
            ):
                run_start = run_end + 1
                continue
            options: list[tuple[float, int, int, BlockCandidate]] = []
            for neighbor_side in ("left", "right"):
                if neighbor_side == "left" and run_start > 0:
                    merge_start = run_start - 1
                    merge_end = run_end
                elif neighbor_side == "right" and run_end + 1 < len(candidates):
                    merge_start = run_start
                    merge_end = run_end + 1
                else:
                    continue
                merged_candidate = _make_candidate_for_range(
                    lines,
                    image_size,
                    config,
                    candidates[merge_start].start_idx,
                    candidates[merge_end].end_idx_exclusive,
                )
                if merged_candidate.stats.fill_ratio < thresholds["min_fill_ratio_soft"] * 0.8:
                    continue
                if merged_candidate.stats.vertical_strip_ratio > thresholds["vertical_strip_ratio_soft"] * 1.25:
                    continue
                if merged_candidate.stats.aspect_ratio > thresholds["aspect_ratio_soft"] * 1.35:
                    continue
                before = _local_score(candidates, config, merge_start, merge_end)
                after_candidates = candidates[:merge_start] + [merged_candidate] + candidates[merge_end + 1 :]
                after = _local_score(after_candidates, config, merge_start, merge_start)
                options.append((before - after, merge_start, merge_end, merged_candidate))
            if options:
                _, merge_start, merge_end, merged_candidate = max(options, key=lambda item: item[0])
                candidates = candidates[:merge_start] + [merged_candidate] + candidates[merge_end + 1 :]
                merge_count += merge_end - merge_start
                edge_cluster_merge_used = True
                changed = True
                break
            run_start = run_end + 1
    return candidates, merge_count


def segment_page(
    lines: list[TextLine],
    image_size: tuple[int, int],
    config: dict[str, Any],
    *,
    page_id: str = "",
    image_path: str = "",
    txt_path: str = "",
    include_debug: bool = False,
) -> PageResult:
    """Split ordered text lines into globally optimal continuous blocks."""

    if not lines:
        raise ValueError("segment_page requires at least one line")

    candidate_lists_by_end, candidate_lists_by_start, candidate_debug = build_candidates(lines, image_size, config)
    n = len(lines)
    configured_cap = config["dp"]["search_max_blocks"]
    max_blocks = min(n, configured_cap)
    best_cost, best_block_count, best_candidate_id, candidates_by_id, backpointer = _solve_dp(
        candidate_lists_by_end,
        candidate_lists_by_start,
        n,
        config,
        max_blocks,
    )

    if best_block_count == 0 and config["dp"].get("allow_over_cap_fallback", False) and max_blocks < n:
        fallback_max_blocks = n
        LOGGER.warning(
            "No valid segmentation for %s under max_blocks=%s; retry with max_blocks=%s",
            page_id or "<unknown>",
            max_blocks,
            fallback_max_blocks,
        )
        max_blocks = fallback_max_blocks
        best_cost, best_block_count, best_candidate_id, candidates_by_id, backpointer = _solve_dp(
            candidate_lists_by_end,
            candidate_lists_by_start,
            n,
            config,
            max_blocks,
        )

    if best_block_count == 0 or best_candidate_id is None:
        raise RuntimeError(f"Failed to find a valid segmentation for page {page_id or '<unknown>'}")

    picked, transitions = _backtrack(best_candidate_id, best_block_count, candidates_by_id, backpointer)
    picked, post_merge_count = _merge_tiny_adjacent_blocks(picked, lines, image_size, config)
    transitions = [
        compute_transition_cost(picked[idx], picked[idx + 1], config)[1]
        for idx in range(len(picked) - 1)
    ]
    best_block_count = len(picked)
    best_cost = sum(candidate.cost for candidate in picked) + sum(
        compute_transition_cost(picked[idx], picked[idx + 1], config)[0]
        for idx in range(len(picked) - 1)
    )
    best_cost += page_block_count_penalty(best_block_count, config)
    block_results: list[BlockResult] = []
    for block_id, candidate in enumerate(picked):
        stats_dict = {
            "line_count": candidate.stats.line_count,
            "fill_ratio": candidate.stats.fill_ratio,
            "area_ratio": candidate.stats.area_ratio,
            "aspect_ratio": candidate.stats.aspect_ratio,
            "vertical_strip_ratio": candidate.stats.vertical_strip_ratio,
            "y_band_stability_score": candidate.stats.y_band_stability_score,
            "mature_low_y_overlap_count": candidate.stats.mature_low_y_overlap_count,
            "max_mature_low_y_overlap_penalty_signal": candidate.stats.max_mature_low_y_overlap_penalty_signal,
        }
        if include_debug:
            stats_dict.update(candidate.stats.to_dict())
        block_results.append(
            BlockResult(
                block_id=block_id,
                label="text_block",
                line_range=candidate.line_range_inclusive(),
                line_ids=[line.line_id for line in lines[candidate.start_idx : candidate.end_idx_exclusive]],
                bbox=candidate.bbox,
                stats=stats_dict,
                cost=candidate.cost,
                cost_breakdown=candidate.cost_breakdown if include_debug else None,
            )
        )

    transition_overlaps = [item.get("overlap_iou", 0.0) for item in transitions]
    transition_containments = [item.get("containment_ratio", 0.0) for item in transitions]
    transition_bad_splits = [item.get("bad_split_penalty", 0.0) for item in transitions]
    debug: dict[str, Any] = {}
    if include_debug:
        debug = {
            "lambda_weights": lambda_weights(config),
            "lambda_scales": lambda_scales(config),
            "candidate_generation": candidate_debug,
            "page_diagnostics": {
                "tiny_block_count": sum(
                    1 for candidate in picked if candidate.stats.line_count <= config["thresholds"]["min_lines_hard"]
                ),
                "max_adjacent_block_overlap": max(transition_overlaps) if transition_overlaps else 0.0,
                "max_adjacent_containment": max(transition_containments) if transition_containments else 0.0,
                "max_vertical_strip_ratio": max(candidate.stats.vertical_strip_ratio for candidate in picked),
                "single_block_penalty_applied": best_block_count == 1,
                "max_bad_split_penalty": max(transition_bad_splits) if transition_bad_splits else 0.0,
                "post_merge_count": post_merge_count,
                "hard_cap_hit": best_block_count > config["dp"]["soft_max_blocks_per_page"],
                "transitions": transitions,
            },
            "dp": {
                "max_blocks_considered": max_blocks,
                "configured_max_blocks_cap": configured_cap,
                "best_block_count": best_block_count,
                "page_block_count_penalty": page_block_count_penalty(best_block_count, config),
            },
        }

    return PageResult(
        page_id=page_id,
        image_path=image_path,
        txt_path=txt_path,
        image_size=image_size,
        block_count=len(block_results),
        total_cost=best_cost,
        blocks=block_results,
        debug=debug,
    )


def page_result_from_dict(payload: dict[str, Any]) -> PageResult:
    return PageResult(
        page_id=str(payload["page_id"]),
        image_path=str(payload.get("image_path") or ""),
        txt_path=str(payload.get("txt_path") or ""),
        image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
        block_count=int(payload["block_count"]),
        total_cost=float(payload.get("total_cost", 0.0)),
        blocks=[
            BlockResult(
                block_id=int(block["block_id"]),
                label=str(block["label"]),
                line_range=[int(block["line_range"][0]), int(block["line_range"][1])],
                line_ids=[int(line_id) for line_id in block["line_ids"]],
                bbox=tuple(float(value) for value in block["bbox"]),
                stats=dict(block.get("stats", {})),
                cost=float(block.get("cost", 0.0)),
                cost_breakdown=dict(block["cost_breakdown"]) if block.get("cost_breakdown") else None,
            )
            for block in payload["blocks"]
        ],
        debug=dict(payload.get("debug", {})),
    )


def segment_page_from_json(
    layout_json: dict[str, Any],
    image_size: tuple[int, int],
    config: dict[str, Any],
    *,
    page_id: str,
    image_path: str,
    debug: bool = False,
) -> dict[str, Any]:
    lines = [
        TextLine(
            line_id=int(line["line_id"]),
            text=str(line.get("text") or ""),
            bbox=tuple(float(value) for value in line["bbox"]),
        )
        for line in layout_json["lines"]
    ]
    page_result = segment_page(
        lines,
        image_size,
        config.get("blocking", config),
        page_id=page_id,
        image_path=image_path,
        txt_path="",
        include_debug=debug,
    )
    return page_result.to_dict(include_debug=debug)
