from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Built-in runtime defaults. An optional user config passed via `--config`
# (YAML or JSON) is deep-merged onto these, so a user file only needs the
# keys it overrides, e.g.
#
#     ocr:
#       backend: vllm
#       vllm:
#         api_base: http://10.0.0.2:8090/v1
#
# The `blocking` section holds the internal block-segmentation solver
# constants of the released pipeline; override only if you know what you
# are doing.
DEFAULTS: dict[str, Any] = {
    "layout_inference": {
        "device": "cuda:0",
        "detector_config_path": str(PACKAGE_ROOT / "layout" / "hdc_speed.py"),
        "detector_checkpoint_path": "./weights/layout/finetune_infer.pth",
        "detector_score_threshold": 0.5,
        "batch_size": 8,
    },
    "blocking": {
        "lambda_weights": {
            "lambda_blk": 2.0,
            "lambda_adj": 2.0,
            "lambda_page": 1.0,
        },
        "candidate_generation": {
            "max_lines_per_block": 62,
            "max_candidates_per_start": 42,
        },
        "dp": {
            "search_max_blocks": 30,
            "soft_max_blocks_per_page": 19,
            "allow_over_cap_fallback": True,
        },
        "postprocess": {
            "merge_tiny_adjacent_blocks": {
                "enabled": True,
                "small_area_multiplier": 2.5,
                "required_improvement_small_neighbor": -2.0,
                "required_improvement_default": 1.0,
                "edge_cluster_merge_enabled": True,
                "edge_cluster_max_blocks": 3,
                "edge_cluster_max_lines": 4,
            },
        },
        # Fixed block-level thresholds used by the internal omega terms.
        "thresholds": {
            "target_block_count_range": [5, 13],
            "min_lines_hard": 2,
            "preferred_min_lines": 3,
            "mature_block_line_count": 5,
            "min_fill_ratio_hard": 0.1,
            "min_fill_ratio_soft": 0.3,
            "aspect_ratio_hard": 9.0,
            "aspect_ratio_soft": 3.7,
            "block_area_ratio_hard_max": 0.7,
            "block_area_ratio_soft_max": 0.45,
            "block_area_ratio_soft_min": 0.008,
            "tiny_block_area_ratio_soft": 0.006,
            "x_gap_norm_hard": 3.0,
            "x_gap_norm_soft": 0.85,
            "y_overlap_ratio_hard": 0.0,
            "y_overlap_ratio_soft": 0.05,
            "mature_y_overlap_soft": 0.37,
            "y_band_stability_soft": 0.72,
            "fill_drop_soft": 0.12,
            "union_area_growth_ratio_soft": 4.5,
            "main_axis_growth_ratio_soft": 2.0,
            "main_axis_growth_ratio_hard": 3.5,
            "vertical_strip_ratio_soft": 4.5,
            "vertical_strip_ratio_hard": 8.4,
            "width_vs_median_line_soft": 2.4,
            "width_vs_median_line_hard": 1.8,
            "cross_axis_coverage_ratio_soft_min": 0.09,
            "cross_axis_coverage_ratio_hard_min": 0.05,
            "main_axis_span_ratio_hard": 0.45,
            "internal_x_gap_spike_soft": 1.02,
            "internal_x_gap_spike_hard": 1.8,
            "transition_overlap_iou_soft": 0.06,
            "transition_overlap_iou_hard": 0.18,
            "transition_containment_soft": 0.35,
            "transition_containment_hard": 0.7,
        },
        # Fixed block, adjacency and page penalty weights (omega, gamma, rho).
        "weights": {
            "open_block_cost": 0.4,
            "line_count_penalty": 0.75,
            "tiny_block_penalty": 1.35,
            "hard_constraint_penalty": 70.0,
            "shape_penalty": 3.8,
            "area_penalty": 1.1,
            "fill_penalty": 2.8,
            "adjacency_penalty": 4.0,
            "strip_penalty": 7.2,
            "mature_y_overlap_penalty": 30.0,
            "growth_penalty": 2.4,
            "page_block_count_penalty": 3.0,
            "single_block_page_penalty": 40.0,
            "transition_overlap_penalty": 10.0,
            "transition_containment_penalty": 12.0,
            "tiny_inside_large_penalty": 20.0,
            "transition_hard_penalty": 80.0,
            "bad_split_penalty": 32.0,
            "page_fallback_penalty": 120.0,
        },
    },
    "ocr": {
        "backend": "transformers",
        # NOTE: the text prompt below is combined with a structured image item
        # by every backend before chat-template rendering, so the effective
        # user turn is `<image tokens> + "Text Recognition:"` — exactly what
        # the training annotations express as "<image>Text Recognition:" (the
        # tokenizer has no literal <image> token; the trainer replaced the
        # placeholder with real image tokens). Do NOT inline "<image>" into
        # the string: it would be tokenized as plain text garbage.
        "prompt": "Text Recognition:",
        # Single source of truth shared by ALL OCR backends so that
        # transformers and vLLM always decode with identical settings: same
        # max_tokens, same pixel budget, same (greedy) sampling. Both
        # backends load the model in bfloat16 (hardcoded).
        "inference": {
            "max_tokens": 8192,
            "image_max_pixels": 1048576,
            "image_min_pixels": 1024,
            "temperature": 0.0,
        },
        "transformers": {
            "model_path": "./weights/ocr",
            "device": "cuda:0",
        },
        # Connection settings only for the vLLM client; decoding parameters
        # come from ocr.inference above.
        "vllm": {
            "api_base": "http://127.0.0.1:8090/v1",
            "model_name": "tonggu-ocr",
            "timeout": 600,
            # Client-side pre-resize (tongguocr/ocr/preprocess.py) so the
            # server receives the same pixel matrix regardless of its
            # processor defaults.
            "pre_resize": True,
        },
    },
    # Fixed Block Crop Refinement settings: margin p, masking and canvas alpha.
    "output": {
        "visualize": True,
        "crop_blocks": True,
        "crop_padding": 8,
        "crop_mask": {
            "enabled": True,
        },
        "crop_canvas": {
            "mode": "padded_square",
            "base_scale": 1.1,
            "sparse_scale": 1.1,
            "sparse_line_threshold": 2,
            "sparse_text_bbox_area_ratio": 0.55,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_user_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unsupported top-level config section(s) {unknown}; "
            f"expected one of {sorted(DEFAULTS)}"
        )
    return config


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return the runtime configuration: DEFAULTS deep-merged with the
    optional user YAML/JSON override."""
    if config_path is None:
        return deepcopy(DEFAULTS)
    return _deep_merge(DEFAULTS, _read_user_config(config_path))
