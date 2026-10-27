#!/usr/bin/env python3
"""Reading-order edit distance (RO-ED) evaluation for TongGuOCR.

Adapts pipeline predictions into the OmniDocBench reading-order evaluation:
each page's GT text and prediction are split into text-line units by the
pinned protocol in ``protocol.py``, converted to OmniDocBench end2end
records, and scored with the official ``Edit_dist`` implementation. The
reported number is ``reading_order / all / Edit_dist / ALL_page_avg``
(lower is better).

Input is the single jsonl produced by ``evaluation/make_eval_jsonl.py``
(rows ``{page_id, predict, label}``). The pipeline converts the model's
relation tokens into newlines (see README.md), and the protocol
splits predictions on newlines — so reading order comes straight from the
line structure of ``predict``.

Usage (repo root):

    python -m evaluation.ro_ed.ro_ed \
        --predict-jsonl runs/eval/m5/predict.jsonl \
        --work-dir      runs/eval/m5/ro_ed \
        --omnidocbench-root evaluation/third_party/OmniDocBench
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

try:
    from .protocol import PreprocessStats, preprocess_gt, preprocess_prediction, protocol_hash, protocol_spec
except ImportError:  # executed as a plain script from evaluation/ro_ed/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protocol import PreprocessStats, preprocess_gt, preprocess_prediction, protocol_hash, protocol_spec  # type: ignore


# The paper's RO-ED numbers were produced under exactly this protocol
# (plain_text_lines_v1 rev 2). The assert keeps this module and protocol.py
# in sync with the archived runs.
EXPECTED_PROTOCOL_HASH = "ff195d9bb8574edf69dac04decfd7facd8cd5c2a19e2bd871bf0285ab0ff4623"

SAVE_NAME = "predictions_quick_match"

PRED_FIELD = "predict"
GT_FIELD = "label"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            rows.append(value)
    return rows


def normalize_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [
        {
            "page_id": str(row.get("page_id", "")),
            GT_FIELD: str(row.get(GT_FIELD, "") or ""),
            PRED_FIELD: str(row.get(PRED_FIELD, "") or ""),
        }
        for row in rows
    ]
    keys = [record["page_id"] for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate page ids in --predict-jsonl")
    return records


def merge_stats(total: Counter[str], stats: PreprocessStats, prefix: str) -> None:
    data = stats.to_dict()
    unknown = data.pop("unknown_angle_tokens")
    data.pop("unknown_angle_token_count")
    for key, value in data.items():
        total[f"{prefix}_{key}"] += int(value)
    for token, count in unknown.items():
        total[f"{prefix}_unknown::{token}"] += int(count)


def build_inputs(records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    gt_pages: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    line_diffs: list[int] = []
    normalized_path = run_dir / "normalized_predictions.jsonl"
    seen: set[str] = set()
    with normalized_path.open("w", encoding="utf-8") as normalized_handle:
        for index, record in enumerate(records, start=1):
            key = str(record["page_id"])
            if key in seen:
                raise ValueError(f"Duplicate page id during build: {key}")
            seen.add(key)
            gt_units, gt_stats = preprocess_gt(record.get(GT_FIELD, ""))
            pred_units, pred_stats = preprocess_prediction(record.get(PRED_FIELD, ""))
            merge_stats(totals, gt_stats, "gt")
            merge_stats(totals, pred_stats, "pred")
            totals["empty_gt_pages"] += not gt_units
            totals["empty_pred_pages"] += not pred_units
            line_diffs.append(len(pred_units) - len(gt_units))

            layout_dets = [
                {
                    "anno_id": order,
                    "category_type": "text_block",
                    "order": order,
                    "text": unit,
                    "attribute": {},
                    "ignore": False,
                }
                for order, unit in enumerate(gt_units, start=1)
            ]
            gt_pages.append(
                {
                    "page_info": {
                        "page_attribute": {},
                        "page_no": index,
                        "height": 0,
                        "width": 0,
                        "image_path": f"{key}.jpg",
                    },
                    "layout_dets": layout_dets,
                    "extra": {"relation": []},
                }
            )
            (predictions_dir / f"{key}.md").write_text("\n".join(pred_units), encoding="utf-8")
            normalized_handle.write(
                json.dumps({"page_id": key, "units": pred_units}, ensure_ascii=False) + "\n"
            )

    unknown_gt = {
        key.split("::", 1)[1]: value
        for key, value in totals.items()
        if key.startswith("gt_unknown::")
    }
    unknown_pred = {
        key.split("::", 1)[1]: value
        for key, value in totals.items()
        if key.startswith("pred_unknown::")
    }
    scalar_totals = {key: value for key, value in totals.items() if "unknown::" not in key}
    stats = {
        "record_count": len(records),
        **scalar_totals,
        "gt_unknown_angle_tokens": dict(sorted(unknown_gt.items())),
        "pred_unknown_angle_tokens": dict(sorted(unknown_pred.items())),
        "gt_unknown_angle_token_count": sum(unknown_gt.values()),
        "pred_unknown_angle_token_count": sum(unknown_pred.values()),
        "line_count_diff_min": min(line_diffs, default=0),
        "line_count_diff_max": max(line_diffs, default=0),
        "line_count_diff_sum": sum(line_diffs),
    }
    gt_path = run_dir / "generated_gt.json"
    stats_path = run_dir / "preprocess_stats.json"
    write_json(gt_path, gt_pages)
    write_json(stats_path, stats)
    return {"gt_path": gt_path, "predictions_dir": predictions_dir, "stats": stats}


def write_config(run_dir: Path, generated: dict[str, Any], match_workers: int) -> Path:
    config = {
        "end2end_eval": {
            "metrics": {"reading_order": {"metric": ["Edit_dist"]}},
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(generated["gt_path"])},
                "prediction": {"data_path": str(generated["predictions_dir"])},
                "match_method": "quick_match",
                "match_workers": match_workers,
                "match_timeout_sec": 420,
                "quick_match_truncated_timeout_sec": 300,
            },
        }
    }
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def plain_text_filter(content: str) -> defaultdict[str, list[dict[str, Any]]]:
    result: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    offset = 0
    for line in str(content or "").splitlines():
        if line:
            result["text_all"].append(
                {
                    "category_type": "text_all",
                    "position": [offset, offset + len(line)],
                    "content": line,
                    "fine_category_type": "text_block",
                }
            )
        offset += len(line) + 1
    return result


def run_official(run_dir: Path, config_path: Path, omnidocbench_root: Path) -> None:
    sys.path.insert(0, str(omnidocbench_root))
    import src.dataset.end2end_dataset as end2end_module
    from src.core.pipeline import run_config_file

    end2end_module.md_tex_filter = plain_text_filter
    previous = Path.cwd()
    try:
        os.chdir(run_dir)
        run_config_file(str(config_path))
    finally:
        os.chdir(previous)


def metric_summary(
    run_dir: Path,
    meta: dict[str, Any],
    stats: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    result_dir = run_dir / "result"
    metric_path = result_dir / f"{SAVE_NAME}_metric_result.json"
    per_page_path = result_dir / f"{SAVE_NAME}_reading_order_per_page_edit.json"
    reading_path = result_dir / f"{SAVE_NAME}_reading_order_result.json"
    metric = read_json(metric_path)
    per_page = read_json(per_page_path)
    reading = read_json(reading_path)
    values = metric["reading_order"]["all"]["Edit_dist"]
    debug = metric.get("match_debug", {})
    fallback = debug.get("text_match_fallback_counts", {})
    return {
        "run_id": meta["run_id"],
        "dataset": meta.get("dataset"),
        "source_kind": "predict_jsonl",
        "source_path": meta["source_path"],
        "status": "ok",
        "record_count": stats["record_count"],
        "limit": limit,
        "official_page_count": len(per_page),
        "reading_order_sample_count": len(reading),
        "empty_gt_pages": stats["empty_gt_pages"],
        "empty_pred_pages": stats["empty_pred_pages"],
        "gt_unit_count": stats["gt_unit_count"],
        "pred_unit_count": stats["pred_unit_count"],
        "gt_removed_spaces": stats["gt_removed_spaces"],
        "pred_removed_spaces": stats["pred_removed_spaces"],
        "pred_relation_boundaries": stats["pred_relation_boundaries"],
        "pred_block_tag_boundaries": stats["pred_block_tag_boundaries"],
        "pred_inline_tags_removed": stats["pred_inline_tags_removed"],
        "pred_think_boundaries": stats["pred_think_boundaries"],
        "pred_ref_blocks_removed": stats["pred_ref_blocks_removed"],
        "pred_det_blocks_removed": stats["pred_det_blocks_removed"],
        "pred_unknown_angle_token_count": stats["pred_unknown_angle_token_count"],
        "ALL_page_avg": values.get("ALL_page_avg"),
        "edit_whole": values.get("edit_whole"),
        "edit_sample_avg": values.get("edit_sample_avg"),
        "quick_match_timeout_count": int(fallback.get("quick_match_timeout", 0) or 0),
        "page_timeout_count": int(fallback.get("page_timeout", 0) or 0),
        "fallback_total": sum(int(value or 0) for value in fallback.values()),
        "run_dir": str(run_dir),
        "metric_path": str(metric_path),
        "protocol_hash": protocol_hash(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-jsonl", required=True,
                        help="predict.jsonl from make_eval_jsonl.py")
    parser.add_argument("--work-dir", required=True,
                        help="output directory for all run artifacts")
    parser.add_argument("--omnidocbench-root", required=True,
                        help="path to the vendored OmniDocBench tree")
    parser.add_argument("--workers", type=int, default=12,
                        help="match_workers for the OmniDocBench scorer")
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N pages (smoke runs)")
    parser.add_argument("--run-id", default=None,
                        help="label recorded in summary.json (default: work-dir name)")
    parser.add_argument("--dataset", default=None, help="optional dataset label")
    args = parser.parse_args()

    if protocol_hash() != EXPECTED_PROTOCOL_HASH:
        raise SystemExit(
            "protocol.py spec hash changed; this module is pinned to the "
            "protocol of the paper's archived RO-ED runs"
        )

    run_dir = Path(args.work_dir).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "protocol.json", {"spec": protocol_spec(), "hash": protocol_hash()})

    rows = read_jsonl(Path(args.predict_jsonl))
    if args.limit is not None:
        rows = rows[: args.limit]
    records = normalize_records(rows)

    generated = build_inputs(records, run_dir)
    config_path = write_config(run_dir, generated, args.workers)
    run_official(run_dir, config_path, Path(args.omnidocbench_root).resolve())

    meta = {
        "run_id": args.run_id or run_dir.name,
        "dataset": args.dataset,
        "source_path": str(Path(args.predict_jsonl).resolve()),
    }
    summary = metric_summary(run_dir, meta, generated["stats"], args.limit)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
