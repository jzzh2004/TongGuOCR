#!/usr/bin/env python3
"""Compatibility entry point for standalone RO-ED; prefer calculate_metrics.py.

Use evaluation/calculate_metrics.py to write every metric to one JSON file.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from .reading_order import calculate_reading_order

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict-jsonl', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--run-id')
    parser.add_argument('--dataset')
    parser.add_argument('--omnidocbench-root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predict_jsonl).read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    summary = calculate_reading_order(rows, args.workers, source_path=args.predict_jsonl, output_dir=args.work_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
