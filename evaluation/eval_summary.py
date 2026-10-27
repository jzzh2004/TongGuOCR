#!/usr/bin/env python3
"""Collect res.txt metric lines (+ optional RO-ED summary.json) into one report.

calculate_metrics.py writes full-width-colon lines such as

    任务ID：1, AR：93.74%, CR：94.45%
    任务ID：1, Edit Distance：6.17%

The paper's NED column is exactly the "Edit Distance" line (mean of
nltk.edit_distance / max(len) over pages), which is an independent metric
and not 1 - CR. The optional RO-ED summary.json is produced by
evaluation/ro_ed (reading-order protocol, see ro_ed/README.md); its
ALL_page_avg value is reported as-is with full precision.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# metric label in res.txt -> summary key
METRIC_LABELS = {
    "AR": "AR",
    "CR": "CR",
    "Edit Distance": "NED",
    "F1-Score": "F1",
    "Precision": "Precision",
    "Recall": "Recall",
    "BLEU": "BLEU",
}
PAIR_RE = re.compile("(" + "|".join(METRIC_LABELS) + r")：([0-9.]+)%")

# display order in the printed table
DISPLAY = ["AR", "CR", "NED", "F1", "Precision", "Recall", "BLEU", "RO-ED"]


def parse_res_txt(res_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in res_path.read_text(encoding="utf-8").splitlines():
        for label, value in PAIR_RE.findall(line):
            metrics[METRIC_LABELS[label]] = float(value)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res-txt", required=True, help="calculate_metrics.py report (res.txt)")
    parser.add_argument("--ro-ed-summary", default=None,
                        help="optional evaluation/ro_ed summary.json (adds RO-ED)")
    parser.add_argument("--output", default=None, help="optional summary.json output path")
    args = parser.parse_args()

    summary = parse_res_txt(Path(args.res_txt))
    if args.ro_ed_summary:
        ro_ed = json.loads(Path(args.ro_ed_summary).read_text(encoding="utf-8"))
        if "ALL_page_avg" not in ro_ed:
            raise SystemExit(f"missing ALL_page_avg in {args.ro_ed_summary}")
        summary["RO-ED"] = ro_ed["ALL_page_avg"]

    print("| " + " | ".join(DISPLAY) + " |")
    print("|" + "|".join(["---"] * len(DISPLAY)) + "|")
    row = []
    for key in DISPLAY:
        if key not in summary:
            row.append("-")
        elif key == "RO-ED":
            row.append(f"{summary[key]:.4f}")
        else:
            row.append(f"{summary[key]:.2f}")
    print("| " + " | ".join(row) + " |")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
