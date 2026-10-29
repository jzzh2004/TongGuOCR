#!/usr/bin/env python3
"""Adapt pipeline texts/*.txt + dataset GT into {predict,label} jsonl for eval.

GT conventions (verified against the on-disk datasets):
  M5:    <m5_root>/txt_line/<page_id>.txt, each line `x1,y1,...,x4,y4,text`
         -> parse with rsplit(',', 1); page label = line texts joined by '\n'.
  MTHv2: <mthv2_root>/<subset>/label_textline/<name>.txt, each line
         `text,x1,y1,...,x4,y4` -> parse with rsplit(',', 8); page label =
         line texts joined by '\n'. Staging page ids look like
         `MTH1000__06-V005P0405` (subset__originalname).

All metrics use texts/: the 24 relation tokens become newline boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .ro_ed.protocol import RELATION_RE
else:
    from ro_ed.protocol import RELATION_RE


def clean_predict(text: str, keep_relation_tokens: bool = False) -> str:
    if keep_relation_tokens:
        return text
    return RELATION_RE.sub("\n", text)


def strip_subset_prefix(page_id: str) -> str:
    """Normalize a staging page id to its dataset-local name.

    Staging ids for MTHv2 look like `MTH1000__06-V005P0405` (subset__name);
    M5 ids carry no prefix. This is the single shared rule for connecting
    page ids across predict.jsonl and archived RO-ED runs.
    """
    return page_id.split("__", 1)[1] if "__" in page_id else page_id


def m5_label(page_id: str, m5_root: Path) -> str | None:
    gt_path = m5_root / "txt_line" / f"{page_id}.txt"
    if not gt_path.exists():
        return None
    lines = []
    for raw in gt_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.rsplit(",", 1)
        lines.append(parts[1] if len(parts) == 2 else raw)
    return "\n".join(lines)


def mthv2_label(page_id: str, mthv2_root: Path) -> str | None:
    if "__" not in page_id:
        return None
    subset, name = page_id.split("__", 1)
    gt_path = mthv2_root / subset / "label_textline" / f"{name}.txt"
    if not gt_path.exists():
        return None
    lines = []
    for raw in gt_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.rsplit(",", 8)
        lines.append(parts[0] if len(parts) == 9 else raw)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["m5", "mthv2"], required=True)
    parser.add_argument("--texts-dir", required=True, help="pipeline output texts/ directory")
    parser.add_argument("--m5-root", default=None, help="path to the M5 dataset root (required for --dataset m5)")
    parser.add_argument("--mthv2-root", default=None, help="path to the MTHv2 dataset root (required for --dataset mthv2)")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--limit", type=int, default=None,
                        help="only take the first N staged pages (smoke runs)")
    parser.add_argument("--keep-relation-tokens", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not (args.m5_root if args.dataset == "m5" else args.mthv2_root):
        parser.error(f"--{'m5-root' if args.dataset == 'm5' else 'mthv2-root'} is required for --dataset {args.dataset}")

    texts_dir = Path(args.texts_dir)
    records = []
    missing_gt = []
    paths = sorted(texts_dir.glob("*.txt"))
    if args.limit is not None:
        paths = paths[: args.limit]
    for path in paths:
        page_id = path.stem
        if args.dataset == "m5":
            label = m5_label(page_id, Path(args.m5_root))
        else:
            label = mthv2_label(page_id, Path(args.mthv2_root))
        if label is None:
            missing_gt.append(page_id)
            continue
        predict = clean_predict(path.read_text(encoding="utf-8"), args.keep_relation_tokens)
        records.append({"page_id": page_id, "predict": predict, "label": label})

    if missing_gt:
        raise ValueError(f"missing ground truth for {len(missing_gt)} pages: {missing_gt[:10]}")
    if not records:
        raise ValueError("no prediction pages found")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {out}; missing GT for {len(missing_gt)} pages")
    if missing_gt:
        print("missing examples:", missing_gt[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
