#!/usr/bin/env python3
"""Build staging directories of ORIGINAL dataset test images (symlinks).

Does not touch the source data. Supports any subset size for consistency
checks and the full test sets for final runs.

Expected dataset layouts (paths are passed via --m5-root / --mthv2-root):
  M5:    <m5-root>/split_test.txt  ->  <m5-root>/im/<id>.jpg
  MTHv2: <mthv2-root>/MTHv2test.txt lists image paths relative to the root
         (absolute paths from the original distribution are remapped by
         locating the subset component); images live under
         {MTH1000,MTH1200,TKH}/img/. Staged names are `<SUBSET>__<name>`.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

SUBSETS = ("MTH1000", "MTH1200", "TKH")


def stage_m5(root: Path, limit: int, out: Path) -> int:
    ids = [l.strip() for l in (root / "split_test.txt").read_text().splitlines() if l.strip()]
    if limit > 0:
        ids = ids[:limit]
    out.mkdir(parents=True, exist_ok=True)
    for page_id in ids:
        src = root / "im" / f"{page_id}.jpg"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    return len(ids)


def _relative_mth_line(line: str) -> str:
    """Map one MTHv2test.txt line to a path relative to the dataset root.

    The original distribution lists absolute paths from the annotator's
    machine; locate the subset component (MTH1000/MTH1200/TKH) inside the
    line and keep everything from there on.
    """
    parts = PurePosixPath(line.replace("\\", "/")).parts
    for i, part in enumerate(parts):
        if part in SUBSETS:
            return "/".join(parts[i:])
    return "/".join(p for p in parts if p not in ("..",))


def stage_mthv2(root: Path, limit: int, out: Path) -> int:
    lines = [l.strip() for l in (root / "MTHv2test.txt").read_text().splitlines() if l.strip()]
    if limit > 0:
        lines = lines[:limit]
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for line in lines:
        rel = _relative_mth_line(line)
        src = root / rel
        if not src.exists():
            raise FileNotFoundError(src)
        subset = rel.split("/", 1)[0]
        dst = out / f"{subset}__{src.name}"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["m5", "mthv2"], required=True)
    parser.add_argument("--m5-root", type=Path, help="M5 dataset root (contains im/ and split_test.txt)")
    parser.add_argument("--mthv2-root", type=Path, help="MTHv2 dataset root (contains MTHv2test.txt)")
    parser.add_argument("--limit", type=int, default=0, help="0 = full test set")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.dataset == "m5":
        if not args.m5_root:
            parser.error("--m5-root is required for --dataset m5")
        n = stage_m5(args.m5_root, args.limit, Path(args.output_dir))
    else:
        if not args.mthv2_root:
            parser.error("--mthv2-root is required for --dataset mthv2")
        n = stage_mthv2(args.mthv2_root, args.limit, Path(args.output_dir))
    print(f"staged {n} {args.dataset} images into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
