#!/usr/bin/env python3
"""Compare two pipeline output dirs (e.g. transformers vs vLLM) for consistency.

Compares, per page and per block:
  - block-level OCR texts (pages/*.json -> ocr_blocks[].ocr_text, matched by block_id)
  - page-level joined texts (texts/*.txt)

Reports exact-match rates, character error rate (CER, via editdistance),
block-count mismatches, and the worst diverging examples. Layout/crops are
expected identical: run one full pass first, then rerun each backend into
its own output dir with --skip-layout (copying intermediate/layout/ between
the two output dirs) so both consume the same layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalized(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_pages(root: Path) -> dict[str, dict]:
    pages = {}
    for path in sorted((root / "pages").glob("*.json")):
        pages[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return pages


def load_texts(root: Path) -> dict[str, str]:
    texts = {}
    for path in sorted((root / "texts").glob("*.txt")):
        texts[path.stem] = normalized(path.read_text(encoding="utf-8"))
    return texts


def cer(reference: str, hypothesis: str) -> float:
    import editdistance

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return editdistance.eval(reference, hypothesis) / len(reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-a", required=True, help="e.g. outputs/run_hf")
    parser.add_argument("--dir-b", required=True, help="e.g. outputs/run_vllm")
    parser.add_argument("--name-a", default="transformers")
    parser.add_argument("--name-b", default="vllm")
    parser.add_argument("--report", default=None, help="output JSON report path")
    parser.add_argument("--show-worst", type=int, default=10)
    args = parser.parse_args()

    root_a, root_b = Path(args.dir_a), Path(args.dir_b)
    pages_a, pages_b = load_pages(root_a), load_pages(root_b)
    texts_a, texts_b = load_texts(root_a), load_texts(root_b)

    common = sorted(set(texts_a) & set(texts_b))
    only_a = sorted(set(texts_a) - set(texts_b))
    only_b = sorted(set(texts_b) - set(texts_a))

    page_exact = 0
    page_cers: list[float] = []
    block_total = 0
    block_exact = 0
    block_cers: list[float] = []
    block_examples: list[dict] = []

    for page_id in common:
        ta, tb = texts_a[page_id], texts_b[page_id]
        if ta == tb:
            page_exact += 1
        page_cers.append(cer(ta, tb))

        pa, pb = pages_a.get(page_id, {}), pages_b.get(page_id, {})
        blocks_a = {int(b["block_id"]): normalized(b.get("ocr_text", "")) for b in pa.get("ocr_blocks", [])}
        blocks_b = {int(b["block_id"]): normalized(b.get("ocr_text", "")) for b in pb.get("ocr_blocks", [])}
        for block_id in sorted(set(blocks_a) & set(blocks_b)):
            va, vb = blocks_a[block_id], blocks_b[block_id]
            block_total += 1
            block_cer = cer(va, vb)
            block_cers.append(block_cer)
            if va == vb:
                block_exact += 1
            else:
                block_examples.append(
                    {
                        "page_id": page_id,
                        "block_id": block_id,
                        "cer": round(block_cer, 6),
                        "len_a": len(va),
                        "len_b": len(vb),
                        "text_a": va[:400],
                        "text_b": vb[:400],
                    }
                )

    block_examples.sort(key=lambda item: -item["cer"])
    summary = {
        "dir_a": str(root_a),
        "dir_b": str(root_b),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "pages_compared": len(common),
        "pages_only_in_a": only_a,
        "pages_only_in_b": only_b,
        "page_exact_match": page_exact,
        "page_exact_rate": round(page_exact / len(common), 6) if common else None,
        "page_mean_cer": round(sum(page_cers) / len(page_cers), 6) if page_cers else None,
        "blocks_compared": block_total,
        "block_exact_match": block_exact,
        "block_exact_rate": round(block_exact / block_total, 6) if block_total else None,
        "block_mean_cer": round(sum(block_cers) / len(block_cers), 6) if block_cers else None,
        "worst_blocks": block_examples[: args.show_worst],
    }

    report_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(report_text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
