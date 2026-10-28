from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tongguocr",
        description="TongGuOCR: layout-aware OCR for Chinese historical documents.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="A page image or a directory of page images (jpg/png/bmp/webp/tif).",
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="OCR backend. 'transformers' runs locally on the GPU; 'vllm' is an "
        "OpenAI-compatible client for a running vLLM server (see docs/vllm.md).",
    )
    parser.add_argument(
        "--device",
        help="GPU/device for layout detection and the transformers backend, e.g. cuda:0.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N images (dir mode).")
    parser.add_argument(
        "--print-result",
        action="store_true",
        help="Print the recognized text of a single-image run to stdout.",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Do not write layout visualization images.",
    )
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="Reuse layout results from a previous run of this output dir (resume).",
    )
    parser.add_argument(
        "--config",
        help="Optional YAML/JSON overriding built-in defaults (see tongguocr/config.py).",
    )
    parser.set_defaults(visualize=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.input).exists():
        raise SystemExit(f"error: input not found: {args.input}")
    summary, preview = run_from_args(args)
    if preview is not None and (args.print_result or Path(args.input).is_file()):
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
