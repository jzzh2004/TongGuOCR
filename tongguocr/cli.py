from __future__ import annotations

import argparse
import json

from .pipeline import run_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tongguocr",
        description="Run the TongGuOCR layout + block + OCR pipeline.",
    )
    parser.add_argument("--image", help="Run one image, e.g. /path/to/image.jpg")
    parser.add_argument("--input-dir", help="Run all images in one directory.")
    parser.add_argument("--output-dir", required=True, help="Output root directory.")
    parser.add_argument("--config", help="Optional YAML/JSON overriding the built-in defaults.")
    parser.add_argument(
        "--mode",
        choices=["full", "reuse-layout", "direct-ocr"],
        default="full",
        help="Pipeline mode. Defaults to full.",
    )
    parser.add_argument("--ocr-backend", choices=["vllm", "glmocr_vllm", "transformers", "glmocr_hf"], help="OCR backend override.")
    parser.add_argument("--layout-device", help="Layout detector device override, e.g. cuda:2 or cpu.")
    parser.add_argument("--vllm-api-base", help="OpenAI-compatible vLLM API base, e.g. http://127.0.0.1:8090/v1")
    parser.add_argument("--vllm-model-name", help="Served vLLM model name.")
    parser.add_argument("--vllm-launch-server", action="store_true", help="Launch and stop a managed vLLM OpenAI server after crop preparation.")
    parser.add_argument("--vllm-gpu-ids", help="Comma-separated GPU ids for the managed vLLM server.")
    parser.add_argument("--vllm-port", type=int, help="Port for the managed vLLM OpenAI server.")
    parser.add_argument("--vllm-python-bin", help="Python binary for the managed vLLM server environment.")
    parser.add_argument("--vllm-model-path", help="Model path served by the managed vLLM server.")
    parser.add_argument("--vllm-max-tokens", type=int, help="Max OCR generation tokens for vLLM backends.")
    parser.add_argument("--vllm-max-model-len", type=int, help="Max model length for managed/local vLLM.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, help="GPU memory utilization for managed/local vLLM.")
    parser.add_argument("--glmocr-python-bin", help="Python binary for external GLMOCR helper environment.")
    parser.add_argument("--glmocr-model-path", help="Model path used by the external GLMOCR helper.")
    parser.add_argument("--glmocr-gpu-ids", help="Comma-separated GPU ids for the external GLMOCR helper.")
    parser.add_argument("--glmocr-concurrency", type=int, help="Worker count for the external GLMOCR helper.")
    parser.add_argument(
        "--crop-mode",
        choices=["padded_square", "raw_bbox", "square_no_padding"],
        help="OCR crop strategy. Defaults to the built-in config.",
    )
    parser.add_argument(
        "--disable-foreign-mask",
        action="store_true",
        help="Do not mask foreign text-line regions inside a block crop.",
    )
    parser.add_argument("--print-result", action="store_true", help="Print single-image OCR preview JSON.")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Do not write visualization images.")
    parser.add_argument("--no-crop-blocks", dest="crop_blocks", action="store_false", help="Do not retain cropped block images.")
    parser.add_argument("--debug", action="store_true", help="Write extra block debug metrics.")
    parser.add_argument("--skip-layout", action="store_true", help="Reuse intermediate/layout/*.json instead of rerunning layout.")
    parser.add_argument("--workers", type=int, default=1, help="Worker process count for batch mode.")
    parser.add_argument("--limit", type=int, help="Only process the first N images in sorted order.")
    parser.set_defaults(visualize=None, crop_blocks=None)
    args = parser.parse_args()
    if bool(args.image) == bool(args.input_dir):
        parser.error("Exactly one of --image or --input-dir must be provided.")
    return args


def main() -> None:
    args = parse_args()
    summary, preview = run_from_args(args)
    if preview is not None and (args.print_result or args.image):
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
