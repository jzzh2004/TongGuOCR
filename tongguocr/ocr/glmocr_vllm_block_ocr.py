#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pkgutil
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter  # type: ignore[attr-defined]

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

try:
    from .relation_tokens import finalize_ocr_text
except ImportError:  # executed as a standalone helper script
    from relation_tokens import finalize_ocr_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local vLLM GLM-OCR on prepared tongguocr pages.")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Text Recognition:")
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)  # single GPU to match the HF backend's device_map; 0 used to silently grab ALL visible GPUs
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--page-id", action="append")
    return parser.parse_args()


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return len([part for part in visible.split(",") if part.strip()])
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 1


def _load_page_payloads(prepared_root: Path, page_ids: list[str] | None) -> list[dict[str, Any]]:
    pages_dir = prepared_root / "pages"
    selected = set(page_ids or [])
    records: list[dict[str, Any]] = []
    for path in sorted(pages_dir.glob("*.json")):
        if selected and path.stem not in selected:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_page_json_path"] = str(path)
        records.append(payload)
    return records


def _should_skip(record: dict[str, Any], prepared_root: Path, resume: bool) -> bool:
    if not resume:
        return False
    page_id = str(record["page_id"])
    return (
        (prepared_root / "texts" / f"{page_id}.txt").exists()
        and record.get("status") == "ok"
        and bool(record.get("ocr_blocks"))
    )


def _open_image(path: Path, max_pixels: int) -> Image.Image:
    # Use PIL's default resample (same as glmocr_block_ocr._prepare_pil_image)
    # so the offline vLLM path and the transformers path see identical pixels.
    image = Image.open(path).convert("RGB")
    if max_pixels > 0 and image.width * image.height > max_pixels:
        resize_factor = (max_pixels / (image.width * image.height)) ** 0.5
        width = max(1, int(image.width * resize_factor))
        height = max(1, int(image.height * resize_factor))
        image = image.resize((width, height))
    return image


def _build_prompt(processor: Any, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _sanitize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _target_line_count(block: dict[str, Any]) -> int:
    line_ids = block.get("line_ids") or []
    if line_ids:
        return max(1, len(line_ids))
    line_range = block.get("line_range") or []
    if len(line_range) == 2:
        try:
            return max(1, int(line_range[1]) - int(line_range[0]) + 1)
        except (TypeError, ValueError):
            return 1
    return 1


def _split_text_by_line_count(text: str, line_count: int) -> str:
    text = _sanitize_text(text)
    if line_count <= 1 or not text:
        return text
    existing_lines = [line for line in text.split("\n") if line.strip()]
    if len(existing_lines) >= max(2, line_count // 2):
        return text
    total = len(text)
    if total <= line_count:
        return "\n".join(text)
    punctuation = set("。！？；;，,、：:")
    boundaries: list[int] = []
    previous = 0
    for idx in range(1, line_count):
        ideal = round(total * idx / line_count)
        min_pos = max(previous + 1, ideal - max(3, total // line_count // 3))
        max_pos = min(total - (line_count - idx), ideal + max(3, total // line_count // 3))
        best = ideal
        for pos in range(min_pos, max_pos + 1):
            if 0 < pos < total and text[pos - 1] in punctuation:
                best = pos
                break
        best = max(previous + 1, min(best, total - (line_count - idx)))
        boundaries.append(best)
        previous = best
    parts: list[str] = []
    start = 0
    for end in boundaries + [total]:
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    return "\n".join(parts)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    prepared_root = Path(args.prepared_root).resolve()
    texts_dir = prepared_root / "texts"
    logs_dir = prepared_root / "logs"
    texts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    records = [
        record
        for record in _load_page_payloads(prepared_root, args.page_id)
        if not _should_skip(record, prepared_root, args.resume)
    ]
    tensor_parallel_size = args.tensor_parallel_size or max(1, _visible_gpu_count())
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        # keep special tokens so the relation tokens survive decoding;
        # finalize_ocr_text() below turns them into newlines
        skip_special_tokens=False,
    )

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in tqdm(records, desc="GLMOCR vLLM", unit="page"):
        page_started = time.perf_counter()
        page_id = str(record["page_id"])
        page_json_path = Path(str(record["_page_json_path"]))
        try:
            page_payload = json.loads(page_json_path.read_text(encoding="utf-8"))
            blocks = sorted(page_payload.get("blocks", []), key=lambda item: int(item["block_id"]))
            ocr_blocks: list[dict[str, Any]] = []
            block_texts: list[str] = []
            block_latencies_ms: list[float] = []
            for block in blocks:
                crop_value = block.get("masked_crop_path") or block.get("crop_path")
                if not crop_value:
                    raise ValueError(f"Block {block.get('block_id')} missing crop path")
                crop_path = Path(str(crop_value))
                image = _open_image(crop_path, args.image_max_pixels)
                prompt = _build_prompt(processor, args.prompt)
                block_started = time.perf_counter()
                outputs = llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    },
                    sampling_params=sampling_params,
                )
                text = _split_text_by_line_count(
                    finalize_ocr_text(outputs[0].outputs[0].text), _target_line_count(block)
                )
                latency_ms = round((time.perf_counter() - block_started) * 1000.0, 3)
                block_latencies_ms.append(latency_ms)
                block_texts.append(text)
                ocr_blocks.append(
                    {
                        "block_id": int(block["block_id"]),
                        "crop_path": str(block.get("crop_path") or crop_path),
                        "masked_crop_path": str(block.get("masked_crop_path") or crop_path),
                        "ocr_text": text,
                        "backend": "glmocr_vllm",
                        "latency_ms": latency_ms,
                    }
                )

            full_text = "\n".join(text for text in block_texts if text.strip())
            text_path = texts_dir / f"{page_id}.txt"
            text_path.write_text(full_text, encoding="utf-8")
            page_payload["backend"] = "glmocr_vllm"
            page_payload["ocr_blocks"] = ocr_blocks
            page_payload["ocr_text"] = full_text
            page_payload["text_path"] = str(text_path)
            page_payload["status"] = "ok"
            page_payload.setdefault("debug", {})["glmocr_vllm_block_latencies_ms"] = block_latencies_ms
            page_json_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            success_rows.append(
                {
                    "page_id": page_id,
                    "latency_sec": round(time.perf_counter() - page_started, 6),
                    "block_count": len(blocks),
                }
            )
        except Exception as exc:
            failed_rows.append({"page_id": page_id, "error": f"{type(exc).__name__}: {exc}"})
            try:
                page_payload = json.loads(page_json_path.read_text(encoding="utf-8"))
                page_payload["status"] = "error"
                page_payload.setdefault("debug", {})["glmocr_vllm_error"] = failed_rows[-1]["error"]
                page_json_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    summary = {
        "backend": "glmocr_vllm",
        "prepared_root": str(prepared_root),
        "record_count": len(records),
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "image_max_pixels": args.image_max_pixels,
        "failed_pages": sorted(failed_rows, key=lambda item: item["page_id"]),
    }
    (logs_dir / "glmocr_vllm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failed_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
