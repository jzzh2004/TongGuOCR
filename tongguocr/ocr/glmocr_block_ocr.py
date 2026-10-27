#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any

from tqdm import tqdm

try:
    from .relation_tokens import finalize_ocr_text
except ImportError:  # executed as a standalone helper script
    from relation_tokens import finalize_ocr_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLMOCR HF on prepared tongguocr pages/*.json + block_crops/")
    parser.add_argument("--prepared-root", required=True, help="Pipeline output root containing pages/ and block_crops/")
    parser.add_argument("--model-path", required=True, help="GLMOCR model path")
    parser.add_argument("--prompt", default="Text Recognition:", help="OCR prompt")
    parser.add_argument(
        "--resize-mode",
        choices=["thumb1024", "image_max_pixels"],
        default="image_max_pixels",
    )
    parser.add_argument("--image-max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--image-min-pixels", type=int, default=32 * 32)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--page-id", action="append", help="Specific page id to process. May be repeated.")
    return parser.parse_args()


def _parse_gpu_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _prepare_pil_image(
    image_path: Path,
    resize_mode: str,
    image_max_pixels: int,
    image_min_pixels: int,
):
    from PIL import Image

    image = Image.open(image_path)
    if resize_mode == "thumb1024":
        image.thumbnail((1024, 1024))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    if resize_mode == "image_max_pixels":
        total_pixels = image.width * image.height
        if total_pixels > image_max_pixels:
            resize_factor = (image_max_pixels / total_pixels) ** 0.5
            width = max(1, int(image.width * resize_factor))
            height = max(1, int(image.height * resize_factor))
            image = image.resize((width, height))

        total_pixels = image.width * image.height
        if total_pixels < image_min_pixels:
            resize_factor = (image_min_pixels / total_pixels) ** 0.5
            width = max(1, int(image.width * resize_factor))
            height = max(1, int(image.height * resize_factor))
            image = image.resize((width, height))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


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


def _restore_block_line_breaks(text: str, block: dict[str, Any], enabled: bool) -> str:
    if not enabled:
        return _sanitize_text(text)
    return _split_text_by_line_count(text, _target_line_count(block))


def _load_page_payloads(prepared_root: Path, page_ids: list[str] | None) -> list[dict[str, Any]]:
    pages_dir = prepared_root / "pages"
    page_paths = sorted(pages_dir.glob("*.json"))
    selected = set(page_ids or [])
    records: list[dict[str, Any]] = []
    for path in page_paths:
        if selected and path.stem not in selected:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_page_json_path"] = str(path)
        records.append(payload)
    return records


def _shard_records(records: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(shard_count)]
    for index, record in enumerate(records):
        shards[index % shard_count].append(record)
    return shards


def _maybe_skip_record(record: dict[str, Any], prepared_root: Path, resume: bool) -> bool:
    if not resume:
        return False
    page_id = str(record["page_id"])
    text_path = prepared_root / "texts" / f"{page_id}.txt"
    if not text_path.exists():
        return False
    if record.get("status") == "ok" and record.get("ocr_blocks"):
        return True
    return False


def _worker_main(
    gpu_id: int,
    records: list[dict[str, Any]],
    worker_args: dict[str, Any],
    result_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch
        from transformers import AutoProcessor
        from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrForConditionalGeneration

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        processor = AutoProcessor.from_pretrained(worker_args["model_path"], trust_remote_code=True)
        setattr(processor, "image_max_pixels", int(worker_args["image_max_pixels"]))
        setattr(processor, "image_min_pixels", int(worker_args["image_min_pixels"]))
        setattr(processor, "image_do_pan_and_scan", False)
        setattr(processor, "crop_to_patches", False)
        model = GlmOcrForConditionalGeneration.from_pretrained(
            worker_args["model_path"],
            torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        prepared_root = Path(worker_args["prepared_root"])
        texts_dir = prepared_root / "texts"
        texts_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
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
                    if not crop_path.exists():
                        raise FileNotFoundError(f"Missing crop image: {crop_path}")

                    image = _prepare_pil_image(
                        crop_path,
                        worker_args["resize_mode"],
                        int(worker_args["image_max_pixels"]),
                        int(worker_args["image_min_pixels"]),
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": worker_args["prompt"]},
                            ],
                        }
                    ]
                    prompt_text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    block_started = time.perf_counter()
                    inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=worker_args["max_new_tokens"],
                            do_sample=False,
                        )
                    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
                    decoded = processor.decode(generated_ids, skip_special_tokens=False)
                    # relation tokens are special tokens: decode with them kept,
                    # then turn them into newlines (reading-order line breaks)
                    decoded = finalize_ocr_text(
                        decoded, getattr(processor, "tokenizer", processor).all_special_tokens
                    )
                    predict = _restore_block_line_breaks(
                        decoded,
                        block,
                        bool(worker_args.get("restore_line_breaks", True)),
                    )
                    latency_ms = round((time.perf_counter() - block_started) * 1000.0, 3)
                    block_latencies_ms.append(latency_ms)
                    block_texts.append(predict)
                    ocr_blocks.append(
                        {
                            "block_id": int(block["block_id"]),
                            "crop_path": str(block.get("crop_path") or crop_path),
                            "masked_crop_path": str(block.get("masked_crop_path") or crop_path),
                            "ocr_text": predict,
                            "backend": "glmocr_hf",
                            "latency_ms": latency_ms,
                        }
                    )

                full_text = "\n".join(text for text in block_texts if text.strip())
                text_path = texts_dir / f"{page_id}.txt"
                text_path.write_text(full_text, encoding="utf-8")

                page_payload["backend"] = "glmocr_hf"
                page_payload["ocr_blocks"] = ocr_blocks
                page_payload["ocr_text"] = full_text
                page_payload["text_path"] = str(text_path)
                page_payload["status"] = "ok"
                debug_payload = dict(page_payload.get("debug", {}))
                debug_payload["glmocr_block_latencies_ms"] = block_latencies_ms
                page_payload["debug"] = debug_payload
                page_json_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                result_queue.put(
                    {
                        "status": "ok",
                        "page_id": page_id,
                        "latency_sec": round(time.perf_counter() - page_started, 6),
                        "block_count": len(blocks),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - external env
                try:
                    page_payload = json.loads(page_json_path.read_text(encoding="utf-8"))
                    page_payload["status"] = "error"
                    page_payload.setdefault("debug", {})["glmocr_error"] = f"{type(exc).__name__}: {exc}"
                    page_json_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                result_queue.put(
                    {
                        "status": "error",
                        "page_id": page_id,
                        "latency_sec": round(time.perf_counter() - page_started, 6),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        result_queue.put({"status": "worker_done", "gpu_id": gpu_id})


def main() -> int:
    args = parse_args()
    prepared_root = Path(args.prepared_root).resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)
    (prepared_root / "logs").mkdir(parents=True, exist_ok=True)

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    worker_count = max(1, min(len(gpu_ids), args.concurrency if args.concurrency > 0 else len(gpu_ids)))

    page_records = _load_page_payloads(prepared_root, args.page_id)
    selected_records: list[dict[str, Any]] = []
    for record in page_records:
        if _maybe_skip_record(record, prepared_root, args.resume):
            continue
        selected_records.append(record)

    started = time.perf_counter()
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    if selected_records:
        shards = _shard_records(selected_records, worker_count)
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        worker_args = {
            "prepared_root": str(prepared_root),
            "model_path": str(Path(args.model_path).resolve()),
            "prompt": args.prompt,
            "resize_mode": args.resize_mode,
            "image_max_pixels": args.image_max_pixels,
            "image_min_pixels": args.image_min_pixels,
            "max_new_tokens": args.max_new_tokens,
            "restore_line_breaks": True,
        }

        for gpu_id, shard in zip(gpu_ids[:worker_count], shards):
            process = ctx.Process(target=_worker_main, args=(gpu_id, shard, worker_args, result_queue))
            process.start()
            processes.append(process)

        progress = tqdm(total=len(selected_records), desc="GLMOCR HF", unit="page")
        completed_workers = 0
        while completed_workers < len(processes):
            try:
                item = result_queue.get(timeout=1.0)
            except Empty:
                continue
            if item.get("status") == "worker_done":
                completed_workers += 1
                continue
            if item.get("status") == "ok":
                success_rows.append(item)
            else:
                failed_rows.append(item)
            progress.update(1)
        progress.close()

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"GLMOCR worker exited with code {process.exitcode}")

    elapsed_sec = round(time.perf_counter() - started, 6)
    all_records = page_records
    summary = {
        "backend": "glmocr_hf",
        "prepared_root": str(prepared_root),
        "record_count": len(all_records),
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "elapsed_sec": elapsed_sec,
        "resize_mode": args.resize_mode,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "max_new_tokens": args.max_new_tokens,
        "gpu_ids": gpu_ids[:worker_count],
        "concurrency": worker_count,
        "avg_page_latency_sec": round(sum(item["latency_sec"] for item in success_rows) / len(success_rows), 6)
        if success_rows
        else 0.0,
        "avg_blocks_per_page": round(sum(item["block_count"] for item in success_rows) / len(success_rows), 6)
        if success_rows
        else 0.0,
        "failed_pages": sorted(
            [
                {"page_id": item["page_id"], "error": item.get("error", "")}
                for item in failed_rows
            ],
            key=lambda item: item["page_id"],
        ),
    }
    summary_path = prepared_root / "logs" / "glmocr_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failed_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
