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
    parser = argparse.ArgumentParser(description="Run GLMOCR HF directly on prepared tongguocr pages/*.json.")
    parser.add_argument("--prepared-root", required=True, help="Pipeline output root containing pages/")
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


def _sanitize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


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
    return record.get("status") == "ok" and bool(record.get("ocr_text", "").strip())


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
        # Mirror the processor patching needed by the OCR inference runtime.
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
                image_path = Path(str(page_payload["image_path"]))
                if not image_path.exists():
                    raise FileNotFoundError(f"Missing image: {image_path}")

                image = _prepare_pil_image(
                    image_path,
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
                predict = _sanitize_text(decoded)

                text_path = texts_dir / f"{page_id}.txt"
                text_path.write_text(predict, encoding="utf-8")
                page_payload["backend"] = "glmocr_hf"
                page_payload["mode"] = "direct_ocr"
                page_payload["ocr_blocks"] = []
                page_payload["ocr_text"] = predict
                page_payload["text_path"] = str(text_path)
                page_payload["status"] = "ok"
                page_payload["block_count"] = 0
                page_payload["line_count"] = 0
                page_payload["blocks"] = []
                page_payload["line_bboxes"] = []
                page_payload["crop_dir"] = None
                page_payload["visual_path"] = None
                page_json_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                result_queue.put(
                    {
                        "status": "ok",
                        "page_id": page_id,
                        "latency_sec": round(time.perf_counter() - page_started, 6),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - external env
                try:
                    page_payload = json.loads(page_json_path.read_text(encoding="utf-8"))
                    page_payload["status"] = "error"
                    page_payload.setdefault("debug", {})["glmocr_direct_error"] = f"{type(exc).__name__}: {exc}"
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
    selected_records = [
        record
        for record in page_records
        if not _maybe_skip_record(record, prepared_root, args.resume)
    ]

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
        }

        for gpu_id, shard in zip(gpu_ids[:worker_count], shards):
            process = ctx.Process(target=_worker_main, args=(gpu_id, shard, worker_args, result_queue))
            process.start()
            processes.append(process)

        progress = tqdm(total=len(selected_records), desc="GLMOCR HF Direct", unit="page")
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
                raise RuntimeError(f"GLMOCR direct worker exited with code {process.exitcode}")

    elapsed_sec = round(time.perf_counter() - started, 6)
    summary = {
        "backend": "glmocr_hf",
        "mode": "direct_ocr",
        "prepared_root": str(prepared_root),
        "record_count": len(page_records),
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "elapsed_sec": elapsed_sec,
        "gpu_ids": gpu_ids[:worker_count],
        "concurrency": worker_count,
        "resize_mode": args.resize_mode,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "max_new_tokens": args.max_new_tokens,
        "avg_page_latency_sec": round(sum(item["latency_sec"] for item in success_rows) / len(success_rows), 6)
        if success_rows
        else 0.0,
        "failed_pages": sorted(
            [{"page_id": item["page_id"], "error": item.get("error", "")} for item in failed_rows],
            key=lambda item: item["page_id"],
        ),
    }
    summary_path = prepared_root / "logs" / "glmocr_direct_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failed_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
