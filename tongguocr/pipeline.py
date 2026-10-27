from __future__ import annotations
import contextlib
import gc
import json
import logging
import os
import socket
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
import requests

from .blocking.dp_solver import segment_page
from .blocking.models import TextLine
from .config import PROJECT_ROOT, load_config
from .io import (
    ensure_output_dirs,
    iter_input_images,
    read_layout_result,
    write_layout_result,
    write_page_result,
    write_summary,
    write_text_result,
)
from .layout import (
    LayoutPageResult,
    clear_layout_extractor_cache,
    extract_line_bboxes,
    layout_page_result_from_dict,
    layout_page_result_to_dict,
)
from .models import OcrBlockResult, PipelinePageResult, PipelineRunSummary
from .ocr import recognize_blocks
from .crop import SavedBlockCrop, save_block_crops
from .visualize import draw_pipeline_visualization


LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedPage:
    page_id: str
    mode: str
    image_path: str
    image_size: tuple[int, int]
    line_bboxes: list[dict[str, Any]]
    line_count: int
    blocks: list[dict[str, Any]]
    block_count: int
    visual_path: str | None
    crop_dir: str | None
    crop_count: int
    debug_payload: dict[str, Any]


def _is_glmocr_hf_backend(config: dict[str, Any]) -> bool:
    return config.get("ocr", {}).get("backend") == "glmocr_hf"


def _is_glmocr_vllm_backend(config: dict[str, Any]) -> bool:
    return config.get("ocr", {}).get("backend") == "glmocr_vllm"


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _glmocr_backend_cfg(config: dict[str, Any], *, direct: bool = False) -> dict[str, Any]:
    ocr_cfg = config.get("ocr", {})
    if direct:
        return ocr_cfg.get("glmocr_hf_direct") or ocr_cfg["glmocr_hf"]
    return ocr_cfg["glmocr_hf"]


def _apply_glmocr_override(config: dict[str, Any], field: str, value: Any) -> None:
    config["ocr"]["glmocr_hf"][field] = value
    if "glmocr_hf_direct" in config.get("ocr", {}):
        config["ocr"]["glmocr_hf_direct"][field] = value


def _apply_vllm_server_override(config: dict[str, Any], field: str, value: Any) -> None:
    config.setdefault("ocr", {}).setdefault("vllm", {}).setdefault("server", {})[field] = value


def _apply_layout_device_override(config: dict[str, Any], layout_device: str) -> None:
    layout_device = str(layout_device).strip()
    if layout_device.startswith("cuda:"):
        physical_gpu = layout_device.split(":", 1)[1]
        if "," not in physical_gpu and physical_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
            config.setdefault("layout_inference", {})["device"] = "cuda:0"
            config.setdefault("_runtime", {})["layout_physical_device"] = layout_device
            return
    config.setdefault("layout_inference", {})["device"] = layout_device
    config.setdefault("_runtime", {})["layout_physical_device"] = layout_device


def setup_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def _layout_lines_to_block_lines(layout_page: LayoutPageResult) -> list[TextLine]:
    return [
        TextLine(line_id=line.line_id, text=line.text, bbox=line.bbox)
        for line in layout_page.lines
    ]


def _layout_lines_to_dicts(layout_page: LayoutPageResult) -> list[dict[str, Any]]:
    return [
        {
            "line_id": line.line_id,
            "text": line.text,
            "bbox": list(line.bbox),
        }
        for line in layout_page.lines
    ]


def _build_block_entries(
    page_result,
    crop_outputs: list[SavedBlockCrop],
) -> list[dict[str, Any]]:
    crop_map = {item.block_id: item for item in crop_outputs}
    block_entries: list[dict[str, Any]] = []
    for block in page_result.blocks:
        crop_output = crop_map.get(block.block_id)
        block_entries.append(
            {
                "block_id": block.block_id,
                "label": block.label,
                "bbox": list(block.bbox),
                "line_ids": block.line_ids,
                "line_range": block.line_range,
                "crop_path": str(crop_output.path) if crop_output else None,
                "masked_crop_path": str(crop_output.path) if crop_output else None,
                "background_fill_color": crop_output.background_fill_color if crop_output else None,
                "crop_canvas_size": crop_output.crop_canvas_size if crop_output else None,
                "sparse_canvas_applied": crop_output.sparse_canvas_applied if crop_output else False,
                "stats": block.stats,
                "cost": block.cost,
            }
        )
    return block_entries


def _compose_ocr_text(ocr_blocks: list[OcrBlockResult]) -> str:
    ordered_texts = [item.ocr_text.strip() for item in sorted(ocr_blocks, key=lambda item: item.block_id)]
    return "\n".join(text for text in ordered_texts if text)


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
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
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


def _restore_ocr_block_line_breaks(
    ocr_blocks: list[OcrBlockResult],
    block_entries: list[dict[str, Any]],
) -> list[OcrBlockResult]:
    block_map = {int(item["block_id"]): item for item in block_entries}
    restored: list[OcrBlockResult] = []
    for item in ocr_blocks:
        block = block_map.get(int(item.block_id), {})
        restored.append(
            OcrBlockResult(
                block_id=item.block_id,
                crop_path=item.crop_path,
                masked_crop_path=item.masked_crop_path,
                ocr_text=_split_text_by_line_count(item.ocr_text, _target_line_count(block)),
                backend=item.backend,
                latency_ms=item.latency_ms,
            )
        )
    return restored


def _build_page_preview(page_result: PipelinePageResult) -> dict[str, Any]:
    return {
        "page_id": page_result.page_id,
        "mode": page_result.mode,
        "backend": page_result.backend,
        "line_count": page_result.line_count,
        "block_count": page_result.block_count,
        "ocr_blocks": [
            {
                "block_id": block.block_id,
                "ocr_text": block.ocr_text,
            }
            for block in page_result.ocr_blocks
        ],
        "ocr_text": page_result.ocr_text,
    }


def _prepared_page_to_stub_result(prepared: PreparedPage, backend: str) -> PipelinePageResult:
    return PipelinePageResult(
        page_id=prepared.page_id,
        mode=prepared.mode,
        image_path=prepared.image_path,
        image_size=prepared.image_size,
        status="prepared",
        backend=backend,
        line_bboxes=prepared.line_bboxes,
        line_count=prepared.line_count,
        blocks=prepared.blocks,
        block_count=prepared.block_count,
        ocr_blocks=[],
        ocr_text="",
        text_path=None,
        visual_path=prepared.visual_path,
        crop_dir=prepared.crop_dir,
        debug=prepared.debug_payload,
    )


def _write_prepared_page_stub(
    prepared: PreparedPage,
    config: dict[str, Any],
    output_dirs: dict[str, Path],
) -> Path:
    stub_result = _prepared_page_to_stub_result(prepared, config["ocr"]["backend"])
    return write_page_result(
        output_dirs["pages"],
        stub_result.to_dict(include_debug=bool(prepared.debug_payload)),
    )


def _page_result_from_payload(payload: dict[str, Any]) -> PipelinePageResult:
    ocr_blocks = [
        OcrBlockResult(
            block_id=int(item["block_id"]),
            crop_path=str(item.get("crop_path") or ""),
            masked_crop_path=(
                str(item["masked_crop_path"])
                if item.get("masked_crop_path") is not None
                else None
            ),
            ocr_text=str(item.get("ocr_text") or ""),
            backend=str(item.get("backend") or payload.get("backend") or ""),
            latency_ms=float(item["latency_ms"]) if item.get("latency_ms") is not None else None,
        )
        for item in payload.get("ocr_blocks", [])
    ]
    image_size = payload.get("image_size") or [0, 0]
    return PipelinePageResult(
        page_id=str(payload["page_id"]),
        image_path=str(payload["image_path"]),
        image_size=(int(image_size[0]), int(image_size[1])),
        status=str(payload.get("status") or "ok"),
        backend=str(payload.get("backend") or ""),
        mode=str(payload.get("mode") or "full"),
        line_bboxes=list(payload.get("line_bboxes", [])),
        line_count=int(payload.get("line_count", 0)),
        blocks=list(payload.get("blocks", [])),
        block_count=int(payload.get("block_count", 0)),
        ocr_blocks=ocr_blocks,
        ocr_text=str(payload.get("ocr_text") or ""),
        text_path=payload.get("text_path"),
        visual_path=payload.get("visual_path"),
        crop_dir=payload.get("crop_dir"),
        debug=dict(payload.get("debug", {})),
    )


def _load_page_result(output_dirs: dict[str, Path], page_id: str) -> PipelinePageResult:
    page_json_path = output_dirs["pages"] / f"{page_id}.json"
    payload = json.loads(page_json_path.read_text(encoding="utf-8"))
    return _page_result_from_payload(payload)


def _summary_entry_from_page_result(page_result: PipelinePageResult) -> dict[str, Any]:
    return {
        "status": page_result.status,
        "page_id": page_result.page_id,
        "mode": page_result.mode,
        "image_path": page_result.image_path,
        "line_count": page_result.line_count,
        "block_count": page_result.block_count,
        "backend": page_result.backend,
        "page_json_path": None,
        "text_path": page_result.text_path,
        "visual_path": page_result.visual_path,
        "crop_dir": page_result.crop_dir,
        "crop_count": len(page_result.blocks) if page_result.crop_dir else 0,
    }


def _build_glmocr_helper_command(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    page_ids: list[str] | None = None,
) -> list[str]:
    backend_cfg = _glmocr_backend_cfg(config)
    command = [
        str(backend_cfg["python_bin"]),
        str(_resolve_project_path(backend_cfg["helper_script"])),
        "--prepared-root",
        str(output_root),
        "--model-path",
        str(_resolve_project_path(backend_cfg["model_path"])),
        "--prompt",
        str(backend_cfg.get("prompt", "Text Recognition:")),
        "--resize-mode",
        str(backend_cfg.get("resize_mode", "thumb1024")),
        "--image-max-pixels",
        str(int(backend_cfg.get("image_max_pixels", 1024 * 1024))),
        "--image-min-pixels",
        str(int(backend_cfg.get("image_min_pixels", 32 * 32))),
        "--max-new-tokens",
        str(int(backend_cfg.get("max_tokens", 8192))),
        "--gpu-ids",
        str(backend_cfg.get("gpu_ids", "0,1,2,3")),
        "--concurrency",
        str(int(backend_cfg.get("concurrency", 4))),
    ]
    if backend_cfg.get("resume", True):
        command.append("--resume")
    for page_id in page_ids or []:
        command.extend(["--page-id", page_id])
    return command


def _build_glmocr_direct_helper_command(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    page_ids: list[str] | None = None,
) -> list[str]:
    backend_cfg = _glmocr_backend_cfg(config, direct=True)
    command = [
        str(backend_cfg["python_bin"]),
        str(_resolve_project_path(backend_cfg.get("helper_script", "./tongguocr/ocr/glmocr_direct_ocr.py"))),
        "--prepared-root",
        str(output_root),
        "--model-path",
        str(_resolve_project_path(backend_cfg["model_path"])),
        "--prompt",
        str(backend_cfg.get("prompt", "Text Recognition:")),
        "--resize-mode",
        str(backend_cfg.get("resize_mode", "thumb1024")),
        "--image-max-pixels",
        str(int(backend_cfg.get("image_max_pixels", 1024 * 1024))),
        "--image-min-pixels",
        str(int(backend_cfg.get("image_min_pixels", 32 * 32))),
        "--max-new-tokens",
        str(int(backend_cfg.get("max_tokens", 8192))),
        "--gpu-ids",
        str(backend_cfg.get("gpu_ids", "0,1,2,3")),
        "--concurrency",
        str(int(backend_cfg.get("concurrency", 4))),
    ]
    if backend_cfg.get("resume", True):
        command.append("--resume")
    for page_id in page_ids or []:
        command.extend(["--page-id", page_id])
    return command


def _run_glmocr_helper(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    page_ids: list[str] | None = None,
) -> None:
    command = _build_glmocr_helper_command(config, output_root, page_ids=page_ids)
    LOGGER.info("Running glmocr_hf helper: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if completed.stdout.strip():
        LOGGER.info("glmocr_hf stdout:\n%s", completed.stdout.strip())
    if completed.stderr.strip():
        LOGGER.warning("glmocr_hf stderr:\n%s", completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"glmocr_hf helper exited with code {completed.returncode}")


def _run_glmocr_direct_helper(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    page_ids: list[str] | None = None,
) -> None:
    command = _build_glmocr_direct_helper_command(config, output_root, page_ids=page_ids)
    LOGGER.info("Running glmocr_hf direct helper: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if completed.stdout.strip():
        LOGGER.info("glmocr_hf direct stdout:\n%s", completed.stdout.strip())
    if completed.stderr.strip():
        LOGGER.warning("glmocr_hf direct stderr:\n%s", completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"glmocr_hf direct helper exited with code {completed.returncode}")


def _build_glmocr_vllm_helper_command(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    page_ids: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    vllm_cfg = config["ocr"]["vllm"]
    server_cfg = vllm_cfg.get("server", {})
    gpu_ids = _split_gpu_ids(server_cfg.get("gpu_ids", "0"))
    max_tokens = int(vllm_cfg.get("max_tokens", 8192))
    max_model_len = int(server_cfg.get("local_max_model_len") or max(32768, max_tokens * 2))
    command = [
        str(server_cfg.get("python_bin", "python")),
        str(_resolve_project_path(server_cfg.get("helper_script", "./tongguocr/ocr/glmocr_vllm_block_ocr.py"))),
        "--prepared-root",
        str(output_root),
        "--model-path",
        str(_resolve_project_path(server_cfg.get("model_path", "./weights/ocr"))),
        "--prompt",
        str(vllm_cfg.get("prompt", "Text Recognition:")),
        "--image-max-pixels",
        str(int(server_cfg.get("image_max_pixels", 589824))),
        "--max-new-tokens",
        str(max_tokens),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(float(server_cfg.get("gpu_memory_utilization", 0.9))),
        "--tensor-parallel-size",
        str(int(server_cfg.get("tensor_parallel_size") or max(1, len(gpu_ids)))),
    ]
    if bool(server_cfg.get("disable_custom_all_reduce", False)):
        command.append("--disable-custom-all-reduce")
    if bool(server_cfg.get("enforce_eager", False)):
        command.append("--enforce-eager")
    if bool(server_cfg.get("resume", True)):
        command.append("--resume")
    for page_id in page_ids or []:
        command.extend(["--page-id", page_id])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return command, env


def _run_glmocr_vllm_helper(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    page_ids: list[str] | None = None,
) -> None:
    command, env = _build_glmocr_vllm_helper_command(config, output_root, page_ids=page_ids)
    LOGGER.info(
        "Running glmocr_vllm helper: CUDA_VISIBLE_DEVICES=%s %s",
        env.get("CUDA_VISIBLE_DEVICES", ""),
        " ".join(command),
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if completed.stdout.strip():
        LOGGER.info("glmocr_vllm stdout:\n%s", completed.stdout.strip())
    if completed.stderr.strip():
        LOGGER.warning("glmocr_vllm stderr:\n%s", completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"glmocr_vllm helper exited with code {completed.returncode}")


def _release_layout_gpu_cache() -> None:
    clear_layout_extractor_cache()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        LOGGER.warning("Failed to release layout CUDA cache cleanly: %s", exc)


def _split_gpu_ids(gpu_ids: str | int | list[Any] | tuple[Any, ...]) -> list[str]:
    if isinstance(gpu_ids, int):
        return [str(gpu_ids)]
    if isinstance(gpu_ids, (list, tuple)):
        return [str(item).strip() for item in gpu_ids if str(item).strip()]
    return [item.strip() for item in str(gpu_ids).split(",") if item.strip()]


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _configure_managed_vllm_endpoint(config: dict[str, Any]) -> dict[str, Any]:
    vllm_cfg = config.setdefault("ocr", {}).setdefault("vllm", {})
    server_cfg = vllm_cfg.setdefault("server", {})
    host = str(server_cfg.get("host", "127.0.0.1"))
    requested_port = int(server_cfg.get("port", 8090))
    port = requested_port
    if not _port_is_available(host, requested_port):
        port = _find_free_port(host)
        LOGGER.warning(
            "Requested vLLM port %s is busy; using free port %s instead.",
            requested_port,
            port,
        )
    server_cfg["port"] = port
    vllm_cfg["api_base"] = f"http://{host}:{port}/v1"
    return server_cfg


def _build_vllm_server_command(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    vllm_cfg = config["ocr"]["vllm"]
    server_cfg = vllm_cfg.get("server", {})
    gpu_ids = _split_gpu_ids(server_cfg.get("gpu_ids", "0"))
    command = [
        str(server_cfg.get("python_bin", "python")),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(_resolve_project_path(server_cfg.get("model_path", vllm_cfg.get("model_name", "")))),
        "--served-model-name",
        str(vllm_cfg.get("model_name", "tonggu-ocr")),
        "--host",
        str(server_cfg.get("host", "127.0.0.1")),
        "--port",
        str(int(server_cfg.get("port", 8090))),
        "--tensor-parallel-size",
        str(int(server_cfg.get("tensor_parallel_size") or max(1, len(gpu_ids)))),
        "--gpu-memory-utilization",
        str(float(server_cfg.get("gpu_memory_utilization", 0.9))),
    ]
    if bool(server_cfg.get("trust_remote_code", True)):
        command.append("--trust-remote-code")
    if bool(server_cfg.get("enforce_eager", False)):
        command.append("--enforce-eager")
    if server_cfg.get("max_model_len"):
        command.extend(["--max-model-len", str(int(server_cfg["max_model_len"]))])
    for extra_arg in server_cfg.get("extra_args", []):
        command.append(str(extra_arg))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env.setdefault("DISABLE_VERSION_CHECK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return command, env


def _wait_for_vllm_ready(api_base: str, process: subprocess.Popen[str], timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    models_url = f"{api_base.rstrip('/')}/models"
    last_error = ""
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Managed vLLM server exited early with code {return_code}; last_error={last_error}")
        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                LOGGER.info("Managed vLLM server is ready at %s", api_base)
                return
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for managed vLLM server at {api_base}; last_error={last_error}")


@contextlib.contextmanager
def _managed_vllm_server(config: dict[str, Any], output_root: str | Path):
    vllm_cfg = config.get("ocr", {}).get("vllm", {})
    server_cfg = vllm_cfg.get("server", {})
    if not bool(server_cfg.get("launch", False)):
        yield
        return

    server_cfg = _configure_managed_vllm_endpoint(config)
    command, env = _build_vllm_server_command(config)
    logs_dir = Path(output_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "vllm_server.log"
    LOGGER.info(
        "Launching managed vLLM server: CUDA_VISIBLE_DEVICES=%s %s",
        env.get("CUDA_VISIBLE_DEVICES", ""),
        " ".join(command),
    )
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_vllm_ready(
            config["ocr"]["vllm"]["api_base"],
            process,
            int(server_cfg.get("startup_timeout_sec", 900)),
        )
        yield
    finally:
        if process.poll() is None:
            LOGGER.info("Stopping managed vLLM server pid=%s", process.pid)
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                LOGGER.warning("Managed vLLM server did not terminate; killing pid=%s", process.pid)
                process.kill()
                process.wait(timeout=30)
        log_handle.close()


def _finalize_glmocr_page(
    output_dirs: dict[str, Path],
    page_id: str,
) -> tuple[PipelinePageResult, dict[str, Any]]:
    page_result = _load_page_result(output_dirs, page_id)
    if page_result.status != "ok":
        raise RuntimeError(f"Page {page_id} not finalized successfully, status={page_result.status}")
    summary_entry = _summary_entry_from_page_result(page_result)
    summary_entry["page_json_path"] = str(output_dirs["pages"] / f"{page_id}.json")
    return page_result, summary_entry


def segment_page_from_lines(
    lines: list[TextLine],
    image_size: tuple[int, int],
    config: dict[str, Any],
    *,
    page_id: str,
    image_path: str,
    txt_path: str = "",
    include_debug: bool = False,
):
    blocking_config = config.get("blocking", config)
    return segment_page(
        lines,
        image_size,
        blocking_config,
        page_id=page_id,
        image_path=image_path,
        txt_path=txt_path,
        include_debug=include_debug,
    )


def _build_direct_page_stub(image_path: str | Path) -> dict[str, Any]:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        image_size = (int(img.width), int(img.height))
    return {
        "page_id": image_path.stem,
        "mode": "direct_ocr",
        "image_path": str(image_path),
        "image_size": list(image_size),
        "status": "prepared",
        "backend": "glmocr_hf",
        "line_bboxes": [],
        "line_count": 0,
        "blocks": [],
        "block_count": 0,
        "ocr_blocks": [],
        "ocr_text": "",
        "text_path": None,
        "visual_path": None,
        "crop_dir": None,
    }


def _write_direct_page_stub(
    image_path: str | Path,
    output_dirs: dict[str, Path],
) -> str:
    payload = _build_direct_page_stub(image_path)
    write_page_result(output_dirs["pages"], payload)
    return str(payload["page_id"])


def _process_one_image(
    image_path: str | Path,
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    *,
    debug: bool,
    visualize: bool,
    crop_blocks: bool,
    skip_layout: bool = False,
) -> tuple[PipelinePageResult, dict[str, Any]]:
    prepared = _prepare_one_image(
        image_path,
        config,
        output_dirs,
        debug=debug,
        visualize=visualize,
        crop_blocks=crop_blocks,
        skip_layout=skip_layout,
    )
    ocr_blocks = recognize_blocks(prepared.blocks, config)
    return _finalize_prepared_page(
        prepared,
        ocr_blocks,
        config,
        output_dirs,
        crop_blocks=crop_blocks,
    )


def _prepare_one_image(
    image_path: str | Path,
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    *,
    debug: bool,
    visualize: bool,
    crop_blocks: bool,
    skip_layout: bool = False,
) -> PreparedPage:
    image_path = Path(image_path)
    page_id = image_path.stem
    if skip_layout:
        layout_page = layout_page_result_from_dict(read_layout_result(output_dirs, page_id))
    else:
        layout_page = extract_line_bboxes(image_path, config)
        write_layout_result(output_dirs, page_id, layout_page_result_to_dict(layout_page))

    block_lines = _layout_lines_to_block_lines(layout_page)
    page_result = segment_page_from_lines(
        block_lines,
        layout_page.image_size,
        config,
        page_id=page_id,
        image_path=str(image_path),
        txt_path="",
        include_debug=debug,
    )

    visual_path = None
    if visualize:
        visual_path = draw_pipeline_visualization(
            image_path,
            block_lines,
            page_result,
            output_dirs["visuals"] / f"{page_id}.jpg",
            show_debug_metrics=debug,
        )

    effective_crop_dir = output_dirs["block_crops"] / page_id
    output_config = config["output"]
    crop_outputs = save_block_crops(
        image_path,
        page_result,
        block_lines,
        effective_crop_dir,
        padding=int(output_config["crop_padding"]),
        base_scale=float(output_config.get("crop_canvas", {}).get("base_scale", 1.25)),
        sparse_scale=float(output_config.get("crop_canvas", {}).get("sparse_scale", 1.5)),
        sparse_line_threshold=int(output_config.get("crop_canvas", {}).get("sparse_line_threshold", 2)),
        sparse_text_bbox_area_ratio=float(
            output_config.get("crop_canvas", {}).get("sparse_text_bbox_area_ratio", 0.55)
        ),
        crop_mode=str(output_config.get("crop_canvas", {}).get("mode", "padded_square")),
        enable_foreign_mask=bool(output_config.get("crop_mask", {}).get("enabled", True)),
    )
    block_entries = _build_block_entries(page_result, crop_outputs)
    final_crop_dir = str(effective_crop_dir) if crop_blocks else None
    LOGGER.info(
        "Prepared %s -> %s lines, %s blocks, OCR backend=%s",
        page_id,
        len(layout_page.lines),
        page_result.block_count,
        config["ocr"]["backend"],
    )
    return PreparedPage(
        page_id=page_id,
        mode="reuse-layout" if skip_layout else "full",
        image_path=str(image_path),
        image_size=layout_page.image_size,
        line_bboxes=_layout_lines_to_dicts(layout_page),
        line_count=len(layout_page.lines),
        blocks=block_entries,
        block_count=page_result.block_count,
        visual_path=str(visual_path) if visual_path else None,
        crop_dir=final_crop_dir,
        crop_count=len(crop_outputs),
        debug_payload=page_result.debug if debug else {},
    )


def _finalize_prepared_page(
    prepared: PreparedPage,
    ocr_blocks: list[OcrBlockResult],
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    *,
    crop_blocks: bool,
) -> tuple[PipelinePageResult, dict[str, Any]]:
    page_id = prepared.page_id
    effective_crop_dir = Path(prepared.crop_dir) if prepared.crop_dir else output_dirs["block_crops"] / page_id
    block_entries = prepared.blocks
    ocr_blocks = _restore_ocr_block_line_breaks(ocr_blocks, block_entries)
    ocr_text = _compose_ocr_text(ocr_blocks)
    text_path = write_text_result(output_dirs["texts"], page_id, ocr_text)

    final_crop_dir = prepared.crop_dir
    if not crop_blocks:
        crop_paths = [item["crop_path"] for item in block_entries if item.get("crop_path")]
        for crop_path in crop_paths:
            path = Path(crop_path)
            if path.exists():
                path.unlink()
        if effective_crop_dir.exists():
            effective_crop_dir.rmdir()
        block_entries = [
            {
                **item,
                "crop_path": None,
                "masked_crop_path": None,
                "background_fill_color": None,
                "crop_canvas_size": None,
                "sparse_canvas_applied": False,
            }
            for item in block_entries
        ]
        ocr_blocks = [
            OcrBlockResult(
                block_id=item.block_id,
                crop_path="",
                masked_crop_path=None,
                ocr_text=item.ocr_text,
                backend=item.backend,
                latency_ms=item.latency_ms,
            )
            for item in ocr_blocks
        ]
        final_crop_dir = None

    pipeline_result = PipelinePageResult(
        page_id=page_id,
        mode=prepared.mode,
        image_path=prepared.image_path,
        image_size=prepared.image_size,
        status="ok",
        backend=config["ocr"]["backend"],
        line_bboxes=prepared.line_bboxes,
        line_count=prepared.line_count,
        blocks=block_entries,
        block_count=prepared.block_count,
        ocr_blocks=ocr_blocks,
        ocr_text=ocr_text,
        text_path=str(text_path),
        visual_path=prepared.visual_path,
        crop_dir=final_crop_dir,
        debug=prepared.debug_payload,
    )
    page_json_path = write_page_result(output_dirs["pages"], pipeline_result.to_dict(include_debug=bool(prepared.debug_payload)))

    LOGGER.info(
        "Processed %s -> %s lines, %s blocks, OCR backend=%s",
        prepared.page_id,
        prepared.line_count,
        prepared.block_count,
        config["ocr"]["backend"],
    )
    return pipeline_result, {
        "status": "ok",
        "page_id": prepared.page_id,
        "image_path": prepared.image_path,
        "line_count": prepared.line_count,
        "block_count": prepared.block_count,
        "backend": config["ocr"]["backend"],
        "page_json_path": str(page_json_path),
        "text_path": str(text_path),
        "visual_path": prepared.visual_path,
        "crop_dir": final_crop_dir,
        "crop_count": prepared.crop_count if crop_blocks else 0,
    }


def _recognize_prepared_page(prepared: PreparedPage, config: dict[str, Any]) -> list[OcrBlockResult]:
    return recognize_blocks(prepared.blocks, config)


def _recognize_prepared_pages_vllm(
    prepared_pages: list[PreparedPage],
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    *,
    crop_blocks: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []
    ocr_cfg = config.get("ocr", {}).get("vllm", {})
    max_inflight_pages = max(1, int(ocr_cfg.get("max_page_concurrency", 4)))
    with _managed_vllm_server(config, output_dirs["root"]):
        with ThreadPoolExecutor(max_workers=max_inflight_pages) as executor:
            future_map = {
                executor.submit(_recognize_prepared_page, prepared, config): prepared
                for prepared in prepared_pages
            }
            for future in as_completed(future_map):
                prepared = future_map[future]
                try:
                    pages.append(
                        _finalize_prepared_page(
                            prepared,
                            future.result(),
                            config,
                            output_dirs,
                            crop_blocks=crop_blocks,
                        )[1]
                    )
                except Exception as exc:  # pragma: no cover - defensive batch safety
                    LOGGER.exception("Failed to run vLLM OCR/finalize %s", prepared.image_path)
                    failed_pages.append(
                        {
                            "page_id": prepared.page_id,
                            "image_path": prepared.image_path,
                            "error": str(exc),
                        }
                    )
    return pages, failed_pages


def _resolve_mode(mode: str, skip_layout: bool) -> str:
    if mode == "full" and skip_layout:
        return "reuse-layout"
    return mode


def _run_direct_ocr_on_image(
    image_path: str | Path,
    output_dirs: dict[str, Path],
    config: dict[str, Any],
) -> PipelinePageResult:
    page_id = _write_direct_page_stub(image_path, output_dirs)
    _run_glmocr_direct_helper(config, output_dirs["root"], page_ids=[page_id])
    page_result, _ = _finalize_glmocr_page(output_dirs, page_id)
    return page_result


def run_pipeline_on_image(
    image_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    mode: str = "full",
    debug: bool = False,
    visualize: bool | None = None,
    crop_blocks: bool | None = None,
    skip_layout: bool = False,
) -> PipelinePageResult:
    output_dirs = ensure_output_dirs(output_dir)
    mode = _resolve_mode(mode, skip_layout)
    visualize = config["output"]["visualize"] if visualize is None else visualize
    crop_blocks = config["output"]["crop_blocks"] if crop_blocks is None else crop_blocks
    if mode == "direct-ocr":
        if not _is_glmocr_hf_backend(config):
            raise ValueError("direct-ocr mode currently requires --ocr-backend glmocr_hf")
        return _run_direct_ocr_on_image(image_path, output_dirs, config)
    if _is_glmocr_hf_backend(config):
        prepared = _prepare_one_image(
            image_path,
            config,
            output_dirs,
            debug=debug,
            visualize=visualize,
            crop_blocks=crop_blocks,
            skip_layout=(mode == "reuse-layout"),
        )
        _write_prepared_page_stub(prepared, config, output_dirs)
        _run_glmocr_helper(config, output_dirs["root"], page_ids=[prepared.page_id])
        page_result, _ = _finalize_glmocr_page(output_dirs, prepared.page_id)
        return page_result
    if _is_glmocr_vllm_backend(config):
        prepared = _prepare_one_image(
            image_path,
            config,
            output_dirs,
            debug=debug,
            visualize=visualize,
            crop_blocks=crop_blocks,
            skip_layout=(mode == "reuse-layout"),
        )
        _write_prepared_page_stub(prepared, config, output_dirs)
        _release_layout_gpu_cache()
        _run_glmocr_vllm_helper(config, output_dirs["root"], page_ids=[prepared.page_id])
        page_result, _ = _finalize_glmocr_page(output_dirs, prepared.page_id)
        return page_result
    if config.get("ocr", {}).get("backend") == "vllm":
        prepared = _prepare_one_image(
            image_path,
            config,
            output_dirs,
            debug=debug,
            visualize=visualize,
            crop_blocks=crop_blocks,
            skip_layout=(mode == "reuse-layout"),
        )
        _write_prepared_page_stub(prepared, config, output_dirs)
        with _managed_vllm_server(config, output_dirs["root"]):
            ocr_blocks = _recognize_prepared_page(prepared, config)
        page_result, _ = _finalize_prepared_page(
            prepared,
            ocr_blocks,
            config,
            output_dirs,
            crop_blocks=crop_blocks,
        )
        return page_result

    page_result, _ = _process_one_image(
        image_path,
        config,
        output_dirs,
        debug=debug,
        visualize=visualize,
        crop_blocks=crop_blocks,
        skip_layout=(mode == "reuse-layout"),
    )
    return page_result


def run_pipeline_on_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    mode: str = "full",
    debug: bool = False,
    visualize: bool | None = None,
    crop_blocks: bool | None = None,
    workers: int = 1,
    limit: int | None = None,
    skip_layout: bool = False,
    config_path: str | None = None,
) -> PipelineRunSummary:
    output_dirs = ensure_output_dirs(output_dir)
    image_paths = iter_input_images(input_dir=input_dir, limit=limit)
    mode = _resolve_mode(mode, skip_layout)
    visualize = config["output"]["visualize"] if visualize is None else visualize
    crop_blocks = config["output"]["crop_blocks"] if crop_blocks is None else crop_blocks

    pages: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []
    ocr_backend = config.get("ocr", {}).get("backend")
    ocr_cfg = config.get("ocr", {}).get("vllm", {})
    if mode == "direct-ocr":
        if ocr_backend != "glmocr_hf":
            raise ValueError("direct-ocr mode currently requires --ocr-backend glmocr_hf")
        direct_pages: list[tuple[str, str]] = []
        for image_path in image_paths:
            try:
                direct_pages.append((_write_direct_page_stub(image_path, output_dirs), str(image_path)))
            except Exception as exc:  # pragma: no cover - defensive batch safety
                LOGGER.exception("Failed to prepare direct OCR stub for %s", image_path)
                failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})
        helper_error: str | None = None
        if direct_pages:
            try:
                _run_glmocr_direct_helper(config, output_dirs["root"], page_ids=[page_id for page_id, _ in direct_pages])
            except Exception as exc:  # pragma: no cover - helper is external
                LOGGER.exception("glmocr_hf direct helper failed")
                helper_error = str(exc)
        for page_id, image_path in direct_pages:
            try:
                pages.append(_finalize_glmocr_page(output_dirs, page_id)[1])
            except Exception as exc:  # pragma: no cover - defensive batch safety
                LOGGER.exception("Failed to finalize direct OCR page %s", image_path)
                error_message = str(exc)
                if helper_error:
                    error_message = f"{error_message}; helper_error={helper_error}"
                failed_pages.append({"page_id": page_id, "image_path": str(image_path), "error": error_message})
    elif ocr_backend in {"glmocr_hf", "glmocr_vllm"}:
        prepared_pages: list[PreparedPage] = []
        if workers <= 1:
            for image_path in image_paths:
                try:
                    prepared = _prepare_one_image(
                        image_path,
                        config,
                        output_dirs,
                        debug=debug,
                        visualize=visualize,
                        crop_blocks=crop_blocks,
                        skip_layout=(mode == "reuse-layout"),
                    )
                    _write_prepared_page_stub(prepared, config, output_dirs)
                    prepared_pages.append(prepared)
                except Exception as exc:  # pragma: no cover - defensive batch safety
                    LOGGER.exception("Failed to prepare %s", image_path)
                    failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        _prepare_one_image,
                        image_path,
                        config,
                        output_dirs,
                        debug=debug,
                        visualize=visualize,
                        crop_blocks=crop_blocks,
                        skip_layout=(mode == "reuse-layout"),
                    ): image_path
                    for image_path in image_paths
                }
                for future in as_completed(future_map):
                    image_path = future_map[future]
                    try:
                        prepared = future.result()
                        _write_prepared_page_stub(prepared, config, output_dirs)
                        prepared_pages.append(prepared)
                    except Exception as exc:  # pragma: no cover - defensive batch safety
                        LOGGER.exception("Failed to prepare %s", image_path)
                        failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})

        helper_error: str | None = None
        if prepared_pages:
            try:
                if ocr_backend == "glmocr_vllm":
                    _release_layout_gpu_cache()
                    _run_glmocr_vllm_helper(
                        config,
                        output_dirs["root"],
                        page_ids=[prepared.page_id for prepared in prepared_pages],
                    )
                else:
                    _run_glmocr_helper(
                        config,
                        output_dirs["root"],
                        page_ids=[prepared.page_id for prepared in prepared_pages],
                    )
            except Exception as exc:  # pragma: no cover - helper is external
                LOGGER.exception("%s helper failed", ocr_backend)
                helper_error = str(exc)

        for prepared in prepared_pages:
            try:
                pages.append(_finalize_glmocr_page(output_dirs, prepared.page_id)[1])
            except Exception as exc:  # pragma: no cover - defensive batch safety
                LOGGER.exception("Failed to finalize %s", prepared.image_path)
                error_message = str(exc)
                if helper_error:
                    error_message = f"{error_message}; helper_error={helper_error}"
                failed_pages.append(
                    {
                        "page_id": prepared.page_id,
                        "image_path": prepared.image_path,
                        "error": error_message,
                    }
                )
    elif ocr_backend == "vllm":
        prepared_pages: list[PreparedPage] = []
        if workers <= 1:
            for image_path in image_paths:
                try:
                    prepared = _prepare_one_image(
                        image_path,
                        config,
                        output_dirs,
                        debug=debug,
                        visualize=visualize,
                        crop_blocks=crop_blocks,
                        skip_layout=(mode == "reuse-layout"),
                    )
                    _write_prepared_page_stub(prepared, config, output_dirs)
                    prepared_pages.append(prepared)
                except Exception as exc:  # pragma: no cover - defensive batch safety
                    LOGGER.exception("Failed to prepare %s", image_path)
                    failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        _prepare_one_image,
                        image_path,
                        config,
                        output_dirs,
                        debug=debug,
                        visualize=visualize,
                        crop_blocks=crop_blocks,
                        skip_layout=(mode == "reuse-layout"),
                    ): image_path
                    for image_path in image_paths
                }
                for future in as_completed(future_map):
                    image_path = future_map[future]
                    try:
                        prepared = future.result()
                        _write_prepared_page_stub(prepared, config, output_dirs)
                        prepared_pages.append(prepared)
                    except Exception as exc:  # pragma: no cover - defensive batch safety
                        LOGGER.exception("Failed to prepare %s", image_path)
                        failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})

        if prepared_pages:
            vllm_pages, vllm_failed_pages = _recognize_prepared_pages_vllm(
                prepared_pages,
                config,
                output_dirs,
                crop_blocks=crop_blocks,
            )
            pages.extend(vllm_pages)
            failed_pages.extend(vllm_failed_pages)
    elif workers <= 1:
        for image_path in image_paths:
            try:
                pages.append(
                    _process_one_image(
                        image_path,
                        config,
                        output_dirs,
                        debug=debug,
                        visualize=visualize,
                        crop_blocks=crop_blocks,
                        skip_layout=(mode == "reuse-layout"),
                    )[1]
                )
            except Exception as exc:  # pragma: no cover - defensive batch safety
                LOGGER.exception("Failed to process %s", image_path)
                failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _process_one_image,
                    image_path,
                    config,
                    output_dirs,
                    debug=debug,
                    visualize=visualize,
                    crop_blocks=crop_blocks,
                    skip_layout=(mode == "reuse-layout"),
                ): image_path
                for image_path in image_paths
            }
            for future in as_completed(future_map):
                image_path = future_map[future]
                try:
                    _, summary_entry = future.result()
                    pages.append(summary_entry)
                except Exception as exc:  # pragma: no cover - defensive batch safety
                    LOGGER.exception("Failed to process %s", image_path)
                    failed_pages.append({"page_id": Path(image_path).stem, "image_path": str(image_path), "error": str(exc)})

    summary = PipelineRunSummary(
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        config_path=config_path,
        page_count=len(pages),
        failed_page_count=len(failed_pages),
        pages=sorted(pages, key=lambda item: item["page_id"]),
        failed_pages=sorted(failed_pages, key=lambda item: item["page_id"]),
    )
    write_summary(output_dirs["root"], summary.to_dict())
    return summary


def run_from_args(args) -> tuple[dict[str, Any], dict[str, Any] | None]:
    output_dirs = ensure_output_dirs(args.output_dir)
    setup_logging(output_dirs["logs"] / "run.log")
    config = load_config(args.config)
    if args.ocr_backend:
        config["ocr"]["backend"] = args.ocr_backend
    if getattr(args, "layout_device", None):
        _apply_layout_device_override(config, args.layout_device)
    if args.vllm_api_base:
        config["ocr"]["vllm"]["api_base"] = args.vllm_api_base
    if args.vllm_model_name:
        config["ocr"]["vllm"]["model_name"] = args.vllm_model_name
    if getattr(args, "vllm_launch_server", False):
        _apply_vllm_server_override(config, "launch", True)
    if getattr(args, "vllm_gpu_ids", None):
        _apply_vllm_server_override(config, "gpu_ids", args.vllm_gpu_ids)
    if getattr(args, "vllm_port", None):
        _apply_vllm_server_override(config, "port", args.vllm_port)
    if getattr(args, "vllm_python_bin", None):
        _apply_vllm_server_override(config, "python_bin", args.vllm_python_bin)
    if getattr(args, "vllm_model_path", None):
        _apply_vllm_server_override(config, "model_path", args.vllm_model_path)
    if getattr(args, "vllm_max_tokens", None):
        config["ocr"]["vllm"]["max_tokens"] = args.vllm_max_tokens
    if getattr(args, "vllm_max_model_len", None):
        _apply_vllm_server_override(config, "local_max_model_len", args.vllm_max_model_len)
        _apply_vllm_server_override(config, "max_model_len", args.vllm_max_model_len)
    if getattr(args, "vllm_gpu_memory_utilization", None):
        _apply_vllm_server_override(config, "gpu_memory_utilization", args.vllm_gpu_memory_utilization)
    if getattr(args, "glmocr_python_bin", None):
        _apply_glmocr_override(config, "python_bin", args.glmocr_python_bin)
    if getattr(args, "glmocr_model_path", None):
        _apply_glmocr_override(config, "model_path", args.glmocr_model_path)
    if getattr(args, "glmocr_gpu_ids", None):
        _apply_glmocr_override(config, "gpu_ids", args.glmocr_gpu_ids)
    if getattr(args, "glmocr_concurrency", None):
        _apply_glmocr_override(config, "concurrency", args.glmocr_concurrency)
    if getattr(args, "crop_mode", None):
        config.setdefault("output", {}).setdefault("crop_canvas", {})["mode"] = args.crop_mode
    if getattr(args, "disable_foreign_mask", False):
        config.setdefault("output", {}).setdefault("crop_mask", {})["enabled"] = False

    visualize = config["output"]["visualize"] if args.visualize is None else args.visualize
    crop_blocks = config["output"]["crop_blocks"] if args.crop_blocks is None else args.crop_blocks
    mode = _resolve_mode(args.mode, args.skip_layout)
    LOGGER.info(
        "Using layout device: requested=%s effective=%s CUDA_VISIBLE_DEVICES=%s",
        config.get("_runtime", {}).get("layout_physical_device", config.get("layout_inference", {}).get("device")),
        config.get("layout_inference", {}).get("device"),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    if args.image:
        page_result = run_pipeline_on_image(
            args.image,
            args.output_dir,
            config,
            mode=mode,
            debug=args.debug,
            visualize=visualize,
            crop_blocks=crop_blocks,
            skip_layout=args.skip_layout,
        )
        page_summary = {
            "status": page_result.status,
            "page_id": page_result.page_id,
            "mode": page_result.mode,
            "image_path": page_result.image_path,
            "line_count": page_result.line_count,
            "block_count": page_result.block_count,
            "backend": page_result.backend,
            "page_json_path": str(output_dirs["pages"] / f"{page_result.page_id}.json"),
            "text_path": page_result.text_path,
            "visual_path": page_result.visual_path,
            "crop_dir": page_result.crop_dir,
            "crop_count": len(page_result.blocks) if page_result.crop_dir else 0,
        }
        summary = PipelineRunSummary(
            generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            config_path=config_path,
            page_count=1,
            failed_page_count=0,
            pages=[page_summary],
            failed_pages=[],
        )
        summary_dict = summary.to_dict()
        write_summary(output_dirs["root"], summary_dict)
        return summary_dict, _build_page_preview(page_result)

    summary = run_pipeline_on_dir(
        args.input_dir,
        args.output_dir,
        config,
        mode=mode,
        debug=args.debug,
        visualize=visualize,
        crop_blocks=crop_blocks,
        workers=args.workers,
        limit=args.limit,
        skip_layout=args.skip_layout,
        config_path=str(args.config) if args.config else None,
    )
    summary_dict = summary.to_dict()
    write_summary(output_dirs["root"], summary_dict)
    return summary_dict, None
