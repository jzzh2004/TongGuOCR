#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def _module_version(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:  # pragma: no cover - environment diagnostics
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the tongguocr inference environment.")
    parser.add_argument("--json-out", help="Optional path for a JSON report.")
    parser.add_argument(
        "--require-transformers",
        default="5.2.0,5.3.0",
        help="Comma-separated acceptable transformers versions for GLMOCR.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "modules": {},
        "checks": {},
    }
    for name in [
        "torch",
        "torchvision",
        "transformers",
        "mmcv",
        "mmdet",
        "mmrotate",
        "numpy",
        "PIL",
        "tqdm",
        "safetensors",
        "accelerate",
    ]:
        info = _module_version(name)
        report["modules"][name] = info

    failures: list[str] = []
    for name, info in report["modules"].items():
        if not info.get("ok"):
            failures.append(f"{name} import failed: {info.get('error')}")

    transformers_info = report["modules"].get("transformers", {})
    allowed_transformers = [v.strip() for v in args.require_transformers.split(",") if v.strip()]
    if transformers_info.get("version") not in allowed_transformers:
        failures.append(
            f"transformers version must be one of {allowed_transformers}, "
            f"got {transformers_info.get('version')}"
        )

    try:
        import torch

        report["checks"]["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["checks"]["torch_cuda_version"] = torch.version.cuda
        report["checks"]["torch_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            report["checks"]["torch_device_name_0"] = torch.cuda.get_device_name(0)
        else:
            failures.append("torch.cuda.is_available() is false")
    except Exception as exc:  # pragma: no cover - environment diagnostics
        failures.append(f"torch CUDA check failed: {type(exc).__name__}: {exc}")

    try:
        import mmcv._ext as mmcv_ext
        from mmcv.ops import nms

        report["checks"]["mmcv_ext_file"] = getattr(mmcv_ext, "__file__", None)
        report["checks"]["mmcv_nms_import"] = bool(nms)
    except Exception as exc:  # pragma: no cover - environment diagnostics
        failures.append(f"mmcv ops check failed: {type(exc).__name__}: {exc}")

    try:
        from transformers import AutoProcessor
        from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrForConditionalGeneration

        report["checks"]["glmocr_imports"] = bool(AutoProcessor and GlmOcrForConditionalGeneration)
    except Exception as exc:  # pragma: no cover - environment diagnostics
        failures.append(f"GLMOCR import check failed: {type(exc).__name__}: {exc}")

    try:
        from vllm.model_executor.models.registry import ModelRegistry

        has_glmocr = "GlmOcrForConditionalGeneration" in ModelRegistry.get_supported_archs()
        report["checks"]["vllm_glmocr_supported"] = bool(has_glmocr)
        if not has_glmocr:
            failures.append("vLLM does not support GlmOcrForConditionalGeneration in this env")
    except ImportError:
        # vLLM serving lives in its own env (tongguocr-vllm2, see setup_vllm_env.sh);
        # this env only needs layout + transformers OCR.
        report["checks"]["vllm_glmocr_supported"] = "skipped (vllm not installed in this env; serving env is separate)"
    except Exception as exc:  # pragma: no cover - environment diagnostics
        failures.append(f"vLLM check failed: {type(exc).__name__}: {exc}")

    report["failures"] = failures
    report["ok"] = not failures
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
