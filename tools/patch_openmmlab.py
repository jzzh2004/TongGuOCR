#!/usr/bin/env python3
from __future__ import annotations

import site
from pathlib import Path


PATCHES = {
    "mmdet/__init__.py": (
        "mmcv_maximum_version = '1.7.0'",
        "mmcv_maximum_version = '1.7.2'",
    ),
    "mmrotate/__init__.py": (
        "mmcv_maximum_version = '1.6.0'",
        "mmcv_maximum_version = '1.7.2'",
    ),
    "mmrotate/core/post_processing/bbox_nms_rotated.py": (
        "labels = torch.arange(num_classes, dtype=torch.long)",
        "labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)",
    ),
    # numpy>=2 removed the np.int0 alias (equal to np.intp).
    "mmrotate/core/visualization/image.py": (
        "np.int0(np.array([p1, p2, p3, p4]))",
        "np.intp(np.array([p1, p2, p3, p4]))",
    ),
}


def _site_packages() -> Path:
    paths = [Path(path) for path in site.getsitepackages()]
    for path in paths:
        if path.exists():
            return path
    raise RuntimeError(f"No existing site-packages directory found: {paths}")


def _replace(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")
        return True
    if new in text:
        return False
    raise RuntimeError(f"Expected patch site not found in {path}")


def main() -> int:
    site_packages = _site_packages()
    changed: list[str] = []
    unchanged: list[str] = []

    for rel_path, (old, new) in PATCHES.items():
        path = site_packages / rel_path
        if _replace(path, old, new):
            changed.append(rel_path)
        else:
            unchanged.append(rel_path)

    nms_path = site_packages / "mmrotate/core/post_processing/bbox_nms_rotated.py"
    old = "    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)\n\n    if max_num > 0:"
    new = "    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)\n    keep = keep.to(bboxes.device)\n\n    if max_num > 0:"
    if _replace(nms_path, old, new):
        changed.append("mmrotate/core/post_processing/bbox_nms_rotated.py::keep_device")
    else:
        unchanged.append("mmrotate/core/post_processing/bbox_nms_rotated.py::keep_device")

    print("OpenMMLab compatibility patch complete.")
    print(f"site-packages: {site_packages}")
    print(f"changed: {changed}")
    print(f"already patched: {unchanged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
