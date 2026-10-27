# Third-Party Notices

This repository is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
This file collects attribution and licensing information for third-party
components used by or included in this repository.

## OpenMMLab stack (Apache-2.0)

The layout-analysis stage of `inference/tonggu_pipe/layout/` is built on the
OpenMMLab ecosystem:

- MMCV (`mmcv-full 1.7.2`) — https://github.com/open-mmlab/mmcv — Apache-2.0
- MMDetection (`mmdet 2.25.2`) — https://github.com/open-mmlab/mmdetection — Apache-2.0
- MMRotate (`mmrotate 0.3.2`) — https://github.com/open-mmlab/mmrotate — Apache-2.0

The prebuilt `mmcv-full` wheel shipped under `inference/wheels/` is provided
for convenience (Linux x86_64, CPython 3.10, sm_86); on other configurations
`inference/setup_env.sh` builds it from source. `inference/scripts/patch_openmmlab.py`
applies small compatibility patches for newer PyTorch versions.

## OmniDocBench (Apache-2.0), vendored

`evaluation/third_party/OmniDocBench/` contains an unmodified subset (source
tree + license + packaging metadata) of OmniDocBench:

- Upstream: https://github.com/opendatalab/OmniDocBench — Apache-2.0
- Vendored revision: ``be7a44d371e43801fabd2e31e02976f106b6aa16` (2026-04-10, "sync: align codebase with local v1.6")` (pinned by matching
  `src/dataset/end2end_dataset.py` against upstream history)
- Only the metric engine surface used by the RO-ED evaluation is vendored;
  upstream demo data, assets, and result directories are omitted.

The reading-order evaluation (RO-ED) reuses OmniDocBench's text matching and
edit-distance machinery; see `evaluation/ro_ed/README.md` for the protocol and
`evaluation/README.md` for how the vendored tree is invoked.

## Evaluation metric scripts

`evaluation/transform.py` and `evaluation/calculate_metrics.py` are in-house
scripts that follow the AR/CR edit-distance evaluation convention widely used
in MLLM-OCR repositories (e.g. GOT-OCR, DeepSeek-OCR). No third-party source
code is copied verbatim.

## Model weights and datasets

- Model weights are **not** distributed in this repository. They are
  distributed separately via Hugging Face; the license for the weights is
  stated on the model repository.
- The M5HisDoc and MTHv2 evaluation datasets are **not** redistributed.
  Users must obtain the datasets from their respective owners
  (see `evaluation/README.md` for the expected directory layout).
