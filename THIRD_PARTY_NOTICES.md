# Third-Party Notices

[简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

This repository is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
This file collects attribution and licensing information for third-party
components used by or included in this repository.

## OpenMMLab stack (Apache-2.0)

The layout-analysis stage of `tongguocr/layout/` is built on the
OpenMMLab ecosystem:

- MMCV (`mmcv-full 1.7.2`) — https://github.com/open-mmlab/mmcv — Apache-2.0
- MMDetection (`mmdet 2.25.2`) — https://github.com/open-mmlab/mmdetection — Apache-2.0
- MMRotate (`mmrotate 0.3.2`) — https://github.com/open-mmlab/mmrotate — Apache-2.0

No binary wheel is shipped with the repository: `tools/setup_env.sh`
downloads or builds `mmcv-full` from source as appropriate for your platform.
`tools/patch_openmmlab.py` applies small compatibility patches for newer
PyTorch versions.

## Detection code adaptations

- `tongguocr/layout/hdc/modeling/backbone/hornet.py` adapts HorNet's backbone
  code, including its recursive gated convolution and global/local filter.
  Upstream: https://github.com/raoyongming/HorNet. The original code retains
  MIT terms and Copyright (c) 2021 Yongming Rao. See
  [the retained HorNet license](THIRD_PARTY_LICENSES/HorNet-MIT.md).
- `tongguocr/layout/hdc/utils/comm.py` retains communication utilities from
  Facebook's maskrcnn-benchmark initial release, with an added `get_rank`
  helper and formatting changes. Its original Facebook copyright notice
  remains in the file. See
  [the retained MIT license](THIRD_PARTY_LICENSES/maskrcnn-benchmark-MIT.md).
- The local mask-pasting implementation in
  `tongguocr/layout/hdc/modeling/roi_heads/mask_head/fcn_mask_head.py`
  records its Detectron2 origin. Upstream:
  https://github.com/facebookresearch/detectron2, Apache-2.0.
- The local midpoint-offset bbox coder records its OpenMMLab and
  OBBDetection origin. Upstream: https://github.com/jbwang1997/OBBDetection,
  Apache-2.0; Copyright 2018-2019 Open-MMLab. The attribution remains in
  `tongguocr/layout/hdc/modeling/rpn/bbox_coder.py`.

These code notices are separate from the model-weight license declarations.
The Apache-2.0 text is included in [LICENSE](LICENSE).

## RO-ED text matcher (Apache-2.0), adapted from OmniDocBench

`evaluation/ro_ed/{matching,normalize,reading_order}.py` retain and adapt the
text-line matching, normalization, timeout fallback, and order-pairing logic
from OmniDocBench. The general dataset/metric framework, formula/table
scorers, registries, Markdown parser, and other vendored modules are removed.

- Upstream: https://github.com/opendatalab/OmniDocBench — Apache-2.0
- Source revision: `be7a44d371e43801fabd2e31e02976f106b6aa16`
- Copyright: OpenDataLab; original Apache-2.0 license retained at
  `evaluation/ro_ed/LICENSE`.
- TongGuOCR provides the standalone driver, token-boundary protocol,
  diagnostics, integration with `calculate_metrics.py`, and tests.

The extraction was checked against the archived M5 (1000 pages) and MTHv2
(800 pages) runs: every page's GT order sequence, predicted order sequence,
and edit distance is identical. The full OmniDocBench tree is not required
at installation or runtime.

## Evaluation metric scripts

`evaluation/calculate_metrics.py` is an in-house script that follows the
AR/CR edit-distance evaluation convention widely used
in MLLM-OCR repositories (e.g. GOT-OCR, DeepSeek-OCR). No third-party source
code is copied verbatim.

## Model weights and datasets

- Model weights are **not** distributed in this code repository. They are
  hosted separately on [Hugging Face](https://huggingface.co/jzzh2004/TongGuOCR)
  and [ModelScope International](https://modelscope.ai/models/jzzh2004/TongGuOCR).
- TongGuOCR's own OCR adaptations, detection checkpoint and documentation
  are distributed under Apache-2.0, as selected by the model owner. This
  grant covers rights controlled by the TongGuOCR contributors and does
  not replace applicable third-party terms. The
  GLM-OCR base-model weights retain MIT terms. See
  [GLM-OCR model license and provenance](THIRD_PARTY_LICENSES/GLM-OCR-MIT.md).
  The upstream SDK's Apache-2.0 license is separate from the model license.
- `layout/finetune_infer.pth` is the HDC text-line detection checkpoint,
  distributed under [Apache-2.0](LICENSE) as selected by the model owner.
  Its license is declared separately from the GLM-OCR base-model license.
- The M5HisDoc and MTHv2 evaluation datasets are **not** redistributed.
  Users must obtain the datasets from their respective owners
  (see `evaluation/README.md` for the expected directory layout).
