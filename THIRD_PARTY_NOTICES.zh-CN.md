# 第三方声明

[English](THIRD_PARTY_NOTICES.md)

本仓库代码使用 [Apache-2.0](LICENSE)。下列组件保留其原有许可与来源说明。

## OpenMMLab（Apache-2.0）

`tongguocr/layout/` 使用以下组件：

- [MMCV](https://github.com/open-mmlab/mmcv)，`mmcv-full 1.7.2`。
- [MMDetection](https://github.com/open-mmlab/mmdetection)，`mmdet 2.25.2`。
- [MMRotate](https://github.com/open-mmlab/mmrotate)，`mmrotate 0.3.2`。

仓库不分发二进制轮子；安装脚本按平台下载或编译。
`tools/patch_openmmlab.py` 为较新 PyTorch 应用兼容补丁。

## 检测代码改编

- `tongguocr/layout/hdc/modeling/backbone/hornet.py` 改编了
  [HorNet](https://github.com/raoyongming/HorNet) 的骨干网络、递归门控卷积与
  全局／局部滤波器。原代码保留 MIT 条款及 Copyright (c) 2021 Yongming Rao，
  见[保留的许可全文](THIRD_PARTY_LICENSES/HorNet-MIT.md)。
- `tongguocr/layout/hdc/utils/comm.py` 保留了 Facebook 的 maskrcnn-benchmark
  初始版本通信工具，增加了 `get_rank` 并调整格式。文件内保留原 Facebook
  版权声明，见[保留的 MIT 许可](THIRD_PARTY_LICENSES/maskrcnn-benchmark-MIT.md)。
- `tongguocr/layout/hdc/modeling/roi_heads/mask_head/fcn_mask_head.py`
  的掩码粘贴实现注明改编自
  [Detectron2](https://github.com/facebookresearch/detectron2)，使用 Apache-2.0。
- `tongguocr/layout/hdc/modeling/rpn/bbox_coder.py` 保留了 OpenMMLab 与
  [OBBDetection](https://github.com/jbwang1997/OBBDetection) 来源说明。
  上游使用 Apache-2.0，版权声明为 Copyright 2018-2019 Open-MMLab。

这些声明针对代码，与模型权重的许可声明分别适用。
Apache-2.0 全文见 [LICENSE](LICENSE)。

## RO-ED 文本匹配（Apache-2.0）

`evaluation/ro_ed/{matching,normalize,reading_order}.py` 改编了文本行匹配、归一化、
超时回退和顺序配对逻辑。原通用评测框架未包含在本仓库中。

- 来源：[OpenDataLab/OmniDocBench](https://github.com/opendatalab/OmniDocBench)。
- 源版本：`be7a44d371e43801fabd2e31e02976f106b6aa16`。
- 版权所有：OpenDataLab；原 Apache-2.0 许可保留在 `evaluation/ro_ed/LICENSE`。
- TongGuOCR 提供本地评测入口、词元边界协议、诊断、统一指标集成和测试。

归档 M5 1000 页与 MTHv2 800 页的 GT／预测顺序及每页编辑距离均已核对一致。

## 其他评测脚本

`evaluation/calculate_metrics.py` 是本项目实现的脚本，
遵循 OCR 中通用的字符编辑距离评测约定。

## 权重与数据集

模型权重单独托管于 [Hugging Face](https://huggingface.co/jzzh2004/TongGuOCR) 和
[ModelScope 国际站](https://modelscope.ai/models/jzzh2004/TongGuOCR)。

按模型所有者选定的许可，TongGuOCR 自有 OCR 改动、检测权重与文档以 Apache-2.0
发布；授权范围限于 TongGuOCR 贡献者有权授予的权利，不替代适用的第三方条款。
GLM-OCR 基座权重保留 MIT 条款，
详见 [GLM-OCR 模型许可与来源](THIRD_PARTY_LICENSES/GLM-OCR-MIT.md)。
GLM-OCR SDK 代码的 Apache-2.0 与基座模型的 MIT 是不同许可范围。

`layout/finetune_infer.pth` 是 HDC 文本行检测权重，按模型所有者选定的
[Apache-2.0](LICENSE) 发布。该许可声明独立于 GLM-OCR 基座模型的许可。

M5HisDoc 和 MTHv2 数据集不随代码仓库分发，使用者应向数据集所有者获取。
