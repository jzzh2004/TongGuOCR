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

模型权重通过独立模型仓库发布，以其页面所列许可为准。
M5HisDoc 和 MTHv2 数据集不随代码仓库分发，使用者应向数据集所有者获取。
