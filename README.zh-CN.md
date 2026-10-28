<div align="center">
  <img src="website/public/tonggu-logo.png" alt="TongGuOCR 标志" width="120" />

  # TongGuOCR

  **面向中文历史文献的布局感知与词元增强 OCR 多模态大语言模型**

  [论文](https://arxiv.org/abs/2608.07917) ·
  [项目主页](https://jzzh2004.github.io/TongGuOCR) ·
  [在线演示](https://guji-ocr.com)

  [English](README.md) · 简体中文
</div>

## 新闻

- **[2026-10-28]** 推理代码与评测套件开源 —— `tongguocr` 包（版面 → 分块 → 识别），AR/CR/NED + 阅读顺序（RO-ED）评测。
- 模型权重正在 Hugging Face 发布：`jzzh2004/TongGuOCR`。若暂时无法访问，请稍后再试，或参见[权重](#权重)一节的说明。

## 项目简介

TongGuOCR 是一个面向中文历史文献 OCR 的布局感知与词元增强多模态大语言模型（MLLM），主要解决复杂版面、历史生僻字和非平凡阅读顺序带来的识别难题。

该框架包含两个相互配合的核心模块：

- **布局感知预处理（Layout-Aware Preprocessing）**：构建并优化局部连贯的识别块，在保留有效上下文的同时减少不同区域之间的干扰。
- **词元增强识别（Token-Augmented Recognition）**：为生僻字提供直接的单词元表示，并引入行间跳转词元，引导模型沿复杂阅读路径完成解码。

![TongGuOCR 框架概览](website/public/paper-assets/figures/figure-2.png)

## 核心特点

- **布局感知识别块：** 在复杂历史文献页面中兼顾局部上下文、字符清晰度与阅读顺序。
- **生僻字词表扩展：** 通过直接的字符级词元缩短难识别历史字形的解码路径。
- **显式跳转建模：** 无需精确坐标，即可编码文本行之间的粗粒度空间位移。
- **三阶段适配训练：** 逐步完成历史文献领域知识、高质量监督数据与识别块推理粒度的对齐。

## 仓库结构

```
tongguocr/        Python 包：版面检测、识别块划分、OCR 后端、命令行入口
evaluation/       AR/CR/NED + 阅读顺序（RO-ED）评测，自成一体
tools/            环境搭建、权重下载、vLLM 服务、辅助脚本
requirements/     推理环境依赖锁定文件
weights/          由 tools/download_weights.sh 填充（不入库）
website/          项目网站源码
```

## 安装

需要两个 conda 环境，且无法合并（版面栈锁定 `mmcv-full 1.7.2` →
torch 2.10 / python 3.10，而 vLLM 0.32 需要 torch 2.13 / python 3.12）：

```bash
bash tools/setup_env.sh        # `tongguocr`：版面 + transformers OCR（默认后端）
bash tools/setup_vllm_env.sh   # `tongguocr-vllm2`：vLLM 服务（可选后端）
```

注意事项：

- **必须 `transformers>=5.3.0`** —— 发布的权重依赖原生 `glm_ocr` 模型类
  （checkpoint 中无 `auto_map` / 远程代码）。
- `setup_env.sh` 会按平台安装或从源码构建 `mmcv-full 1.7.2`；源码构建需要
  `CUDA_HOME` 下的 CUDA 工具链（约 30–60 分钟）。
- `setup_vllm_env.sh` 自动选择安装路线：NVIDIA 驱动 ≥ 580 → 直接安装 PyPI
  `vllm==0.32.0`；驱动 575.x → 官方 `+cu129` 轮子（PyPI 的 torch/vllm 轮子
  链接的是 CUDA 13，在该驱动上无法运行）。
- **不要对 GlmOcr 模型使用 vLLM 0.18**：其文本栈存在数值 bug，输出会偏离
  transformers 后端。
- 若无法访问 pypi.org / huggingface.co，可通过 `PIP_INDEX` / `HF_ENDPOINT`
  覆盖包索引 / 模型主机（如镜像站）。

## 权重

```bash
bash tools/download_weights.sh
# → weights/ocr/       完整 HF 模型目录（model.safetensors 约 4.7 GB + 配置 + 分词器）
# → weights/layout/    版面检测器 checkpoint（finetune_infer.pth）
```

可通过 `HF_REPO_ID` 覆盖默认仓库 ID。

## 快速上手

```bash
conda activate tongguocr

# 页面图像目录 → 识别纯文本
python -m tongguocr --input-dir <images_dir> --output-dir <output_dir>

# 单张图像
python -m tongguocr --image <page.jpg> --output-dir <output_dir> --print-result
```

（若以 pip 安装本包，`tongguocr` 命令行脚本等价于 `python -m tongguocr`。）

**vLLM 后端**（多页批量更快；与 transformers 后端字节一致）：

```bash
GPU_IDS=0 bash tools/serve_vllm.sh          # 在另一个终端：启动 OpenAI 兼容服务
python -m tongguocr --input-dir <images_dir> --output-dir <output_dir> --ocr-backend vllm
```

全部选项见 `python -m tongguocr --help`。运行行为由内置默认值
（`tongguocr/config.py`）控制，可选的 `--config user.yaml`（YAML/JSON）会
深合并到默认值上——用户配置只需写出要覆盖的键：

```yaml
ocr:
  backend: vllm
  vllm:
    api_base: http://127.0.0.1:8090/v1
```

输出位于 `<output_dir>`：逐页 JSON（`pages/`，含版面、分块、`ocr_text`）、
识别纯文本（`texts/`）、可视化（`visuals/`）、块裁剪图（`block_crops/`），
以及运行摘要 `summary.json`。

### 阅读顺序词元与换行

解码时，模型会在文本行之间输出行间跳转关系词元（如 `<left>`、`<down>`、
`<right|up>` 等，共 24 个——见 `tongguocr/ocr/relation_tokens.py`）。它们是
模型表达阅读顺序的方式。**推理时每个关系词元都会被转换为换行符**，因此
`texts/` 中的纯文本已经编码了完整阅读顺序，每行输出对应一个文本行：

```
原始解码:  第一行文本<down>第二行文本
texts/ 文件: 第一行文本⏎第二行文本     （每行输出对应一个文本行）
```

其余特殊/控制词元一律剥除。该转换在所有后端中完全一致，且 RO-ED 评测将其
视为与原始词元流等价的输入。

## 评测

评测套件可从一次流水线输出复现论文的四个指标（AR、CR、NED、RO-ED）。RO-ED
步骤有独立的依赖锁定文件（`evaluation/requirements_eval.txt`，无需 torch）：

```bash
conda activate tongguocr                          # 步骤 1-3（AR/CR/NED）

# RO-ED 步骤：独立的 torch-free 环境，其锁定的 numpy/pandas 不会与
# 推理环境冲突
conda create -n tongguocr-eval python=3.10 -y
conda run -n tongguocr-eval pip install -r evaluation/requirements_eval.txt
RO_ED_PYTHON="$(conda info --base)/envs/tongguocr-eval/bin/python" \
./evaluation/run_eval.sh \
  --dataset   m5 \                                # 或 mthv2
  --run-dir   <含 texts/ 的流水线输出目录> \
  --dataset-root /path/to/M5 \                    # 或 MTHv2 数据集根目录
  --output-dir <评测输出目录>
```

数据集格式、指标定义、选项（`--skip-ro-ed`、`--match-workers`、`--limit`）
以及存档运行的期望指标值，见 [evaluation/README.md](evaluation/README.md)。

## 实验结果

TongGuOCR 在两个中文历史文献 OCR 基准上取得了当前最佳性能。

| 数据集 | AR ↑ | CR ↑ | NED ↓ | RO-ED ↓ |
| --- | ---: | ---: | ---: | ---: |
| MTHv2 | **97.93** | **98.33** | **2.05** | **1.51** |
| M5HisDoc | **93.76** | **94.46** | **6.15** | **3.49** |

表中数值与论文一致。本仓库的逐后端全量运行可稳定复现：两个 OCR 后端
（transformers 与 vLLM）在所有指标上互相差异不超过 0.06，与表中 M5HisDoc
行的逐后端差异不超过 0.04（与其精确均值的差异不超过 0.01）；MTHv2 行差异
不超过 0.01。存档运行的精确数值见
[evaluation/README.md](evaluation/README.md)。

在更具挑战性的 M5HisDoc 基准上，相较各指标最优的竞争方法，TongGuOCR 将 NED 从 10.43 降低至 6.15，并将 RO-ED 从 7.53 降低至 3.49。完整对比实验、消融研究与定性结果请参阅[论文](https://arxiv.org/abs/2608.07917)和[项目主页](https://jzzh2004.github.io/TongGuOCR)。

## 引用

如果 TongGuOCR 对你的研究有所帮助，请引用我们的论文：

```bibtex
@article{zhou2026tongguocr,
  title   = {TongGuOCR: A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents},
  author  = {Zhou, Zhongheng and Sun, Yi and He, Huiguo and Zhang, Yuyi and Zhang, Peirong and Fang, Yulin and Peng, Dezhi and Liao, Minghui and Jin, Lianwen},
  journal = {arXiv preprint arXiv:2608.07917},
  year    = {2026}
}
```

## 许可证

本仓库代码以 [Apache License 2.0](LICENSE) 开源。第三方致谢（OpenMMLab
栈、内嵌的 OmniDocBench）见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
模型权重另行在 Hugging Face 发布，许可以其模型页为准；M5HisDoc 与 MTHv2
数据集不在本仓库分发范围内。
