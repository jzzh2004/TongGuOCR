<div align="center">
  <img src="website/public/tonggu-logo.png" alt="TongGuOCR logo" width="120" />

  # TongGuOCR

  **面向中文历史文献的版面感知·词元增强 OCR 多模态大模型**

  [论文](https://arxiv.org/abs/2608.07917) ·
  [项目主页](https://jzzh2004.github.io/TongGuOCR) ·
  [在线演示](https://guji-ocr.com) ·
  [模型权重](https://huggingface.co/jzzh2004/TongGuOCR)

  [English](README.md) · 简体中文
</div>

## 📰 新闻

- **[2026-10]** 开源推理代码、评测套件与模型权重。
- **[2026-08]** 论文发布于 arXiv。

## 📖 简介

TongGuOCR 是面向中文历史文献 OCR 的多模态大模型,针对复杂版面、生僻字与非平凡阅读顺序的页面。框架由两个互补组件构成:

- **版面感知预处理**:旋转文本行检测 + 动态规划分块,把文本行组织为局部连贯的识别块,保留有效上下文的同时抑制跨区域干扰;
- **词元增强识别**:生僻字获得直接的单字词元表示;行间转移词元(如 `<left|up>`)在解码时显式建模阅读顺序。

<div align="left">
  <img src="website/public/paper-assets/figures/figure-2.png" alt="TongGuOCR 总览" width="90%">
</div>

## 📊 实验结果

| 基准 | AR ↑ | CR ↑ | NED ↓ | RO-ED ↓ |
| :--- | ---: | ---: | ---: | ---: |
| MTHv2 | **97.93** | **98.33** | **2.05** | **1.51** |
| M5HisDoc | **93.76** | **94.46** | **6.15** | **3.49** |

与对比方法的完整比较见[论文](https://arxiv.org/abs/2608.07917)。本仓库两组后端(transformers 与 vLLM,所有指标差异 ≤ 0.06)的归档复现数值见
[evaluation/README.md](evaluation/README.md)。

## 🛠️ 环境要求

| | |
|---|---|
| 操作系统 | Linux |
| GPU | NVIDIA,显存 ≥ 16 GB(24 GB 实测),驱动支持 CUDA 12.8(≥ 525.60) |
| Python | 3.10(setup 脚本自动创建 conda 环境) |
| PyTorch | 2.10.0(cu128 轮子,自动安装) |
| transformers | ≥ 5.3.0(原生 `glm_ocr` 支持——必需) |
| 磁盘 | ~10 GB(权重 ≈ 5 GB + 环境) |

## 🚀 安装

```bash
git clone https://github.com/jzzh2004/TongGuOCR.git
cd TongGuOCR

# 1) 权重(或把本地副本放到 weights/ocr 与 weights/layout)
bash tools/download_weights.sh

# 2) 环境(~10 分钟;mmcv-full 1.7.2 需编译一次,见 FAQ)
bash tools/setup_env.sh
conda activate tongguocr
```

pypi.org 不可达时使用镜像:
`PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash tools/setup_env.sh`;
权重下载镜像:`HF_ENDPOINT=https://hf-mirror.com`。

## 🖼️ 快速上手

```bash
conda activate tongguocr

# 页面图像目录 → 每页一个识别文本文件
python -m tongguocr -i <images_dir> -o output/

# 单页(打印识别结果)
python -m tongguocr -i <page.jpg> -o output/ --print-result

# 只识别目录中前 10 张
python -m tongguocr -i <images_dir> -o output/ --limit 10
```

`output/` 下的输出:

| 路径 | 内容 |
|---|---|
| `texts/<page>.txt` | 识别纯文本,一行文本对应一个输出行 |
| `pages/<page>.json` | 完整页面记录:文本行、块、每块 OCR 文本 |
| `visuals/<page>.jpg` | 版面 / 分块可视化 |
| `block_crops/<page>/` | 送入识别器的块裁剪图 |

**阅读顺序词元。** 解码过程中,模型在文本行之间输出转移词元(`<down>`、`<left|up>` 等,共 24 个,见
[tongguocr/ocr/relation_tokens.py](tongguocr/ocr/relation_tokens.py)),用以引导生成沿预测阅读路径进行。管线解码时剥离全部特殊词元,因此这些词元不会出现在输出文本中;`texts/` 的行结构由管线按 layout 检出的文本行数**确定性重切**得到(优先在标点处断行),行序跟随页面的块阅读顺序——这正是论文基准数字所用的输出语义。

### 可选:vLLM 后端

OpenAI 兼容的 vLLM 服务在多页批处理时吞吐更高,且与 transformers 后端指标等价(全量基准差异 ≤ 0.06;探针页逐字节一致)。由于 vLLM 0.32 需要 torch 2.13、而版面栈锁定 torch 2.10,需第二个 conda 环境:

```bash
bash tools/setup_vllm_env.sh              # 创建 `tongguocr-vllm2`
GPU_IDS=0 bash tools/serve_vllm.sh        # 终端 1:启动服务
python -m tongguocr -i <images_dir> -o output/ --backend vllm   # 终端 2
```

细节、驱动相关的安装路线与排障见 [docs/vllm.md](docs/vllm.md)。

## 📏 评测

评测套件从管线输出目录复现论文四个指标(AR / CR / NED / RO-ED)。M5HisDoc 与 MTHv2 测试集**不随仓库分发**,请指向本地副本:

```bash
./evaluation/run_eval.sh \
  --dataset m5 \                      # 或 mthv2
  --run-dir output/ \                 # 含 texts/ 的管线输出目录
  --dataset-root /path/to/M5 \        # 或 MTHv2 根目录
  --output-dir eval_out/m5
```

数据集格式、指标定义、参数与归档运行的期望数值见
[evaluation/README.md](evaluation/README.md)。

## ❓ 常见问题

<details>
<summary><code>import cv2</code> / <code>No module named 'shapely'</code> / <code>requires accelerate</code></summary>

`bash tools/setup_env.sh` 会全部装好。手动安装请按 `requirements/inference.txt`;注意 `mmdet 2.25` 未声明 `shapely` 依赖,transformers 后端还需要 `accelerate`。
</details>

<details>
<summary>mmcv-full 1.7.2 没有 torch 2.10 的轮子,setup 怎么处理?</summary>

`tools/setup_env.sh` 先在 `wheels/` 里找匹配的轮子(`mmcv_full-1.7.2-cp310-*linux_x86_64.whl`,例如在别的 sm_86 机器上编好的),没有则用本地 CUDA 工具链源码编译(`CUDA_HOME`,5–60 分钟,结果缓存进 `wheels/`)。
</details>

<details>
<summary>能否用 vLLM 0.18 / 把两个环境合并成一个?</summary>

不能。vLLM 0.18 存在 GlmOcr 文本栈数值 bug(输出退化为复读循环);必须 0.32,而它需要 torch 2.13,与 mmcv 锁定的 torch 2.10 版面栈无法共存。详见 [docs/vllm.md](docs/vllm.md)。
</details>

<details>
<summary>已有本地权重,放在哪里?</summary>

`weights/ocr/` = 完整 HF 模型目录(model.safetensors + 配置 + tokenizer);`weights/layout/` = `finetune_infer.pth`。两个路径均相对仓库根目录,可通过 `--config` 覆盖(见 [tongguocr/config.py](tongguocr/config.py))。
</details>

## 📚 引用

```bibtex
@article{zhou2026tongguocr,
  title   = {TongGuOCR: A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents},
  author  = {Zhou, Zhongheng and Sun, Yi and He, Huiguo and Zhang, Yuyi and Zhang, Peirong and Fang, Yulin and Peng, Dezhi and Liao, Minghui and Jin, Lianwen},
  journal = {arXiv preprint arXiv:2608.07917},
  year    = {2026}
}
```

## 📄 许可

代码以 [Apache License 2.0](LICENSE) 开源;第三方声明(OpenMMLab 系、内置的 OmniDocBench 评测器)见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型权重在 Hugging Face 单独发布,许可以其页面为准。M5HisDoc 与 MTHv2 数据集不随仓库分发。
