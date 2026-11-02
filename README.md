<div align="center">
  <img src="website/public/tonggu-logo.png" alt="TongGuOCR logo" width="120" />

  # TongGuOCR

  **A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents**

  [![Paper](https://img.shields.io/badge/arXiv-2608.07917-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.07917)
  [![Project Page](https://img.shields.io/badge/Project-Page-2563EB?style=for-the-badge&logo=googlechrome&logoColor=white)](https://jzzh2004.github.io/TongGuOCR)
  [![Online Demo](https://img.shields.io/badge/Online-Demo-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://guji-ocr.com)

  [![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=1F2937)](https://huggingface.co/jzzh2004/TongGuOCR)
  [![ModelScope International](https://img.shields.io/badge/ModelScope-624AFF?style=for-the-badge&logo=modelscope&logoColor=white)](https://modelscope.ai/models/jzzh2004/TongGuOCR)

  English · [简体中文](README.zh-CN.md)
</div>

## 📰 News

- **[2026-10]** Inference code and evaluation tools.
- **[2026-08]** Paper released on arXiv.

## 📖 Introduction

TongGuOCR is a multimodal LLM for OCR of Chinese historical documents — pages
with complex layouts, rare historical characters (生僻字), and non-trivial
reading orders. It combines two components:

- **Layout-aware preprocessing** — a rotated-text-line detector plus a DP
  block segmentation that groups lines into locally coherent recognition
  blocks, preserving context while reducing cross-region interference.
- **Token-augmented recognition** — rare characters get direct single-token
  representations, and line-to-line transition tokens (e.g. `<left|up>`)
  explicitly encode reading order during decoding.

<div align="left">
  <img src="website/public/paper-assets/figures/figure-2.png" alt="TongGuOCR overview" width="90%">
</div>

## 📊 Results

| Benchmark | AR ↑ | CR ↑ | NED ↓ | RO-ED ↓ |
| :--- | ---: | ---: | ---: | ---: |
| MTHv2 | **97.93** | **98.33** | **2.05** | **1.51** |
| M5HisDoc | **93.76** | **94.46** | **6.15** | **3.49** |

See the [paper](https://arxiv.org/abs/2608.07917) for full comparisons.

## Repository structure

| Directory | Contents |
| --- | --- |
| `tongguocr/` | Inference pipeline: layout detection, blocking, crops, and OCR backends |
| [evaluation/](evaluation/README.md) | Dataset adapters, combined metrics, and the local RO-ED implementation |
| `tools/` | Setup, weight download, and serving scripts; development checks in `tools/dev/` |
| `requirements/` | Main inference environment dependencies |
| `weights/` | Local weights; Git tracks only directory placeholders |
| `tests/`, `evaluation/tests/` | Inference protocol and evaluation regression tests |
| [website/](website/README.md) | Independent project page built with Node.js |

## Installation

Tested stack: Linux, Python 3.10, PyTorch 2.10.0 (CUDA 12.8),
Transformers 5.3.0, and an NVIDIA GPU with bf16 support. The setup script
installs these dependencies. The NVIDIA driver must support the selected
CUDA runtime; the current test machine uses driver 575.64 and an RTX A6000.
VRAM usage depends on image size and backend; a minimum GPU size has not
been established. Reserve about 20 GB of disk space for weights, the main
environment, and installation files; the optional vLLM environment needs more.

```bash
git clone https://github.com/jzzh2004/TongGuOCR.git
cd TongGuOCR
bash tools/setup_env.sh
conda activate tongguocr
```

Download the final Stage 3 weights from
[Hugging Face](https://huggingface.co/jzzh2004/TongGuOCR):

```bash
bash tools/download_weights.sh huggingface
```

If Hugging Face is unreachable, use
[ModelScope International](https://modelscope.ai/models/jzzh2004/TongGuOCR):

```bash
python -m pip install modelscope
bash tools/download_weights.sh modelscope
```

The ModelScope repository is on `modelscope.ai`, not `modelscope.cn`.
While a repository is private, access requires an authorized account:
set `HF_TOKEN` for Hugging Face or `MODELSCOPE_API_TOKEN` for ModelScope
International. The script places the seven OCR files in `weights/ocr/`
and the detector at `weights/layout/finetune_infer.pth`. Local copies can
also be placed at these paths.

The detector uses a compiled `mmcv-full` extension. Setup uses a compatible
wheel from `wheels/`, or builds it with the local CUDA toolkit. A source
build can take several minutes. For a Python package mirror, set `PIP_INDEX`.

## Recognition

```bash
python -m tongguocr -i <images_dir> -o output/
python -m tongguocr -i <page.jpg> -o output/ --print-result
# Optional: --limit 10, --no-visualize, --config config.yaml
```

| Output | Content |
| --- | --- |
| `texts/<page>.txt` | Recognized text; each of the 24 relation tokens becomes a newline |
| `pages/<page>.json` | Layout, blocks, recognized text, and raw block decodes |
| `visuals/<page>.jpg` | Layout and block visualization |
| `block_crops/<page>/` | Recognition crops |

For example, `甲乙<left|up>丙丁` becomes `甲乙\n丙丁`.
Line breaks follow the model's output. Text is not re-split using detection
counts or punctuation.

### Optional vLLM backend

```bash
bash tools/setup_vllm_env.sh
GPU_IDS=0 bash tools/serve_vllm.sh  # terminal 1
python -m tongguocr -i <images_dir> -o output/ --backend vllm  # terminal 2
```

All crops from one page are sent in one batch request, with one independent
conversation per crop. vLLM schedules them on the server. See
[the vLLM guide](docs/vllm.md) for configuration and memory settings.

## Evaluation

```bash
python -m pip install -r evaluation/requirements_eval.txt
bash evaluation/run_eval.sh \
  --dataset m5 \
  --run-dir output/ \
  --dataset-root /path/to/M5 \
  --output-dir eval_out/m5
```

AR, CR, NED, F1, Precision, Recall, BLEU, and RO-ED are written together to
**`eval_out/m5/metrics.json`**, using the same recognized text. Select
`--dataset mthv2` for MTHv2. Evaluation runs in the inference environment
or a CPU environment. Dataset formats and metric definitions are in
[the evaluation guide](evaluation/README.md). The datasets are not included.

## Citation

```bibtex
@article{zhou2026tongguocr,
  title   = {TongGuOCR: A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents},
  author  = {Zhou, Zhongheng and Sun, Yi and He, Huiguo and Zhang, Yuyi and Zhang, Peirong and Fang, Yulin and Peng, Dezhi and Liao, Minghui and Jin, Lianwen},
  journal = {arXiv preprint arXiv:2608.07917},
  year    = {2026}
}
```

## License

Code, TongGuOCR's own OCR adaptations, and the detection checkpoint:
[Apache-2.0](LICENSE). The GLM-OCR base model retains MIT terms. See
[third-party notices](THIRD_PARTY_NOTICES.md) for license scope and attribution.
