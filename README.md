<div align="center">
  <img src="website/public/tonggu-logo.png" alt="TongGuOCR logo" width="120" />

  # TongGuOCR

  **A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents**

  [Paper](https://arxiv.org/abs/2608.07917) ·
  [Project Page](https://jzzh2004.github.io/TongGuOCR) ·
  [Online Demo](https://guji-ocr.com) ·
  [Weights](https://huggingface.co/jzzh2004/TongGuOCR)

  English · [简体中文](README.zh-CN.md)
</div>

## 📰 News

- **[2026-10]** Inference code, evaluation suite, and model weights released.
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

Full comparisons with competing methods are in the [paper](https://arxiv.org/abs/2608.07917).
Per-backend archived runs of this repository (transformers and vLLM agree
within 0.06 pts on every metric) are listed in
[evaluation/README.md](evaluation/README.md).

## 🛠️ Requirements

| | |
|---|---|
| OS | Linux |
| GPU | NVIDIA, ≥ 16 GB VRAM (24 GB tested), CUDA driver ≥ 12.8 runtime support |
| Python | 3.10 (the setup script creates the conda env) |
| PyTorch | 2.10.0 (cu128 wheel, installed automatically) |
| transformers | ≥ 5.3.0 (native `glm_ocr` support — required) |
| Disk | ~10 GB (weights ≈ 5 GB + environment) |

## 🚀 Installation

```bash
git clone https://github.com/jzzh2004/TongGuOCR.git
cd TongGuOCR

# 1) weights  (or place your local copy under weights/ocr and weights/layout)
bash tools/download_weights.sh

# 2) environment  (~10 min; builds mmcv-full 1.7.2 once, see FAQ)
bash tools/setup_env.sh
conda activate tongguocr
```

If pypi.org is unreachable, prepend a mirror:
`PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash tools/setup_env.sh`.
`HF_ENDPOINT=https://hf-mirror.com` does the same for weight downloads.

## 🖼️ Quickstart

```bash
conda activate tongguocr

# a directory of page images → one recognized text file per page
python -m tongguocr -i <images_dir> -o output/

# a single page (prints the recognized text)
python -m tongguocr -i <page.jpg> -o output/ --print-result

# recognize only the first 10 images of a directory
python -m tongguocr -i <images_dir> -o output/ --limit 10
```

Outputs under `output/`:

| Path | Content |
|---|---|
| `texts/<page>.txt` | plain-form text, one output line per detected layout text line |
| `texts_tokens/<page>.txt` | token form (relation tokens kept; RO-ED input) |
| `pages/<page>.json` | full page record: lines, blocks, per-block OCR text |
| `visuals/<page>.jpg` | layout / block visualization |
| `block_crops/<page>/` | the block crops fed to the recognizer |

**Reading-order tokens.** While decoding, the model emits transition tokens
between text lines (`<down>`, `<left|up>`, …; 24 in total — see
[tongguocr/ocr/relation_tokens.py](tongguocr/ocr/relation_tokens.py)) to
steer generation along the predicted reading path. Each page is written in
two equivalent forms:

- `texts_tokens/<page>.txt` — **token form**: the raw decode with the
  relation tokens kept. Line boundaries here are the model's own
  reading-order predictions; this is the form the paper's RO-ED metric is
  computed on.
- `texts/<page>.txt` — **plain form**: relation tokens removed and the text
  deterministically re-split to one output line per detected layout text
  line. This is the form of the archived AR/CR/NED/F1/BLEU benchmark texts.

```
token form:  太眞家眞人無那隱又以滅百魔<left|far_up>宣王時童謡
plain form:  太眞家眞人無⏎那隱又以滅百⏎魔宣王時童謡   (3 layout lines)
```

### Optional: vLLM backend

An OpenAI-compatible vLLM server gives higher multi-page throughput and is
metric-equivalent to the transformers backend (verified within 0.06 pts on
full benchmarks; byte-identical on probe pages). It needs a second conda env
because vLLM 0.32 requires torch 2.13 while the layout stack pins torch 2.10:

```bash
bash tools/setup_vllm_env.sh      # creates `tongguocr-vllm2`
GPU_IDS=0 bash tools/serve_vllm.sh        # terminal 1: start the server
python -m tongguocr -i <images_dir> -o output/ --backend vllm   # terminal 2
```

Details, driver-specific install routes, and troubleshooting:
[docs/vllm.md](docs/vllm.md).

## 📏 Evaluation

The evaluation suite reproduces the paper's four metrics (AR / CR / NED /
RO-ED) from a pipeline output directory. The M5HisDoc and MTHv2 test sets are
**not** redistributed; point the script at your local copy:

```bash
# --dataset m5 | mthv2; --run-dir = pipeline output containing texts/;
# --dataset-root = your local M5 or MTHv2 copy
./evaluation/run_eval.sh \
  --dataset m5 \
  --run-dir output/ \
  --dataset-root /path/to/M5 \
  --output-dir eval_out/m5
```

Dataset formats, metric definitions, options, and the expected values of the
archived runs: [evaluation/README.md](evaluation/README.md).

## ❓ FAQ

<details>
<summary><code>import cv2</code> / <code>No module named 'shapely'</code> / <code>requires accelerate</code></summary>

`bash tools/setup_env.sh` installs all of these. If you install manually,
follow `requirements/inference.txt` and note that `mmdet 2.25` does not
declare `shapely`, and the transformers backend needs `accelerate`.
</details>

<details>
<summary>mmcv-full 1.7.2 has no wheel for torch 2.10 — what does setup do?</summary>

`tools/setup_env.sh` first looks for a matching wheel in `wheels/`
(`mmcv_full-1.7.2-cp310-*linux_x86_64.whl`, e.g. one built on another
sm_86 machine), then falls back to a source build against your local CUDA
toolkit (`CUDA_HOME`, 5–60 min, cached into `wheels/`).
</details>

<details>
<summary>Can I use vLLM 0.18 / a single merged environment?</summary>

No. vLLM 0.18 has a GlmOcr text-stack numeric bug (degenerate repetition
loops); 0.32 is required and needs torch 2.13, which cannot coexist with the
mmcv-locked torch 2.10 layout stack. See [docs/vllm.md](docs/vllm.md).
</details>

<details>
<summary>Where do the weights go if I have them locally?</summary>

`weights/ocr/` = the full HF model directory (model.safetensors + configs +
tokenizer), `weights/layout/` = `finetune_infer.pth`. Both paths are relative
to the repository root and can be overridden via `--config` (see
[tongguocr/config.py](tongguocr/config.py)).
</details>

## 📚 Citation

```bibtex
@article{zhou2026tongguocr,
  title   = {TongGuOCR: A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents},
  author  = {Zhou, Zhongheng and Sun, Yi and He, Huiguo and Zhang, Yuyi and Zhang, Peirong and Fang, Yulin and Peng, Dezhi and Liao, Minghui and Jin, Lianwen},
  journal = {arXiv preprint arXiv:2608.07917},
  year    = {2026}
}
```

## 📄 License

Code is released under the [Apache License 2.0](LICENSE); third-party
attributions (OpenMMLab stack, vendored OmniDocBench scorer) are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are
distributed separately on Hugging Face under the license stated there. The
M5HisDoc and MTHv2 datasets are not redistributed.
