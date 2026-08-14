<div align="center">
  <img src="website/public/tonggu-logo.png" alt="TongGuOCR logo" width="120" />

  # TongGuOCR

  **A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents**

  [Paper](https://arxiv.org/abs/2608.07917) ·
  [Project Page](https://jzzh2004.github.io/TongGuOCR) ·
  [Online Demo](http://121.41.49.212:6767/)

  English · [简体中文](README.zh-CN.md)
</div>

> **Release status:** This repository currently hosts the TongGuOCR project website. The model weights and inference code will be released after the paper is accepted and published.

## Overview

TongGuOCR is a layout-aware and token-augmented multimodal large language model (MLLM) for OCR of Chinese historical documents. It is designed for pages with complex layouts, rare historical characters, and nontrivial reading orders.

The framework combines two complementary components:

- **Layout-Aware Preprocessing** constructs and refines locally coherent recognition blocks, preserving useful context while reducing interference across regions.
- **Token-Augmented Recognition** gives rare characters direct single-token representations and introduces line-to-line transition tokens to guide decoding along complex reading paths.

![TongGuOCR overview](website/public/paper-assets/figures/figure-2.png)

## Highlights

- **Layout-aware recognition blocks:** balances local context, character legibility, and reading order for complex historical pages.
- **Rare-character vocabulary expansion:** shortens the decoding path for difficult historical glyphs through direct character-level tokens.
- **Explicit transition modeling:** encodes coarse spatial displacement between text lines without requiring precise coordinates.
- **Three-stage adaptation:** progressively aligns historical-domain knowledge, high-quality supervision, and block-level inference granularity.

## Results

TongGuOCR achieves state-of-the-art performance on two Chinese historical document OCR benchmarks.

| Benchmark | AR ↑ | CR ↑ | NED ↓ | RO-ED ↓ |
| --- | ---: | ---: | ---: | ---: |
| MTHv2 | **97.93** | **98.33** | **2.05** | **1.51** |
| M5HisDoc | **93.76** | **94.46** | **6.15** | **3.49** |

On the more challenging M5HisDoc benchmark, TongGuOCR reduces NED from 10.43 to 6.15 and RO-ED from 7.53 to 3.49 relative to the best competing score for each metric. See the [paper](https://arxiv.org/abs/2608.07917) and [project page](https://jzzh2004.github.io/TongGuOCR) for full comparisons, ablations, and qualitative results.

## Project Website

The website is built with React, TypeScript, and Vite.

```bash
cd website
npm install
npm run dev
```

Useful commands:

```bash
npm run build
npm test
```

## Citation

If you find TongGuOCR useful in your research, please cite our paper:

```bibtex
@article{zhou2026tongguocr,
  title   = {TongGuOCR: A Layout-Aware and Token-Augmented OCR MLLM for Chinese Historical Documents},
  author  = {Zhou, Zhongheng and Sun, Yi and He, Huiguo and Zhang, Yuyi and Zhang, Peirong and Fang, Yulin and Peng, Dezhi and Liao, Minghui and Jin, Lianwen},
  journal = {arXiv preprint arXiv:2608.07917},
  year    = {2026}
}
```

## Acknowledgements

TongGuOCR is developed at the South China University of Technology with collaborators from Huawei Technologies Co., Ltd. We thank the creators and maintainers of the datasets and open-source systems used in this work.
