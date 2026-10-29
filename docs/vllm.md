# Optional: the vLLM backend

The default `transformers` backend runs everything in one process and one
conda env. The `vllm` backend is an OpenAI-compatible HTTP client: it keeps
the layout stack running in the `tongguocr` env and sends only the block
crops to a running vLLM server, which batches them and is faster on
multi-page runs.

**Accuracy.** Archived serial runs of the two backends agree within
0.06 pts on AR/CR/NED/F1/BLEU over the full M5HisDoc (1000 pages) and MTHv2
(800 pages) benchmarks. The default client now batches same-page crops;
its measured speed and metric comparison are listed below. Changes in
batching can change individual characters through bf16 numeric differences.

## Why a second environment (and why vLLM 0.18 is not an option)

| env | contents | used for |
|---|---|---|
| `tongguocr` | python 3.10 · torch 2.10.0+cu128 · transformers 5.3.0 · mmcv-full 1.7.2 / mmdet 2.25.2 / mmrotate 0.3.2 | layout + transformers OCR (default backend), client side of vLLM |
| `tongguocr-vllm2` | python 3.12 · vllm 0.32.0 · torch 2.13.0 · transformers 5.19.0 | serving the OCR model only |

Two hard constraints force this split:

1. **mmcv-full 1.7.2 pins torch 2.10** (the layout stack), while **vLLM 0.32
   needs torch 2.13**. No single env can hold both.
2. **vLLM ≥ 0.32 is required.** vLLM 0.18's GlmOcr text stack has a numeric
   bug: logits diverge from transformers by multiple nats, generations fork
   mid-text and degenerate into repetition loops. 0.32 fixes it.

## Setup

```bash
bash tools/setup_vllm_env.sh        # creates `tongguocr-vllm2`
```

The script picks an install route from your NVIDIA driver:

- **driver ≥ 580** → plain `pip install vllm==0.32.0` (the PyPI wheel is
  CUDA-13-linked and runs natively).
- **driver 575.x (CUDA 12.9)** → the PyPI torch/vllm wheels are CUDA-13
  linked and cannot even initialize CUDA. The script then installs
  `torch 2.13.0+cu129` from download.pytorch.org and the official
  `vllm-0.32.0+cu129` wheel (cached into `wheels/` if present), plus
  `nvidia-cuda-nvrtc/nvidia-cuda-runtime==13.4.92` with two symlinks inside
  `torchcodec.libs/` (vLLM imports torchcodec at startup; the OCR path never
  executes it, the symlinks just satisfy its cu13 sonames).

Override the detection with `VLLM_ROUTE=pypi|cu129`. Mirrors:
`PIP_INDEX=...`, `TORCH_INDEX=...`, `VLLM_INDEX=...`.

## Serve and run

```bash
# terminal 1: server (from the repo root; waits for readiness on :8090)
GPU_IDS=0 bash tools/serve_vllm.sh
# readiness probe: wget -qO- http://127.0.0.1:8090/v1/models

# terminal 2: pipeline against the server
conda activate tongguocr
python -m tongguocr -i <images_dir> -o output/ --backend vllm
```

`serve_vllm.sh` defaults (override via env vars of the same name):

| var | default | notes |
|---|---|---|
| `MODEL_PATH` | `weights/ocr` | |
| `PORT` | `8090` | one server per machine; the client default matches |
| `GPU_MEMORY_UTILIZATION` | `0.55` | leave room for same-card layout (MTH dewarp peaks 1–1.9 GB/page). Use 0.9 only if nothing else shares the card |
| `DTYPE` | `bfloat16` | must match the transformers backend; `auto` would load fp32 |
| `MAX_MODEL_LEN` | `32768` | 8192 generation tokens + ~1400 image tokens + margin |
| `SEED` | `42` | |
| `ENFORCE_EAGER` | `1` | torch.compile clashed with transformers 5.x on 0.18; all alignment evidence was produced in eager mode. Re-validate before changing |

The server-side `--mm-processor-kwargs` pixel budget is a backstop; the
client (`tongguocr/ocr/vllm_backend.py`) pre-resizes every crop to the same
budget before upload, so both layers always agree.

## Concurrent crops from one page

The client submits a page's block crops concurrently, with up to four
independent requests in flight (`ocr.vllm.concurrency: 4`). vLLM batches
these requests on the server. Each crop retains its own image and prompt;
results are collected in the original block order before page assembly.
A failed request fails that page instead of dropping a block silently.

Set `ocr.vllm.concurrency: 1` in a `--config` YAML/JSON file to compare with
serial requests. The image budget, dtype, prompt, greedy decoding, and
maximum output length are shared by both modes. Changing the batch can
slightly change bf16 outputs, so compare recognition and reading-order
metrics when tuning concurrency.

### Measured crop-recognition throughput

On one RTX A6000, vLLM 0.32.0, bf16, greedy decoding, and eager execution,
a fixed seed-42 sample compared serial requests with concurrency 4 on the
same saved crops (one untimed warmup request):

| Dataset | Pages / crops | Serial OCR | Concurrent OCR | Speedup | ΔAR | ΔCR | ΔNED | ΔRO-ED ×100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 | 20 / 50 | 192.31 s | 100.83 s | 1.91× | +0.01 | +0.05 | −0.02 | −0.6087 |
| MTHv2 | 20 / 43 | 132.81 s | 70.76 s | 1.88× | −0.03 | −0.04 | +0.03 | 0.0000 |

Deltas are concurrent minus serial, in percentage points. Higher AR/CR and
lower NED/RO-ED are better. M5 BLEU changes by −0.01 pt and MTHv2 by −0.06 pt;
13/20 and 18/20 page texts respectively remain byte-identical. A page with
11 crops takes 19.99 s serially and 5.25 s with concurrency 4.

A second control selected the five pages with the most crops from each
sample. Each page was warmed immediately before both measurements, and
serial/concurrent execution order alternated. M5 recognition took 66.03 s
serially versus 25.28 s concurrently (2.61×); MTHv2 took 37.39 s versus
18.06 s (2.07×). The benefit therefore persists with both modes warm.

These numbers cover recognition, client image preparation, and HTTP;
layout detection is excluded. Server caches are enabled. This 40-page
comparison checks that batching has no substantial metric regression on
the sample; full-dataset validation is reported separately.

## Troubleshooting

- **Server not reachable / requests time out** — check
  `wget -qO- http://127.0.0.1:8090/v1/models`; first cold start can take
  several minutes. One server per machine (the port is fixed unless you
  change it on both sides).
- **Client `ReadTimeout` on a huge page** — a degenerate block can generate
  up to 8192 tokens; the client timeout is 600 s
  (`ocr.vllm.timeout` in `tongguocr/config.py`). This should not happen on
  vLLM 0.32.
- **OOM on the layout side** — lower `GPU_MEMORY_UTILIZATION` (0.55 is the
  tested same-card value on 24 GB / 48 GB GPUs).
- **Different GPU than the archived runs** — bf16 cross-hardware drift of
  ±0.02–0.06 pts on the benchmark metrics is normal; the archived fresh-environment HF check
  preserved exact page text on 922/1000 M5 and 788/800 MTHv2 pages.
