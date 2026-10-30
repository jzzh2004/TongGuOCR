# vLLM

[简体中文](vllm.zh-CN.md) · [Main guide](../README.md)

The layout process uses the `tongguocr` environment. The OCR server uses
`tongguocr-vllm2` (Python 3.12, vLLM 0.32.0, PyTorch 2.13.0).
These are separate because the installed layout extension was compiled for
PyTorch 2.10.0. This is the tested software combination.

```bash
bash tools/setup_vllm_env.sh
GPU_IDS=0 bash tools/serve_vllm.sh  # terminal 1
conda activate tongguocr          # terminal 2
python -m tongguocr -i <images_dir> -o output/ --backend vllm
```

Setup selects CUDA 13 wheels for driver 580 or newer, or CUDA 12.9 wheels
for the tested 575 driver. `VLLM_ROUTE=pypi|cu129` overrides selection.

## Page batches

The client calls `/v1/chat/completions/batch` once per page with all its
crops. Each crop has an independent image and prompt. Response indices
restore crop order; a failed or incomplete response fails the page.
The server controls how many sequences run together and queues the rest.

`ocr.vllm.batch_size` defaults to `0` (whole page). A positive value caps
crops per request; `1` runs serially. For example, `--config serial.yaml`:

```yaml
ocr:
  vllm:
    batch_size: 1
```

Both modes use the same bf16 weights, prompt, image budget, greedy decoding,
and 8192-token output limit. Batching can change individual characters;
check recognition and reading-order metrics when changing settings.

## Memory and connection settings

`serve_vllm.sh` accepts these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `weights/ocr` | OCR weights |
| `GPU_IDS` | `0` | GPU selection |
| `PORT` | `8090` | Server port |
| `KV_CACHE_MEMORY_BYTES` | `2147483648` | Fixed 2 GiB KV cache |
| `MAX_NUM_SEQS` | `16` | Maximum active sequences; remaining crops wait |
| `MAX_MODEL_LEN` | `16384` | Context limit, including image, prompt and output |

The fixed KV budget replaces proportional cache reservation. GPU memory
includes weights, KV cache, activations and the layout process. A server
reserving tens of GiB does not establish a minimum hardware requirement.
For a remote server, set `ocr.vllm.api_base` to its `/v1` URL in a config.
Check readiness with `wget -qO- http://127.0.0.1:8090/v1/models`.

## Measured comparison

A6000, vLLM 0.32, bf16, eager execution, 2 GiB KV cache. Each mode warms
the full sample first and uses the same fixed seed-42 crops. Timings include
image preparation and HTTP; layout detection is excluded.

| Dataset | Pages / crops | Serial | Four crops / request | Whole page | Whole-page speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| M5 | 20 / 50 | 183.98 s | 110.75 s | 104.77 s | 1.76× |
| MTHv2 | 20 / 43 | 130.57 s | 77.87 s | 77.49 s | 1.68× |

Whole-page minus serial: M5 AR −0.01, CR +0.03, NED +0.01, BLEU −0.07 pt;
MTHv2 +0.02, +0.02, −0.02, +0.01 pt respectively. RO-ED improves on M5 and
is unchanged on MTHv2. One 11-crop page takes 17.41, 6.74 and 3.60 seconds
respectively. The sample shows no substantial regression. Full-set results are listed below.

The updated page-batched pipeline completed fresh end-to-end runs on all
1000 M5 and 800 MTHv2 pages with zero failures. Every page/block backend,
raw-decode-to-newline conversion and combined metric output was checked.

| Dataset | AR | CR | NED | F1 | BLEU | RO-ED ×100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 | 93.81 | 94.48 | 6.11 | 94.43 | 88.93 | 3.4813 |
| MTHv2 | 97.93 | 98.32 | 2.05 | 98.03 | 96.22 | 1.5051 |

AR/CR/NED/F1/BLEU differ from archived serial vLLM results by at most 0.01 pt.
Sampled total device peaks for same-card layout plus serving were 22.6 GiB
(M5) and 25.3 GiB (MTHv2), sampled every 0.5 s. OCR-only crop measurements
used about 6.4 GiB. These A6000 observations include runtime caches and
temporary allocations and do not establish minimum VRAM requirements.
