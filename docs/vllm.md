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
