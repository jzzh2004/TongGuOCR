# vLLM

[English](vllm.md) · [主文档](../README.zh-CN.md)

版面处理使用 `tongguocr` 环境，OCR 服务使用 `tongguocr-vllm2`
（Python 3.12、vLLM 0.32.0、PyTorch 2.13.0）。已安装的版面扩展按
PyTorch 2.10.0 编译，因此服务采用独立环境；以上是已验证的软件组合。

```bash
bash tools/setup_vllm_env.sh
GPU_IDS=0 bash tools/serve_vllm.sh  # 终端 1
conda activate tongguocr          # 终端 2
python -m tongguocr -i <images_dir> -o output/ --backend vllm
```

安装脚本为 580 及以上驱动选择 CUDA 13 轮子，为已测试的 575 驱动选择
CUDA 12.9 轮子。可通过 `VLLM_ROUTE=pypi|cu129` 覆盖选择。

## 整页批量处理

客户端通过 `/v1/chat/completions/batch`，用一次请求提交同页全部裁剪图。
每个裁剪图使用独立图片和提示词，根据返回索引恢复顺序；请求失败或结果缺失时，
该页报告失败。服务端控制同时执行的序列数，其余裁剪图排队等待。

`ocr.vllm.batch_size` 默认为 `0`（整页）。正数表示每次请求最多提交的裁剪图数，
`1` 表示串行。例如通过 `--config serial.yaml` 使用：

```yaml
ocr:
  vllm:
    batch_size: 1
```

两种模式使用相同的 bf16 权重、提示词、像素预算、贪心解码和 8192 token 输出上限。
批处理可能带来个别字符差异；调整配置时应同时检查识别和阅读顺序指标。

## 显存与连接配置

`serve_vllm.sh` 支持以下环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MODEL_PATH` | `weights/ocr` | OCR 权重 |
| `GPU_IDS` | `0` | GPU 编号 |
| `PORT` | `8090` | 服务端口 |
| `KV_CACHE_MEMORY_BYTES` | `2147483648` | 固定 2 GiB KV cache |
| `MAX_NUM_SEQS` | `16` | 同时执行的序列上限，其余裁剪图等待 |
| `MAX_MODEL_LEN` | `16384` | 包含图像、提示词和输出的上下文上限 |

固定缓存预算替代按整卡比例预留缓存。实际显存还包括权重、激活和版面处理。
服务预留了数十 GiB 显存，并不代表模型要求这么大的显卡。
连接远程服务时，在配置文件中将 `ocr.vllm.api_base` 设置为服务的 `/v1` 地址。
可用 `wget -qO- http://127.0.0.1:8090/v1/models` 检查服务状态。
