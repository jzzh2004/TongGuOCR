# 评测

[English](README.md) · [主文档](../README.zh-CN.md)

```bash
python -m pip install -r evaluation/requirements_eval.txt
bash evaluation/run_eval.sh \
  --dataset m5 \
  --run-dir output/ \
  --dataset-root /path/to/M5 \
  --output-dir eval_out/m5
```

MTHv2 使用 `--dataset mthv2`。全部指标统一写入
**`eval_out/m5/metrics.json`**，完成后自动清理临时评测输入。
评测读取 `texts/*.txt`；模型的 24 个 relation token 已转换为换行，
无需额外的预测文本形态。可直接使用推理环境，或安装了
`requirements_eval.txt` 的 CPU 环境。

## 指标

| 字段 | 定义 | 方向 |
| --- | --- | --- |
| `AR` | 页面准确率的均值：`(N − S − D − I) / N` | 越高越好 |
| `CR` | 页面正确率的均值：`(N − S − D) / N` | 越高越好 |
| `NED` | `distance(pred, gt) / max(len(pred), len(gt), 1)` 的页面均值 | 越低越好 |
| `F1`、`Precision`、`Recall` | 以字符集合计算的页面分数均值 | 越高越好 |
| `BLEU` | 使用平滑的字符级 BLEU 页面均值 | 越高越好 |
| `RO-ED` | 匹配文本行后，阅读顺序序列的归一化编辑距离均值 | 越低越好 |

`N` 为 GT 字符数；`S`、`D`、`I` 为替换、删除、插入次数，编辑路径沿用历史实现的
平局处理规则。字符指标忽略空格和换行；RO-ED 保留行边界，先按内容匹配文本行，
再比较顺序。漏掉的 GT 行计作删除；未匹配的预测行没有 GT 顺序索引。
空顺序对不参与 RO-ED 均值。细节见 [RO-ED 说明](ro_ed/README.zh-CN.md)。

JSON 中 AR/CR/NED/F1/Precision/Recall/BLEU 使用百分数单位；RO-ED 保存原始比例，
乘以 100 后可与论文表格对照。归档论文预测的 RO-ED 为
M5 0.0348767895414561、MTHv2 0.015083750445055579。
重新推理时可能有轻微差异。

## 数据集目录

- M5：`<root>/txt_line/<page_id>.txt`，每行格式为
  `x1,y1,x2,y2,x3,y3,x4,y4,text`。
- MTHv2：`<root>/<subset>/label_textline/<name>.txt`，每行格式为
  `text,x1,y1,x2,y2,x3,y3,x4,y4`。预测页 ID 使用 `<subset>__<name>`。

缺失 GT 时会报错，数据集不随仓库分发。
`--limit N` 选择前 N 页，`--match-workers N` 设置 CPU 评测进程数。

## 已有预测结果

对于保留换行的 JSONL 记录 `{page_id, predict, label}`：

```bash
python evaluation/calculate_metrics.py \
  --input_path predictions.jsonl --output_path metrics.json
```

也兼容使用 `Answer_Simplified` 和 `Response` 字段的旧 `metric.json`。
RO-ED 需要原始行边界，旧输入中已被删除的换行无法恢复。

## 本轮全量复测

2026-10-30：使用词元换行输出和整页批量 vLLM 的管线，完成两数据集、
两后端全量推理。四组共 3600 页，全部成功，逐页／逐块后端、换行转换与
统一指标文件检查通过。

| 数据集 | 后端 | 页数 | AR | CR | NED | F1 | BLEU | RO-ED ×100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 | transformers | 1000 | 93.74 | 94.44 | 6.17 | 94.44 | 88.90 | 3.4730 |
| M5 | vllm | 1000 | 93.81 | 94.48 | 6.11 | 94.43 | 88.93 | 3.4813 |
| MTHv2 | transformers | 800 | 97.93 | 98.32 | 2.05 | 98.03 | 96.23 | 1.5114 |
| MTHv2 | vllm | 800 | 97.93 | 98.32 | 2.05 | 98.03 | 96.22 | 1.5051 |

AR/CR/NED/F1/BLEU 与各后端归档基准的差异均不超过 0.01 个百分点。
RO-ED 保存在结果 JSON 中的仍是原始比例，表格仅为方便与论文对照而乘以 100。
