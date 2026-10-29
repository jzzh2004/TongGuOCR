# RO-ED

[English](README.md) · [评测说明](../README.zh-CN.md)

RO-ED 先按内容将识别文本行与 GT 文本行匹配，再比较阅读顺序。
预测文本以换行和 24 个 relation token 为行边界，去除空行、半角和全角空格。
`protocol.py` 处理结构标签，同时保留实际识别内容。

每条预测行获得其匹配的 GT 行顺序索引。按预测出现次序排列得到 `P`，
参考索引排序后得到 `G`：

```text
page_RO_ED = LevenshteinDistance(G, P) / max(len(G), len(P))
```

最终分数为非空顺序对的页面均值，各页权重相同。
漏掉的 GT 行计作删除；未匹配的预测行没有 GT 顺序索引，不会直接增加插入项，
因此应结合字符识别指标一起解读。

`calculate_metrics.py` 将 RO-ED 和其他指标一起计算，写入同一个 `metrics.json`。
保留模型换行的文本与保留 relation token 的原始文本产生相同的阅读顺序单元。
已核对归档论文结果的全部 1800 页，匹配顺序序列和每页分数均保持一致。

匹配实现及其许可保留在本地，来源见[第三方声明](../../THIRD_PARTY_NOTICES.zh-CN.md)。
