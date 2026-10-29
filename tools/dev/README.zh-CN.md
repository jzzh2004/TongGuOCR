# 开发工具

[English](README.md)

这些工具用于开发和验证，推理与评测不需要使用。

- `compare_backends.py`：比较两次推理的页面／块文本一致率和字符错误率。
- `check_prompt_equivalence.py`：检查后端的聊天模板提示词是否与训练标注
  `<image>Text Recognition:` 一致，可选择连接运行中的 vLLM 服务。
