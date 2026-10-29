# Developer tools

[简体中文](README.zh-CN.md)

Small utilities used while producing and validating the release; not needed
for inference or evaluation.

- `compare_backends.py` — page/block-level agreement (exact match, CER)
  between two pipeline output directories, e.g. a transformers run vs a vLLM
  run.
- `check_prompt_equivalence.py` — token-level check that the chat-template
  prompt used by the backends matches the training annotation form
  (`<image>Text Recognition:`), optionally against a live vLLM server.
