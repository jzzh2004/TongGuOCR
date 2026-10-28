#!/usr/bin/env python3
"""Prove the inference prompt is token-identical to the training prompt.

Training annotations use the literal user content "<image>Text Recognition:".
The tokenizer has NO "<image>" token, so the trainer (LLaMA-Factory) replaced
that placeholder with the model's real image token sequence before tokenizing.
Inference passes a structured image item plus the text "Text Recognition:",
and the chat template expands the image item into the same token sequence.

This script demonstrates, at string AND token-id level:

  A. inference rendering (structured image + "Text Recognition:")
  B. training content with "<image>" treated as plain text  -> GARBAGE tokens
  C. training content with "<image>" replaced by the image token sequence
     (what the trainer did)                                   -> identical to A

Optionally (--api-base) it also POSTs the same messages to a live vLLM server
and compares usage.prompt_tokens with the locally computed token count,
verifying the server renders the identical prompt (template + pixels).

Exit code 0 only if A == C and (if checked) server token count == local.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_messages_structured(prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./weights/ocr")
    parser.add_argument("--image", required=True, help="path to a sample page image")
    parser.add_argument("--prompt", default="Text Recognition:")
    parser.add_argument("--api-base", default=None, help="e.g. http://127.0.0.1:8090/v1")
    parser.add_argument("--model-name", default="tonggu-ocr")
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    # Mirror the runtime overrides used by the OCR backends
    # (tongguocr/ocr/transformers_backend.py / vllm_backend.py).
    setattr(processor, "image_max_pixels", int(args.image_max_pixels))
    setattr(processor, "image_min_pixels", int(args.image_min_pixels))
    setattr(processor, "image_do_pan_and_scan", False)
    setattr(processor, "crop_to_patches", False)

    image = Image.open(args.image).convert("RGB")

    # ---- A: inference rendering -------------------------------------------
    text_a = processor.apply_chat_template(
        build_messages_structured(args.prompt), tokenize=False, add_generation_prompt=True
    )
    inputs_a = processor(text=[text_a], images=[image], return_tensors="pt")
    ids_a = inputs_a["input_ids"][0].tolist()

    # ---- B: training string passed through as plain text -------------------
    text_b = processor.apply_chat_template(
        [{"role": "user", "content": f"<image>{args.prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs_b = processor(text=[text_b], images=[image], return_tensors="pt")
    ids_b = inputs_b["input_ids"][0].tolist()

    tokenizer = processor.tokenizer
    literal_tokens = tokenizer.tokenize("<image>")
    print("B) literal '<image>' inside plain text tokenizes to:", literal_tokens)

    # ---- C: training content with placeholder replaced by image tokens -----
    wrapper = "<|begin_of_image|><|image|><|end_of_image|>"
    text_c = processor.apply_chat_template(
        [{"role": "user", "content": f"{wrapper}{args.prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs_c = processor(text=[text_c], images=[image], return_tensors="pt")
    ids_c = inputs_c["input_ids"][0].tolist()

    ok = True
    print("\n--- string level ---")
    print("A == C (inference rendering == trainer-style rendering):", text_a == text_c)
    ok &= text_a == text_c
    print("A == B (would only match if '<image>' were a real token):", text_a == text_b)

    print("\n--- token level ---")
    print("len(ids A) =", len(ids_a), "| len(ids C) =", len(ids_c), "| len(ids B) =", len(ids_b))
    print("ids A == ids C:", ids_a == ids_c)
    ok &= ids_a == ids_c
    print("ids A == ids B:", ids_a == ids_b, "(expected False: B is corrupted)")

    n_image_tokens = sum(1 for t in ids_a if t == 59280)  # image_token_id per config.json
    print(f"image placeholder tokens (id 59280) in A: {n_image_tokens}")

    # ---- optional: live vLLM server cross-check ----------------------------
    if args.api_base:
        import base64
        import io
        import requests

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        resp = requests.post(
            f"{args.api_base.rstrip('/')}/chat/completions",
            json={
                "model": args.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ],
                "max_tokens": 16,
                "temperature": 0.0,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        server_prompt_tokens = data["usage"]["prompt_tokens"]
        print("\n--- vLLM server cross-check ---")
        print("server usage.prompt_tokens =", server_prompt_tokens, "| local len(ids A) =", len(ids_a))
        match = server_prompt_tokens == len(ids_a)
        print("match:", match)
        ok &= match

    print("\nRESULT:", "PASS" if ok else "FAIL")
    print(
        "Conclusion: effective prompt = <image tokens> + %r, which is exactly what "
        "the training annotations '<image>%s' expand to after trainer-side "
        "placeholder replacement." % (args.prompt, args.prompt)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
