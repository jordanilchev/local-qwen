#!/usr/bin/env python3
"""Single-turn inference: Qwen3.8-27B-4bit + official DFlash2 (target-verified draft).

Same path as benchmark dflash2-mlx: int4 target, 4-bit draft, block_size capped at 5.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import apply_chat_prompt, set_benchmark_seed
from benchmark.dflash2 import DEFAULT_DFLASH2_DRAFT_BITS, load_dflash2_mlx_runtime

TARGET = os.environ.get("TARGET", "mlx-community/Qwen3.8-27B-4bit")
DRAFT = os.environ.get("DRAFT", "z-lab/Qwen3.8-27B-DFlash2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
DRAFT_BITS = int(os.environ.get("DRAFT_BITS", str(DEFAULT_DFLASH2_DRAFT_BITS)))


def run(prompt: str) -> None:
    model, tokenizer, draft, block_size = load_dflash2_mlx_runtime(
        TARGET, DRAFT, DRAFT_BITS
    )
    if override := os.environ.get("BLOCK_SIZE"):
        block_size = max(1, int(override))

    from dflash.model_mlx import stream_generate

    set_benchmark_seed()
    prompt_str = apply_chat_prompt(tokenizer, prompt)

    t0 = time.perf_counter()
    t_first: float | None = None
    n_tokens = 0
    accepted_chunks: list[float] = []

    for response in stream_generate(
        model,
        draft,
        tokenizer,
        prompt_str,
        block_size=block_size,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    ):
        if response.text:
            print(response.text, end="", flush=True)
        if not response.tokens:
            continue
        if t_first is None:
            t_first = time.perf_counter()
        n_tokens += len(response.tokens)
        if response.accepted is not None:
            accepted_chunks.append(float(response.accepted))

    print()
    t_end = time.perf_counter()
    if t_first is None:
        print("→ 0 tok/s", file=sys.stderr)
        return

    decode_s = max(t_end - t_first, 1e-9)
    decode_tps = (n_tokens - 1) / decode_s if n_tokens > 1 else 0.0
    ttft_ms = (t_first - t0) * 1000.0
    msg = (
        f"→ {decode_tps:.1f} tok/s decode, TTFT {ttft_ms:.0f} ms, "
        f"block_size {block_size}, {n_tokens} tok"
    )
    if accepted_chunks:
        avg_acc = sum(accepted_chunks) / len(accepted_chunks)
        msg += f", ~{avg_acc:.1f} accepted/block"
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ")
    run(prompt)
