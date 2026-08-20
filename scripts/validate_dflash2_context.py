#!/usr/bin/env python3
"""Validate DFlash2 MLX prefill + decode at a context that fits 32 GB UMA.

Default 8192 tokens. 100k is refused before loading weights (~24 GB KV + 20 GB model).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark.dflash2 import DEFAULT_MAX_CONTEXT, require_context_fits


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DFlash2 long-context inference")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--target", default="mlx-community/Qwen3.8-27B-4bit")
    parser.add_argument("--draft", default="z-lab/Qwen3.8-27B-DFlash2")
    args = parser.parse_args()

    require_context_fits(args.target_tokens)

    from dflash.model_mlx import stream_generate

    from benchmark._lib import set_benchmark_seed
    from benchmark.dflash2 import build_filler_prompt, load_dflash2_mlx_runtime

    print("Loading target + draft...", flush=True)
    model, tokenizer, draft, block_size = load_dflash2_mlx_runtime(args.target, args.draft)

    print(f"Building ~{args.target_tokens:,} token prompt...", flush=True)
    t_build = time.perf_counter()
    prompt_str, prompt_tokens = build_filler_prompt(tokenizer, args.target_tokens)
    build_s = time.perf_counter() - t_build
    print(f"  prompt_tokens={prompt_tokens:,} (build {build_s:.1f}s)", flush=True)
    require_context_fits(prompt_tokens)

    set_benchmark_seed()
    t0 = time.perf_counter()
    t_first = None
    gen_tokens = 0
    for response in stream_generate(
        model,
        draft,
        tokenizer,
        prompt_str,
        block_size=block_size,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
    ):
        if response.tokens:
            if t_first is None:
                t_first = time.perf_counter()
            gen_tokens += len(response.tokens)

    t_end = time.perf_counter()
    if t_first is None:
        print("FAIL: no tokens generated", file=sys.stderr)
        return 1

    ttft_s = t_first - t0
    total_s = t_end - t0
    decode_s = max(t_end - t_first, 1e-9)
    decode_tps = (gen_tokens - 1) / decode_s if gen_tokens > 1 else 0.0

    ok = prompt_tokens >= args.target_tokens and gen_tokens >= 1
    print(
        f"{'PASS' if ok else 'FAIL'}: context={prompt_tokens:,} tok, "
        f"TTFT={ttft_s:.1f}s, decode={decode_tps:.1f} tok/s, "
        f"generated={gen_tokens}, block_size={block_size}, total={total_s:.1f}s",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
