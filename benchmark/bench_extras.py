#!/usr/bin/env python3
"""
Plain MLX benchmark for targets without a DFlash drafter (no DDTree path).

Currently: Qwen3.6-27B-dense (uniform int4) and Qwen3.6-27B-dense (OptiQ 4-bit).

Methodology matches bench_compare: 2 warmups + 5 runs per prompt, multi-prompt
(code/prose/json), greedy sampler (temperature=0), TTFT + decode tok/s, cool-downs,
JSON output to benchmark/results/.

Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_extras.py
"""
import os, time, statistics
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    PROMPTS, HOST, cooldown,
    PRE_BENCH_COOLDOWN_S, INTER_PROMPT_COOLDOWN_S, INTER_CONFIG_COOLDOWN_S, POST_BENCH_COOLDOWN_S,
    make_greedy_sampler, write_results, get_timestamp_iso8601,
)

WARMUPS = 2
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200

# (label, mlx_model_ref) — plain MLX only, no drafter
TARGETS = [
    ("Qwen3.6-27B-dense  [MLX-int4]",       "mlx-community/Qwen3.6-27B-4bit"),
    ("Qwen3.6-27B-dense  [MLX-OptiQ-4bit]", "mlx-community/Qwen3.6-27B-OptiQ-4bit"),
]


def bench_one(model, tokenizer, prompt: str) -> tuple[float, float]:
    """Returns (ttft_ms, decode_tps). Same convention as bench_compare.bench_plain."""
    from mlx_lm import stream_generate
    sampler = make_greedy_sampler()

    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

    t0_stream = time.perf_counter()
    t0_first = None
    count = 0
    for _ in stream_generate(model, tokenizer, prompt=prompt_str,
                             max_tokens=MAX_TOKENS, sampler=sampler):
        if t0_first is None:
            t0_first = time.perf_counter()
            ttft_ms = (t0_first - t0_stream) * 1000.0
        count += 1

    if t0_first is None:
        return 0.0, 0.0

    elapsed_after_first = time.perf_counter() - t0_first
    decode_tps = (count - 1) / elapsed_after_first if elapsed_after_first > 0 else 0.0
    return ttft_ms, decode_tps


from mlx_lm import load
import gc

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before extras")

for target_idx, (label, model_ref) in enumerate(TARGETS):
    if target_idx > 0:
        cooldown(INTER_CONFIG_COOLDOWN_S, "between extras targets")

    print(f"\nLoading {model_ref}...", flush=True)
    model, tokenizer = load(model_ref)
    print("Loaded.", flush=True)

    results = []
    for p_idx, (prompt_name, prompt_text) in enumerate(PROMPTS):
        if p_idx > 0:
            cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {label}")

        print(f"\n  Prompt: {prompt_name}", flush=True)
        for w in range(1, WARMUPS + 1):
            print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
            bench_one(model, tokenizer, prompt_text)
            print("done", flush=True)

        ttft_runs, decode_runs = [], []
        for i in range(1, RUNS_PER_PROMPT + 1):
            print(f"  [run {i}/{RUNS_PER_PROMPT}] ", end="", flush=True)
            ttft_ms, decode_tps = bench_one(model, tokenizer, prompt_text)
            ttft_runs.append(ttft_ms)
            decode_runs.append(decode_tps)
            print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s", flush=True)

        results.append({
            "prompt": prompt_name,
            "ttft_ms_runs": ttft_runs,
            "decode_tps_runs": decode_runs,
            "avg_acceptance": None,
            "ttft_ms_median": statistics.median(ttft_runs),
            "decode_tps_median": statistics.median(decode_runs),
        })

    payload = {
        "ts": get_timestamp_iso8601(),
        "host": HOST,
        "method": "plain-mlx",
        "model_label": label,
        "model_ref": model_ref,
        "drafter_ref": None,
        "tree_budget": None,
        "sampling": {"temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS},
        "warmups": WARMUPS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "results": results,
    }
    path = write_results(payload)
    print(f"\nWrote {label} results to {path}", flush=True)

    del model, tokenizer
    gc.collect()

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")
