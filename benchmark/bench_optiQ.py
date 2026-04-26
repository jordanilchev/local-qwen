#!/usr/bin/env python3
"""
Benchmark: plain mlx_lm on Qwen3.6-27B-OptiQ-4bit vs Qwen3.6-27B-4bit.

Timing: output tok/s (generation time only, prefill excluded).
  clock starts at first emitted token from stream_generate.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_optiQ.py
"""
import os, time, statistics, gc
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

WARMUPS = 2
RUNS = 5
MAX_TOKENS = 200

MODELS = [
    ("Qwen3.6-27B-dense  [MLX-int4]",       "mlx-community/Qwen3.6-27B-4bit"),
    ("Qwen3.6-27B-dense  [MLX-OptiQ-4bit]", "mlx-community/Qwen3.6-27B-OptiQ-4bit"),
]

PROMPT = (
    "Implement a red-black tree in Python with insert, delete, and search methods. "
    "Include proper rebalancing logic and type hints."
)


def bench_plain(model, tokenizer) -> float:
    from mlx_lm import stream_generate
    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    t0 = None
    count = 0
    for _ in stream_generate(model, tokenizer, prompt=prompt_str, max_tokens=MAX_TOKENS):
        if t0 is None:
            t0 = time.perf_counter()
        count += 1
    elapsed = time.perf_counter() - t0 if t0 else 0.0
    return count / elapsed if elapsed > 0 else 0.0


results = []

for label, model_ref in MODELS:
    print(f"\n{'='*62}", flush=True)
    print(f"  Loading {model_ref}...", flush=True)
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_ref)
    print(f"  Loaded. Starting benchmark: {label}", flush=True)
    print(f"{'='*62}", flush=True)

    for w in range(1, WARMUPS + 1):
        print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
        bench_plain(model, tokenizer)
        print("done", flush=True)

    tps_runs = []
    for i in range(1, RUNS + 1):
        print(f"  [run {i}/{RUNS}] ", end="", flush=True)
        tps = bench_plain(model, tokenizer)
        tps_runs.append(tps)
        print(f"{tps:.1f} tok/s", flush=True)

    med = statistics.median(tps_runs)
    print(f"  MEDIAN: {med:.1f} tok/s", flush=True)
    results.append((label, med))

    del model, tokenizer
    gc.collect()

print(f"\n\n{'='*62}")
print(f"  SUMMARY  ({WARMUPS} warmups · {RUNS} runs · median · output tok/s · {MAX_TOKENS} max tok)")
print(f"{'='*62}")
baseline = results[0][1]
for label, tps in results:
    ratio = tps / baseline
    print(f"  {label:<40}  {tps:6.1f} tok/s  {ratio:.2f}×")
print()
