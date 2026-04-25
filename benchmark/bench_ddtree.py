#!/usr/bin/env python3
"""
Benchmark: plain mlx_lm vs DFlash vs DFlash+DDTree on Qwen3.5-27B-4bit.
Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_ddtree.py
"""
import os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

import mlx.core as mx
from mlx_lm import load as mlx_load, generate as mlx_generate
from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once

TARGET = "mlx-community/Qwen3.5-27B-4bit"
RUNS = 3
MAX_TOKENS = 200
TREE_BUDGET = 4

PROMPT = (
    "Implement a red-black tree in Python with insert, delete, and search methods. "
    "Include proper rebalancing logic and type hints."
)


def bench_plain(model, tokenizer) -> float:
    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    t0 = time.perf_counter()
    out = mlx_generate(model, tokenizer, prompt=prompt_str, max_tokens=MAX_TOKENS, verbose=False)
    elapsed = time.perf_counter() - t0
    tokens = len(tokenizer.encode(out))
    return tokens / elapsed


def bench_ddtree(target_model, tokenizer, draft_model) -> tuple[float, float]:
    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=MAX_TOKENS,
        tree_budget=TREE_BUDGET,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )
    return result["tokens_per_second"], result.get("acceptance_rate", float("nan"))


print("Loading models...", flush=True)
plain_model, plain_tokenizer = mlx_load(TARGET)
target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET, draft_ref=None)
print("Models loaded.\n", flush=True)

summary = []

for label, fn in [
    ("plain mlx_lm", lambda: bench_plain(plain_model, plain_tokenizer)),
    ("DFlash+DDTree", lambda: bench_ddtree(target_model, tokenizer, draft_model)),
]:
    print(f"{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)

    print("  [warmup]...", end=" ", flush=True)
    fn()
    print("done", flush=True)

    runs = []
    for i in range(1, RUNS + 1):
        print(f"  [run {i}/{RUNS}] ", end="", flush=True)
        result = fn()
        if isinstance(result, tuple):
            tps, acc = result
        else:
            tps, acc = result, float("nan")
        runs.append((tps, acc))
        acc_str = f"  accept {acc:.0%}" if acc == acc else ""
        print(f"{tps:.1f} tok/s{acc_str}", flush=True)

    avg_tps = sum(r[0] for r in runs) / RUNS
    valid_accs = [r[1] for r in runs if r[1] == r[1]]
    avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else float("nan")
    summary.append((label, avg_tps, avg_acc))

    print(f"  AVG: {avg_tps:.1f} tok/s\n", flush=True)

print(f"\n{'='*60}")
print(f"  SUMMARY  ({RUNS} runs, post-warmup, {MAX_TOKENS} max tokens)")
print(f"{'='*60}")
print(f"  {'Method':<20}  {'tok/s':>7}  {'speedup':>8}  {'accept':>8}")
print(f"  {'-'*20}  {'-'*7}  {'-'*8}  {'-'*8}")
baseline = summary[0][1]
for label, tps, acc in summary:
    speedup = tps / baseline
    acc_str = f"{acc:.0%}" if acc == acc else "  —"
    print(f"  {label:<20}  {tps:7.1f}  {speedup:7.2f}×  {acc_str:>8}")
print()
