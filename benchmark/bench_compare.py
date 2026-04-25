#!/usr/bin/env python3
"""
3-way benchmark: Ollama (qwen3.6:27b) vs plain mlx_lm vs DFlash+DDTree.
Same prompt, same token budget, unified summary table.

Timing convention: output tok/s (generation time only, prefill excluded).
  - Ollama:     eval_count / eval_duration
  - Plain MLX:  stream_generate, clock starts at first emitted token
  - DDTree:     generation_tokens / (elapsed_us - prefill_us)

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_compare.py
"""
import os, time, json, urllib.request, statistics
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once
from mlx_lm import stream_generate

WARMUPS = 2
RUNS = 5
MAX_TOKENS = 200
OLLAMA_MODEL = "qwen3.6:27b"
MLX_MODEL = "mlx-community/Qwen3.5-27B-4bit"

PROMPT = (
    "Implement a red-black tree in Python with insert, delete, and search methods. "
    "Include proper rebalancing logic and type hints."
)

# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_call(num_predict: int) -> dict:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def bench_ollama() -> float:
    d = ollama_call(MAX_TOKENS)
    gt  = d.get("eval_count", 0)
    gns = d.get("eval_duration", 1)
    return gt / (gns / 1e9) if gns > 0 else 0.0

# ── Plain MLX — clock starts at first emitted token ──────────────────────────

def bench_plain(model, tokenizer) -> float:
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

# ── DDTree — generation_tokens / generation_time (prefill excluded) ───────────

def bench_ddtree(target, tokenizer, draft, stop_ids) -> float:
    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True, add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=target, draft_model=draft, tokenizer=tokenizer,
        prompt_tokens=prompt_tokens, max_new_tokens=MAX_TOKENS,
        tree_budget=4, stop_token_ids=stop_ids,
    )
    gen_time_s = (result["elapsed_us"] - result["prefill_us"]) / 1e6
    return result["generation_tokens"] / gen_time_s if gen_time_s > 0 else 0.0

# ── Runner — 2 warmups, 5 runs, median ───────────────────────────────────────

def run_suite(label: str, bench_fn) -> float:
    print(f"\n{'='*62}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*62}", flush=True)
    for w in range(1, WARMUPS + 1):
        print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
        bench_fn()
        print("done", flush=True)
    tps_runs = []
    for i in range(1, RUNS + 1):
        print(f"  [run {i}/{RUNS}] ", end="", flush=True)
        tps = bench_fn()
        tps_runs.append(tps)
        print(f"{tps:.1f} tok/s", flush=True)
    med = statistics.median(tps_runs)
    print(f"  MEDIAN: {med:.1f} tok/s", flush=True)
    return med

# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading MLX models (~19 GB, loaded once)...", flush=True)
ddtree_target, ddtree_tok, ddtree_draft, _ = load_runtime_components(
    model_ref=MLX_MODEL, draft_ref=None
)
ddtree_stop = get_stop_token_ids(ddtree_tok)
print("MLX models loaded.\n", flush=True)

results = []

results.append((
    f"Ollama  {OLLAMA_MODEL}",
    run_suite(f"Ollama  {OLLAMA_MODEL}", bench_ollama),
))

results.append((
    "Plain mlx_lm",
    run_suite(
        f"Plain mlx_lm  {MLX_MODEL}",
        lambda: bench_plain(ddtree_target, ddtree_tok),
    ),
))

results.append((
    "DFlash+DDTree",
    run_suite(
        f"DFlash+DDTree  {MLX_MODEL}",
        lambda: bench_ddtree(ddtree_target, ddtree_tok, ddtree_draft, ddtree_stop),
    ),
))

# ── Summary ───────────────────────────────────────────────────────────────────
baseline = results[0][1]
print(f"\n\n{'='*62}")
print(f"  COMPARISON SUMMARY")
print(f"  {WARMUPS} warmups · {RUNS} runs · median · output tok/s · {MAX_TOKENS} max tok")
print(f"{'='*62}")
print(f"  {'Method':<28}  {'tok/s':>7}  {'vs Ollama':>10}")
print(f"  {'-'*28}  {'-'*7}  {'-'*10}")
for label, tps in results:
    speedup = tps / baseline
    print(f"  {label:<28}  {tps:7.1f}  {speedup:9.2f}×")
print()
