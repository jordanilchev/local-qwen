#!/usr/bin/env python3
"""
3-way benchmark: Ollama (qwen3.6:27b) vs plain mlx_lm vs DFlash+DDTree.
Same prompt, same token budget, unified summary table.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_compare.py
"""
import os, time, json, urllib.request
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

RUNS = 3
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

def bench_ollama() -> tuple[float, float]:
    d = ollama_call(MAX_TOKENS)
    pt  = d.get("prompt_eval_count", 0)
    pns = d.get("prompt_eval_duration", 1)
    gt  = d.get("eval_count", 0)
    gns = d.get("eval_duration", 1)
    ttft_ms = pns / 1e6
    out_tps = gt / (gns / 1e9) if gns > 0 else 0.0
    return out_tps, ttft_ms

# ── MLX (plain) ───────────────────────────────────────────────────────────────

def load_mlx():
    from mlx_lm import load as mlx_load
    return mlx_load(MLX_MODEL)

def bench_plain(model, tokenizer) -> tuple[float, float]:
    from mlx_lm import generate as mlx_generate
    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    t0 = time.perf_counter()
    out = mlx_generate(model, tokenizer, prompt=prompt_str, max_tokens=MAX_TOKENS, verbose=False)
    elapsed = time.perf_counter() - t0
    tokens = len(tokenizer.encode(out))
    return tokens / elapsed, 0.0

# ── DDTree ────────────────────────────────────────────────────────────────────

def load_ddtree():
    from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
    target, tok, draft, _ = load_runtime_components(model_ref=MLX_MODEL, draft_ref=None)
    stop_ids = get_stop_token_ids(tok)
    return target, tok, draft, stop_ids

def bench_ddtree(target, tokenizer, draft, stop_ids) -> tuple[float, float]:
    from ddtree_mlx.runtime import generate_ddtree_once
    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True, add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=target, draft_model=draft, tokenizer=tokenizer,
        prompt_tokens=prompt_tokens, max_new_tokens=MAX_TOKENS,
        tree_budget=4, stop_token_ids=stop_ids,
    )
    return result["tokens_per_second"], 0.0

# ── Runner ────────────────────────────────────────────────────────────────────

def run_suite(label: str, warmup_fn, bench_fn) -> float:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print("  [warmup]...", end=" ", flush=True)
    warmup_fn()
    print("done")
    tps_runs = []
    for i in range(1, RUNS + 1):
        print(f"  [run {i}/{RUNS}] ", end="", flush=True)
        tps, _ = bench_fn()
        tps_runs.append(tps)
        print(f"{tps:.1f} tok/s")
    avg = sum(tps_runs) / RUNS
    print(f"  AVG: {avg:.1f} tok/s")
    return avg

# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading MLX models (plain + DDTree share weights)...")
plain_model, plain_tok = load_mlx()
ddtree_target, ddtree_tok, ddtree_draft, ddtree_stop = load_ddtree()
print("MLX models loaded.\n")

results = []

# Ollama
avg = run_suite(
    f"Ollama  {OLLAMA_MODEL}",
    lambda: ollama_call(64),
    bench_ollama,
)
results.append(("Ollama  " + OLLAMA_MODEL, avg))

# Plain MLX
avg = run_suite(
    f"Plain mlx_lm  {MLX_MODEL}",
    lambda: bench_plain(plain_model, plain_tok),
    lambda: bench_plain(plain_model, plain_tok),
)
results.append(("Plain mlx_lm", avg))

# DDTree
avg = run_suite(
    f"DFlash+DDTree  {MLX_MODEL}",
    lambda: bench_ddtree(ddtree_target, ddtree_tok, ddtree_draft, ddtree_stop),
    lambda: bench_ddtree(ddtree_target, ddtree_tok, ddtree_draft, ddtree_stop),
)
results.append(("DFlash+DDTree", avg))

# ── Summary ───────────────────────────────────────────────────────────────────
baseline = results[0][1]
print(f"\n\n{'='*62}")
print(f"  COMPARISON SUMMARY  ({RUNS} runs, post-warmup, {MAX_TOKENS} max tok)")
print(f"{'='*62}")
print(f"  {'Method':<28}  {'tok/s':>7}  {'vs Ollama':>10}")
print(f"  {'-'*28}  {'-'*7}  {'-'*10}")
for label, tps in results:
    speedup = tps / baseline
    print(f"  {label:<28}  {tps:7.1f}  {speedup:9.2f}×")
print()
