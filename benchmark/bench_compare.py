#!/usr/bin/env python3
"""
Benchmark: multiple Ollama models vs plain mlx_lm vs DFlash+DDTree.
Same prompt, same token budget, unified summary table.

Timing convention: output tok/s (generation time only, prefill excluded).
  - Ollama:     eval_count / eval_duration
  - Plain MLX:  stream_generate, clock starts at first emitted token
  - DDTree:     generation_tokens / (elapsed_us - prefill_us)

Memory safety: Ollama models are unloaded before MLX is loaded.
  Peak during Ollama phase: up to 28 GB (35b-q5)
  Peak during MLX phase:    ~19 GB

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_compare.py
"""
import os, time, json, urllib.request, statistics
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

WARMUPS = 2
RUNS = 5
MAX_TOKENS = 200

# Labels include model size, architecture, and quant type for clarity.
# Ollama: quant is part of the tag name (q4 = Q4_K_M, :27b = default Ollama quant)
OLLAMA_MODELS = [
    ("Qwen3.6-27B-dense  [GGUF-Q4_K_M]",       "qwen3.6:27b"),
    ("Qwen3.6-35B-MoE    [GGUF-Q4_K_P]",       "qwen3.6-uncensored:35b-q4"),
]

# Each entry: (label, mlx_model_ref, draft_ref)
# draft_ref=None → auto-resolve from dflash registry; set explicitly when not in registry.
MLX_MODELS = [
    ("Qwen3.5-27B-dense  [MLX-int4]",     "mlx-community/Qwen3.5-27B-4bit",          None),
    ("Qwen3.6-35B-MoE    [MLX-int4-DWQ]", "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ", "z-lab/Qwen3.6-35B-A3B-DFlash"),
]

PROMPT = (
    "Implement a red-black tree in Python with insert, delete, and search methods. "
    "Include proper rebalancing logic and type hints."
)

# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_call(model: str, num_predict: int) -> dict:
    payload = json.dumps({
        "model": model,
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

def ollama_unload(model: str) -> None:
    """Evict model from Ollama's memory immediately."""
    payload = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception:
        pass

def bench_ollama(model: str) -> float:
    d = ollama_call(model, MAX_TOKENS)
    gt  = d.get("eval_count", 0)
    gns = d.get("eval_duration", 1)
    return gt / (gns / 1e9) if gns > 0 else 0.0

# ── Plain MLX — clock starts at first emitted token ──────────────────────────

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

# ── DDTree — generation_tokens / generation_time (prefill excluded) ───────────

def bench_ddtree(target, tokenizer, draft, stop_ids) -> float:
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

results = []

# Phase 1: Ollama — run and unload each model before loading MLX
print("=" * 62, flush=True)
print("  PHASE 1: Ollama models (unloaded before MLX loads)", flush=True)
print("=" * 62, flush=True)

for label, ollama_model in OLLAMA_MODELS:
    med = run_suite(
        f"Ollama  {label}",
        lambda m=ollama_model: bench_ollama(m),
    )
    results.append((f"Ollama  {label}", med))
    print(f"\n  Unloading {ollama_model}...", end=" ", flush=True)
    ollama_unload(ollama_model)
    print("done", flush=True)

# Phase 2: MLX — safe to load now that Ollama is unloaded
# Each model loads, benchmarks, then is deleted before the next loads.
print(f"\n{'='*62}", flush=True)
print("  PHASE 2: MLX models (loaded one at a time)", flush=True)
print(f"{'='*62}", flush=True)

from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
import gc

for label, mlx_ref, explicit_draft in MLX_MODELS:
    print(f"\nLoading {mlx_ref}...", flush=True)
    target, tok, draft, _ = load_runtime_components(model_ref=mlx_ref, draft_ref=explicit_draft)
    stop = get_stop_token_ids(tok)
    print("Loaded.", flush=True)

    results.append((
        f"Plain mlx_lm  {label}",
        run_suite(
            f"Plain mlx_lm  {label}",
            lambda m=target, t=tok: bench_plain(m, t),
        ),
    ))

    results.append((
        f"DFlash+DDTree  {label}",
        run_suite(
            f"DFlash+DDTree  {label}",
            lambda m=target, t=tok, d=draft, s=stop: bench_ddtree(m, t, d, s),
        ),
    ))

    # Free before loading next model
    del target, tok, draft, stop
    gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
baseline = results[0][1]
print(f"\n\n{'='*62}")
print(f"  COMPARISON SUMMARY")
print(f"  {WARMUPS} warmups · {RUNS} runs · median · output tok/s · {MAX_TOKENS} max tok")
print(f"{'='*62}")
print(f"  {'Method':<32}  {'tok/s':>7}  {'vs #1':>8}")
print(f"  {'-'*32}  {'-'*7}  {'-'*8}")
for label, tps in results:
    ratio = tps / baseline
    print(f"  {label:<32}  {tps:7.1f}  {ratio:7.2f}×")
print()
