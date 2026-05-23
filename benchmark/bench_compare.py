#!/usr/bin/env python3
"""
Benchmark: multiple Ollama models vs plain mlx_lm vs DFlash+DDTree.
Multi-prompt run (code, prose, JSON).
TTFT + decode tok/s, cool-down discipline, JSON output.

Timing convention:
  - Ollama:     TTFT from request send to first chunk; decode = eval_count / eval_duration
  - Plain MLX:  TTFT from before stream_generate to first emitted token; decode starts after first token
  - DDTree:     TTFT = result["prefill_us"] / 1000.0 (ms); decode from elapsed_us - prefill_us

Memory safety: Ollama models are unloaded before MLX is loaded.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_compare.py
"""
import os, time, json, urllib.request, statistics, gc
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    CODING_PROMPTS, HOST, cooldown,
    PRE_BENCH_COOLDOWN_S, INTER_PROMPT_COOLDOWN_S, INTER_CONFIG_COOLDOWN_S, POST_BENCH_COOLDOWN_S,
    make_greedy_sampler, write_results, get_timestamp_iso8601,
)

WARMUPS = 2
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200

# BENCH_PHASE gates which phases run: "all" (default), "ollama", "mlx".
BENCH_PHASE = os.environ.get("BENCH_PHASE", "all").lower()
assert BENCH_PHASE in ("all", "ollama", "mlx"), f"BENCH_PHASE must be all|ollama|mlx (got {BENCH_PHASE!r})"
RUN_OLLAMA = BENCH_PHASE in ("all", "ollama")
RUN_MLX = BENCH_PHASE in ("all", "mlx")

# Labels include model size, architecture, and quant type for clarity.
# Ollama: quant is part of the tag name (q4 = Q4_K_M, :27b = default Ollama quant)
OLLAMA_MODELS = [
    ("Qwen3.6-27B-dense  [GGUF-Q4_K_M]",            "qwen3.6:27b"),
    ("Qwen3.6-35B-MoE    [GGUF-Q4_K_M]",            "qwen3.6:35b"),
    ("Qwen3.6-35B-MoE    [GGUF-Q4_K_M-uncensored]", "qwen3.6-uncensored:35b-q4"),
]

# Each entry: (label, mlx_model_ref, draft_ref)
# draft_ref=None → auto-resolve from dflash registry; set explicitly when not in registry.
MLX_MODELS = [
    ("Qwen3.6-35B-MoE    [MLX-int4-DWQ]", "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ", "z-lab/Qwen3.6-35B-A3B-DFlash"),
]

# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_call_streaming(model: str, prompt: str, num_predict: int) -> tuple[float, dict]:
    """Stream request to Ollama, return (ttft_s_wallclock, final_done_chunk).

    TTFT is measured wall-clock from request send to the first response chunk
    (matches plain-MLX TTFT semantics). The final chunk (done=true) carries
    eval_count and eval_duration which we use for decode tok/s.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict, "temperature": 0, "seed": 42},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_s = None
    final = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            if not raw.strip():
                continue
            if ttft_s is None:
                ttft_s = time.perf_counter() - t0
            chunk = json.loads(raw)
            if chunk.get("done"):
                final = chunk
    if ttft_s is None:
        ttft_s = time.perf_counter() - t0
    return ttft_s, final

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

def bench_ollama(model: str, prompt: str) -> tuple[float, float]:
    """Run Ollama, return (ttft_ms, decode_tps).

    TTFT: wall time from request send to first streamed chunk (matches plain-MLX semantics).
    Decode: eval_count / eval_duration (Ollama's own decode-only measurement).
    """
    ttft_s, d = ollama_call_streaming(model, prompt, MAX_TOKENS)
    gt = d.get("eval_count", 0)
    gns = d.get("eval_duration", 1)
    decode_tps = gt / (gns / 1e9) if gns > 0 else 0.0
    ttft_ms = ttft_s * 1000.0
    return ttft_ms, decode_tps

# ── Plain MLX — clock starts before stream_generate ───────────────────────────

def bench_plain(model, tokenizer, prompt: str) -> tuple[float, float]:
    """Run plain MLX, return (ttft_ms, decode_tps).

    TTFT: wall time from before stream_generate to first yielded token.
    Decode: tokens after first / elapsed time after first token.
    """
    from mlx_lm import stream_generate
    sampler = make_greedy_sampler()

    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t0_stream = time.perf_counter()
    t0_first = None
    count = 0
    for _ in stream_generate(model, tokenizer, prompt=prompt_str, max_tokens=MAX_TOKENS, sampler=sampler):
        if t0_first is None:
            t0_first = time.perf_counter()
            ttft_ms = (t0_first - t0_stream) * 1000.0
        count += 1

    if t0_first is None:
        return 0.0, 0.0

    elapsed_after_first = time.perf_counter() - t0_first
    decode_tps = (count - 1) / elapsed_after_first if elapsed_after_first > 0 else 0.0
    return ttft_ms, decode_tps

# ── DDTree — prefill_us + generation stats ────────────────────────────────────

def bench_ddtree(target, tokenizer, draft, stop_ids, prompt: str) -> tuple[float, float, float]:
    """Run DDTree, return (ttft_ms, decode_tps, avg_acceptance).

    TTFT: result["prefill_us"] / 1000.0 (measures prefill-to-first-token, same operational definition).
    Decode: generation_tokens / generation_time_s, where generation_time_s = (elapsed_us - prefill_us) / 1e6.
    Acceptance: average draft acceptance rate.

    NOTE: generate_ddtree_once does NOT accept a temperature/sampler argument; it always uses
    greedy decoding internally. Fair comparison with plain MLX (also greedy via make_greedy_sampler()).
    """
    from ddtree_mlx.runtime import generate_ddtree_once

    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=target, draft_model=draft, tokenizer=tokenizer,
        prompt_tokens=prompt_tokens, max_new_tokens=MAX_TOKENS,
        tree_budget=4, stop_token_ids=stop_ids,
    )

    ttft_ms = result["prefill_us"] / 1000.0
    gen_time_s = (result["elapsed_us"] - result["prefill_us"]) / 1e6
    decode_tps = result["generation_tokens"] / gen_time_s if gen_time_s > 0 else 0.0
    avg_acceptance = result.get("avg_acceptance", float("nan"))

    return ttft_ms, decode_tps, avg_acceptance

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_suite(label: str, bench_fn) -> dict:
    """Run warmups and timed runs, return dict with ttft_ms_runs and decode_tps_runs."""
    print(f"\n{'='*70}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*70}", flush=True)
    for w in range(1, WARMUPS + 1):
        print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
        bench_fn()
        print("done", flush=True)

    ttft_runs = []
    decode_runs = []
    for i in range(1, RUNS_PER_PROMPT + 1):
        print(f"  [run {i}/{RUNS_PER_PROMPT}] ", end="", flush=True)
        result = bench_fn()
        if isinstance(result, tuple):
            if len(result) == 2:
                ttft_ms, decode_tps = result
                ttft_runs.append(ttft_ms)
                decode_runs.append(decode_tps)
                print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s", flush=True)
            else:  # len == 3, DDTree with acceptance
                ttft_ms, decode_tps, acc = result
                ttft_runs.append(ttft_ms)
                decode_runs.append(decode_tps)
                acc_str = f"  accept {acc:.2f} tok/cycle" if acc == acc else ""
                print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s{acc_str}", flush=True)

    return {
        "ttft_ms_runs": ttft_runs,
        "decode_tps_runs": decode_runs,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

results_by_method_model = {}  # (method, model_label, model_ref) → {prompt_name → metrics}

if RUN_OLLAMA:
  print("=" * 70, flush=True)
  print("  PHASE 1: Ollama models", flush=True)
  print("=" * 70, flush=True)

  cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before Ollama phase")

  for label, ollama_model in OLLAMA_MODELS:
    print(f"\nBenchmarking Ollama {label}", flush=True)
    cooldown(INTER_CONFIG_COOLDOWN_S, "between configs: thermal reset before new model")

    method = "ollama"
    model_ref = ollama_model
    results_this_model = {}

    for prompt_name, prompt_text in CODING_PROMPTS:
        suite_result = run_suite(
            f"Ollama {label} [{prompt_name}]",
            lambda p=ollama_model, pr=prompt_text: bench_ollama(p, pr),
        )
        results_this_model[prompt_name] = {
            "ttft_ms_runs": suite_result["ttft_ms_runs"],
            "decode_tps_runs": suite_result["decode_tps_runs"],
            "avg_acceptance": None,  # N/A for Ollama
            "ttft_ms_median": statistics.median(suite_result["ttft_ms_runs"]),
            "decode_tps_median": statistics.median(suite_result["decode_tps_runs"]),
        }
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {method}/{label}")

    results_by_method_model[(method, label, model_ref)] = results_this_model

    print(f"\nUnloading {ollama_model}...", end=" ", flush=True)
    ollama_unload(ollama_model)
    print("done", flush=True)

if RUN_MLX:
  # Phase 2: MLX — safe to load now that Ollama is unloaded
  print(f"\n{'='*70}", flush=True)
  print("  PHASE 2: MLX models (loaded one at a time)", flush=True)
  print(f"{'='*70}", flush=True)

  cooldown(INTER_CONFIG_COOLDOWN_S, "before MLX phase: reset after Ollama")

  from dflash_mlx.generate import load_runtime_components, get_stop_token_ids

  for label, mlx_ref, explicit_draft in MLX_MODELS:
    print(f"\nLoading {mlx_ref}...", flush=True)
    target, tok, draft, _ = load_runtime_components(model_ref=mlx_ref, draft_ref=explicit_draft)
    stop = get_stop_token_ids(tok)
    print("Loaded.", flush=True)

    # Plain MLX
    method = "plain-mlx"
    results_plain = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        suite_result = run_suite(
            f"Plain mlx_lm {label} [{prompt_name}]",
            lambda m=target, t=tok, pr=prompt_text: bench_plain(m, t, pr),
        )
        ttft_runs = suite_result["ttft_ms_runs"]
        decode_runs = suite_result["decode_tps_runs"]
        results_plain[prompt_name] = {
            "ttft_ms_runs": ttft_runs,
            "decode_tps_runs": decode_runs,
            "avg_acceptance": None,
            "ttft_ms_median": statistics.median(ttft_runs),
            "decode_tps_median": statistics.median(decode_runs),
        }
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {method}/{label}")

    results_by_method_model[(method, label, mlx_ref)] = results_plain

    # DDTree
    method = "ddtree-mlx"
    results_ddtree = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        suite_result_raw = []
        print(f"\n{'='*70}", flush=True)
        print(f"  DFlash+DDTree {label} [{prompt_name}]", flush=True)
        print(f"{'='*70}", flush=True)
        for w in range(1, WARMUPS + 1):
            print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
            bench_ddtree(target, tok, draft, stop, prompt_text)
            print("done", flush=True)

        ttft_runs = []
        decode_runs = []
        acc_runs = []
        for i in range(1, RUNS_PER_PROMPT + 1):
            print(f"  [run {i}/{RUNS_PER_PROMPT}] ", end="", flush=True)
            ttft_ms, decode_tps, acc = bench_ddtree(target, tok, draft, stop, prompt_text)
            ttft_runs.append(ttft_ms)
            decode_runs.append(decode_tps)
            if acc == acc:  # not NaN
                acc_runs.append(acc)
            acc_str = f"  accept {acc:.2f} tok/cycle" if acc == acc else ""
            print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s{acc_str}", flush=True)

        avg_acc = sum(acc_runs) / len(acc_runs) if acc_runs else None
        results_ddtree[prompt_name] = {
            "ttft_ms_runs": ttft_runs,
            "decode_tps_runs": decode_runs,
            "avg_acceptance": avg_acc,
            "ttft_ms_median": statistics.median(ttft_runs),
            "decode_tps_median": statistics.median(decode_runs),
        }
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {method}/{label}")

    results_by_method_model[(method, label, mlx_ref)] = results_ddtree

    # Free before loading next model
    del target, tok, draft, stop
    gc.collect()

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

# ── Write JSON outputs ─────────────────────────────────────────────────────────

ts = get_timestamp_iso8601()

for (method, label, model_ref), results_dict in results_by_method_model.items():
    payload = {
        "ts": ts,
        "host": HOST,
        "method": method,
        "model_label": label,
        "model_ref": model_ref,
        "drafter_ref": None,
        "tree_budget": None,
        "sampling": {"temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS},
        "warmups": WARMUPS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "prompt_set": "coding",
        "results": [
            {
                "prompt": prompt_name,
                **results_dict[prompt_name],
            }
            for prompt_name, _ in CODING_PROMPTS
        ],
    }
    path = write_results(payload)
    print(f"Wrote {method} {label} results to {path}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print(f"  COMPARISON SUMMARY")
print(f"  {WARMUPS} warmups · {RUNS_PER_PROMPT} runs per prompt · median · {MAX_TOKENS} max tok")
print(f"{'='*70}")

for (method, label, _), results_dict in sorted(results_by_method_model.items()):
    print(f"\n  {method} {label}")
    print(f"  {'-'*60}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}")
    for prompt_name, _ in CODING_PROMPTS:
        ttft_med = results_dict[prompt_name]["ttft_ms_median"]
        decode_med = results_dict[prompt_name]["decode_tps_median"]
        print(f"  {prompt_name:<12}  {ttft_med:>10.1f}  {decode_med:>15.1f}")

print()
