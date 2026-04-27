#!/usr/bin/env python3
"""
Benchmark: plain mlx_lm vs DFlash+DDTree on configurable Qwen model.
Multi-prompt run (code, prose, JSON).
TTFT + decode tok/s, cool-down discipline, JSON output.

Timing convention:
  - Plain MLX:  TTFT from before stream_generate to first emitted token; decode after first token
  - DDTree:     TTFT = result["prefill_us"] / 1000.0 (ms); decode from elapsed_us - prefill_us

Environment variables (optional overrides):
  TARGET: Model ref to use (default: mlx-community/Qwen3.5-27B-4bit)
  DRAFT:  Draft model ref (default: auto-resolve via dflash registry)

Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_ddtree.py
       TARGET=mlx-community/Qwen3.6-35B-A3B-4bit-DWQ DRAFT=z-lab/Qwen3.6-35B-A3B-DFlash .venv/bin/python benchmark/bench_ddtree.py
"""
import os, time, statistics, gc
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    PROMPTS, HOST, cooldown,
    PRE_BENCH_COOLDOWN_S, INTER_PROMPT_COOLDOWN_S, INTER_CONFIG_COOLDOWN_S, POST_BENCH_COOLDOWN_S,
    make_greedy_sampler, write_results, get_timestamp_iso8601,
)

TARGET = os.environ.get("TARGET", "mlx-community/Qwen3.5-27B-4bit")
DRAFT = os.environ.get("DRAFT", None)
WARMUPS = 2
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200
TREE_BUDGET = 4

# Derive label from TARGET; map known refs to friendly names.
_LABEL_MAP = {
    "mlx-community/Qwen3.5-27B-4bit": "Qwen3.5-27B-dense  [MLX-int4]",
    "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ": "Qwen3.6-35B-MoE    [MLX-int4-DWQ]",
}
TARGET_LABEL = _LABEL_MAP.get(TARGET, TARGET)


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


def bench_ddtree(target_model, tokenizer, draft_model, stop_ids, prompt: str) -> tuple[float, float, float]:
    """Run DDTree, return (ttft_ms, decode_tps, avg_acceptance).

    TTFT: result["prefill_us"] / 1000.0 (ms); measures prefill-to-first-token.
    Decode: generation_tokens / generation_time_s, where generation_time_s = (elapsed_us - prefill_us) / 1e6.
    Acceptance: average draft acceptance rate.

    NOTE: generate_ddtree_once does NOT accept a temperature/sampler argument; it always uses
    greedy decoding internally. This is acceptable for fair comparison with plain MLX (also greedy
    via sampler=make_greedy_sampler()).
    """
    from ddtree_mlx.runtime import generate_ddtree_once

    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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
        stop_token_ids=stop_ids,
    )

    ttft_ms = result["prefill_us"] / 1000.0
    gen_time_s = (result["elapsed_us"] - result["prefill_us"]) / 1e6
    decode_tps = result["generation_tokens"] / gen_time_s if gen_time_s > 0 else 0.0
    avg_acceptance = result.get("avg_acceptance", float("nan"))

    return ttft_ms, decode_tps, avg_acceptance


print(f"Loading {TARGET} + {DRAFT or 'auto'}...", flush=True)
from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
target_model, tokenizer, draft_model, _ = load_runtime_components(
    model_ref=TARGET, draft_ref=DRAFT
)
stop_ids = get_stop_token_ids(tokenizer)
print("Models loaded.\n", flush=True)

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool")

results_by_method = {}

for method_name, bench_fn in [
    ("plain-mlx", lambda p: bench_plain(target_model, tokenizer, p)),
    ("ddtree-mlx", lambda p: bench_ddtree(target_model, tokenizer, draft_model, stop_ids, p)),
]:
    print(f"\n{'='*70}", flush=True)
    print(f"  {method_name} · {TARGET_LABEL}", flush=True)
    print(f"{'='*70}", flush=True)

    results_this_method = {}

    for prompt_name, prompt_text in PROMPTS:
        print(f"\n  Prompt: {prompt_name}", flush=True)
        print(f"  {'-'*66}", flush=True)

        for w in range(1, WARMUPS + 1):
            print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
            bench_fn(prompt_text)
            print("done", flush=True)

        ttft_runs = []
        decode_runs = []
        acc_runs = []

        for i in range(1, RUNS_PER_PROMPT + 1):
            print(f"  [run {i}/{RUNS_PER_PROMPT}] ", end="", flush=True)
            result = bench_fn(prompt_text)
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
                    if acc == acc:  # not NaN
                        acc_runs.append(acc)
                    acc_str = f"  accept {acc:.2f} tok/cycle" if acc == acc else ""
                    print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s{acc_str}", flush=True)

        avg_acc = sum(acc_runs) / len(acc_runs) if acc_runs else None
        results_this_method[prompt_name] = {
            "ttft_ms_runs": ttft_runs,
            "decode_tps_runs": decode_runs,
            "avg_acceptance": avg_acc,
            "ttft_ms_median": statistics.median(ttft_runs),
            "decode_tps_median": statistics.median(decode_runs),
        }
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {method_name}")

    results_by_method[method_name] = results_this_method
    cooldown(INTER_CONFIG_COOLDOWN_S, f"between methods")

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

# ── Write JSON outputs ─────────────────────────────────────────────────────────

ts = get_timestamp_iso8601()

for method, results_dict in results_by_method.items():
    payload = {
        "ts": ts,
        "host": HOST,
        "method": method,
        "model_label": TARGET_LABEL,
        "model_ref": TARGET,
        "drafter_ref": DRAFT,
        "tree_budget": TREE_BUDGET if method == "ddtree-mlx" else None,
        "sampling": {"temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS},
        "warmups": WARMUPS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "results": [
            {
                "prompt": prompt_name,
                **results_dict[prompt_name],
            }
            for prompt_name, _ in PROMPTS
        ],
    }
    path = write_results(payload)
    print(f"Wrote {method} results to {path}", flush=True)

# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print(f"  SUMMARY  ({WARMUPS} warmups · {RUNS_PER_PROMPT} runs per prompt · median · {MAX_TOKENS} max tok)")
print(f"{'='*70}")
print(f"  Model: {TARGET_LABEL}")
if DRAFT:
    print(f"  Draft: {DRAFT}")
print(f"{'='*70}")

baseline_decode = None
for method in ["plain-mlx", "ddtree-mlx"]:
    if method not in results_by_method:
        continue
    if baseline_decode is None:
        baseline_decode = results_by_method[method][PROMPTS[0][0]]["decode_tps_median"]

    print(f"\n  {method}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}  {'vs baseline':>12}")
    for prompt_name, _ in PROMPTS:
        metrics = results_by_method[method][prompt_name]
        ttft_med = metrics["ttft_ms_median"]
        decode_med = metrics["decode_tps_median"]
        speedup = decode_med / baseline_decode if baseline_decode > 0 else 1.0
        acc_str = f"  accept {metrics['avg_acceptance']:.2f} tok/cycle" if metrics["avg_acceptance"] is not None else ""
        print(f"  {prompt_name:<12}  {ttft_med:>10.1f}  {decode_med:>15.1f}  {speedup:>11.2f}×{acc_str}")

print()
