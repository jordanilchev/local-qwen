#!/usr/bin/env python3
"""
Sweep DDTree tree_budget on Qwen3.6-35B-A3B (MLX-int4-DWQ) with the DFlash drafter.
Multi-prompt run (code, prose, JSON).
TTFT + decode tok/s, cool-down discipline, JSON output.

Methodology matches the headline benchmark: 2 warmups + 5 runs per budget per prompt, median tok/s.
Reports TTFT and tok/s, vs-budget=4 ratio, and average draft acceptance per budget.

Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_tree_budget_sweep.py
"""
import os, time, statistics
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    CODING_PROMPTS, HOST, cooldown,
    PRE_BENCH_COOLDOWN_S, INTER_PROMPT_COOLDOWN_S, INTER_CONFIG_COOLDOWN_S, POST_BENCH_COOLDOWN_S,
    write_results, get_timestamp_iso8601,
)

TARGET = "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ"
DRAFT = "z-lab/Qwen3.6-35B-A3B-DFlash"
WARMUPS = 2
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200
# BUDGETS_TO_RUN env var (comma-separated) selects which budgets to bench in this process.
# Default: all five. For OOM-safe overnight runs, set BUDGETS_TO_RUN to a single value
# and orchestrate via the shell loop in benchmark/run_sweep_overnight.sh — each
# invocation gets a fresh Python process and fresh model load.
_budgets_env = os.environ.get("BUDGETS_TO_RUN", "2,3,4,5,6")
BUDGETS = [int(b) for b in _budgets_env.split(",") if b.strip()]

TARGET_LABEL = "Qwen3.6-35B-MoE    [MLX-int4-DWQ]"


def run_one(target, tokenizer, draft, stop_ids, budget, prompt: str) -> tuple[float, float, float]:
    """Run DDTree with given budget on one prompt, return (ttft_ms, decode_tps, avg_acceptance).

    TTFT: result["prefill_us"] / 1000.0 (ms).
    Decode: generation_tokens / generation_time_s, where generation_time_s = (elapsed_us - prefill_us) / 1e6.
    Acceptance: average draft acceptance rate.

    NOTE: generate_ddtree_once always uses greedy decoding (no temperature/sampler parameter);
    fair comparison since sweep is against fixed DDTree method only.
    """
    from ddtree_mlx.runtime import generate_ddtree_once

    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=target, draft_model=draft, tokenizer=tokenizer,
        prompt_tokens=prompt_tokens, max_new_tokens=MAX_TOKENS,
        tree_budget=budget, stop_token_ids=stop_ids,
    )

    ttft_ms = result["prefill_us"] / 1000.0
    gen_time_s = (result["elapsed_us"] - result["prefill_us"]) / 1e6
    decode_tps = result["generation_tokens"] / gen_time_s if gen_time_s > 0 else 0.0
    avg_acceptance = result.get("avg_acceptance", float("nan"))

    return ttft_ms, decode_tps, avg_acceptance


print(f"Loading {TARGET} + {DRAFT} (~21.6 GB)...", flush=True)
from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
target, tokenizer, draft, _ = load_runtime_components(model_ref=TARGET, draft_ref=DRAFT)
stop_ids = get_stop_token_ids(tokenizer)
print("Loaded.\n", flush=True)

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool")

# summary[(budget, prompt)] = {"ttft_ms_runs": [...], "decode_tps_runs": [...], ...}
summary = {}
ts = get_timestamp_iso8601()

for budget in BUDGETS:
    print(f"\n{'='*70}", flush=True)
    print(f"  tree_budget = {budget}", flush=True)
    print(f"{'='*70}", flush=True)

    budget_results = {}

    for prompt_name, prompt_text in CODING_PROMPTS:
        print(f"\n  Prompt: {prompt_name}", flush=True)
        print(f"  {'-'*66}", flush=True)

        for w in range(1, WARMUPS + 1):
            print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
            run_one(target, tokenizer, draft, stop_ids, budget, prompt_text)
            print("done", flush=True)

        ttft_runs = []
        decode_runs = []
        acc_runs = []

        for i in range(1, RUNS_PER_PROMPT + 1):
            print(f"  [run {i}/{RUNS_PER_PROMPT}] ", end="", flush=True)
            ttft_ms, decode_tps, acc = run_one(target, tokenizer, draft, stop_ids, budget, prompt_text)
            ttft_runs.append(ttft_ms)
            decode_runs.append(decode_tps)
            if acc == acc:  # not NaN
                acc_runs.append(acc)
            acc_str = f"  accept {acc:.2f} tok/cycle" if acc == acc else ""
            print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s{acc_str}", flush=True)

        avg_acc = sum(acc_runs) / len(acc_runs) if acc_runs else None
        budget_results[prompt_name] = {
            "ttft_ms_runs": ttft_runs,
            "decode_tps_runs": decode_runs,
            "avg_acceptance": avg_acc,
            "ttft_ms_median": statistics.median(ttft_runs),
            "decode_tps_median": statistics.median(decode_runs),
        }
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of budget={budget}")

    summary[budget] = budget_results

    # Checkpoint: write JSON for this budget immediately so a later crash doesn't lose it.
    payload = {
        "ts": ts,
        "host": HOST,
        "method": "ddtree-budget-sweep",
        "model_label": TARGET_LABEL,
        "model_ref": TARGET,
        "drafter_ref": DRAFT,
        "tree_budget": budget,
        "sampling": {"temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS},
        "warmups": WARMUPS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "prompt_set": "coding",
        "results": [
            {"prompt": prompt_name, **budget_results[prompt_name]}
            for prompt_name, _ in CODING_PROMPTS
        ],
    }
    path = write_results(payload)
    print(f"Wrote budget={budget} results to {path}", flush=True)

    if budget != BUDGETS[-1]:
        cooldown(INTER_CONFIG_COOLDOWN_S, "between budgets")

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  SUMMARY  ({WARMUPS} warmups · {RUNS_PER_PROMPT} runs per prompt · median · {MAX_TOKENS} max tok)")
print(f"  Model: {TARGET_LABEL}")
print(f"  Draft: {DRAFT}")
print(f"{'='*70}\n")

baseline_budget = 4 if 4 in summary else BUDGETS[0]
baseline_decode = summary[baseline_budget][CODING_PROMPTS[0][0]]["decode_tps_median"]

for budget in BUDGETS:
    print(f"  budget={budget}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}  {'vs b=4':>10}  {'tok/cycle':>10}")
    for prompt_name, _ in CODING_PROMPTS:
        metrics = summary[budget][prompt_name]
        ttft_med = metrics["ttft_ms_median"]
        decode_med = metrics["decode_tps_median"]
        ratio = decode_med / baseline_decode if baseline_decode > 0 else 1.0
        acc_str = f"{metrics['avg_acceptance']:.2f}" if metrics["avg_acceptance"] is not None else "  —"
        print(f"  {prompt_name:<12}  {ttft_med:>10.1f}  {decode_med:>15.1f}  {ratio:>9.2f}×  {acc_str:>10}")
    print()

best_budget = max(BUDGETS, key=lambda b: max(summary[b][p]["decode_tps_median"] for p, _ in CODING_PROMPTS))
best_decode = max(summary[best_budget][p]["decode_tps_median"] for p, _ in CODING_PROMPTS)
print(f"  Best overall: budget={best_budget} at {best_decode:.1f} tok/s")
print()
