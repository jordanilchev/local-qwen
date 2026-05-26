#!/usr/bin/env python3
"""
Sweep DDTree tree_budget on Qwen3.6-35B-A3B (MLX-int4-DWQ) with the DFlash drafter.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_tree_budget_sweep
    BUDGETS_TO_RUN=3 HF_HOME=... .venv/bin/python -m benchmark.bench_tree_budget_sweep
"""
import os

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    CODING_PROMPTS,
    DEFAULT_DRAFT_BY_TARGET,
    DEFAULT_SAMPLING,
    INTER_CONFIG_COOLDOWN_S,
    INTER_PROMPT_COOLDOWN_S,
    POST_BENCH_COOLDOWN_S,
    PRE_BENCH_COOLDOWN_S,
    PROMPT_SET,
    RUNS_PER_PROMPT,
    WARMUPS,
    bench_ddtree,
    cooldown,
    get_timestamp_iso8601,
    make_results_payload,
    run_prompt_suite,
    write_results,
)

TARGET = "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ"
DRAFT = os.environ.get("DRAFT", DEFAULT_DRAFT_BY_TARGET[TARGET])
_budgets_env = os.environ.get("BUDGETS_TO_RUN", "2,3,4,5,6")
BUDGETS = [int(b) for b in _budgets_env.split(",") if b.strip()]
TARGET_LABEL = "Qwen3.6-35B-MoE    [MLX-int4-DWQ]"


def run_budget_suite(target, tok, draft, stop, budget, prompt_name, prompt_text):
    accept_runs: list[float] = []

    def timed():
        metrics, acc = bench_ddtree(
            target, tok, draft, stop, prompt_text, tree_budget=budget
        )
        if acc == acc:
            accept_runs.append(acc)
        return metrics

    suite = run_prompt_suite(
        f"budget={budget} [{prompt_name}]",
        bench_fn=timed,
        warmup_fn=lambda wp: bench_ddtree(
            target, tok, draft, stop, wp, tree_budget=budget
        )[0],
    )
    suite["avg_acceptance"] = (
        sum(accept_runs) / len(accept_runs) if accept_runs else None
    )
    return suite


print(f"Loading {TARGET} + {DRAFT} (~21.6 GB)...", flush=True)
from benchmark._lib import load_ddtree_runtime

target, tokenizer, draft, loaded_draft_ref, stop_ids = load_ddtree_runtime(TARGET, DRAFT)
print("Loaded.\n", flush=True)

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool")

summary: dict[int, dict] = {}
ts = get_timestamp_iso8601()

for budget in BUDGETS:
    print(f"\n{'='*70}", flush=True)
    print(f"  tree_budget = {budget}", flush=True)
    print(f"{'='*70}", flush=True)

    budget_results = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        budget_results[prompt_name] = run_budget_suite(
            target, tokenizer, draft, stop_ids, budget, prompt_name, prompt_text
        )
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of budget={budget}")

    summary[budget] = budget_results

    payload = make_results_payload(
        ts=ts,
        method="ddtree-budget-sweep",
        model_label=TARGET_LABEL,
        model_ref=TARGET,
        drafter_ref=DRAFT,
        tree_budget=budget,
        results=[
            {"prompt": prompt_name, **budget_results[prompt_name]}
            for prompt_name, _ in CODING_PROMPTS
        ],
        sampling=dict(DEFAULT_SAMPLING),
        warmups=WARMUPS,
        runs_per_prompt=RUNS_PER_PROMPT,
        prompt_set=PROMPT_SET,
    )
    path = write_results(payload)
    print(f"Wrote budget={budget} results to {path}", flush=True)

    if budget != BUDGETS[-1]:
        cooldown(INTER_CONFIG_COOLDOWN_S, "between budgets")

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

print(f"\n{'='*70}")
print(f"  SUMMARY  (tree_budget sweep · {WARMUPS} warmups · {RUNS_PER_PROMPT} runs/prompt)")
print(f"  Model: {TARGET_LABEL}")
print(f"  Draft: {DRAFT}")
print(f"{'='*70}\n")

baseline_budget = 3 if 3 in summary else BUDGETS[0]
baseline_decode = summary[baseline_budget][CODING_PROMPTS[0][0]]["decode_tps_median"]

for budget in BUDGETS:
    print(f"  budget={budget}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}  {'vs b=3':>10}  {'tok/cycle':>10}")
    for prompt_name, _ in CODING_PROMPTS:
        m = summary[budget][prompt_name]
        ratio = m["decode_tps_median"] / baseline_decode if baseline_decode > 0 else 1.0
        acc = m.get("avg_acceptance")
        acc_str = f"{acc:.2f}" if acc is not None else "  —"
        print(
            f"  {prompt_name:<12}  {m['ttft_ms_median']:>10.1f}"
            f"  {m['decode_tps_median']:>15.1f}  {ratio:>9.2f}×  {acc_str:>10}"
        )
    print()

best_budget = max(
    BUDGETS,
    key=lambda b: sum(summary[b][p]["decode_tps_median"] for p, _ in CODING_PROMPTS),
)
best_avg = sum(summary[best_budget][p]["decode_tps_median"] for p, _ in CODING_PROMPTS) / len(
    CODING_PROMPTS
)
print(f"  Best overall (avg across prompts): budget={best_budget} at {best_avg:.1f} tok/s")
print()
