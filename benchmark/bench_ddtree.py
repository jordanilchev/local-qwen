#!/usr/bin/env python3
"""
Benchmark: plain mlx_lm vs DFlash+DDTree on configurable Qwen model.

Environment variables:
  TARGET  Model ref (default: mlx-community/Qwen3.6-35B-A3B-4bit-DWQ)
  DRAFT   Draft model ref (default: auto via resolve_draft_ref)
  TREE_BUDGET  DDTree budget (default: 3, best on M4 per sweep)

Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_ddtree
"""
import os

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    CODING_PROMPTS,
    DEFAULT_SAMPLING,
    DEFAULT_TREE_BUDGET,
    INTER_CONFIG_COOLDOWN_S,
    INTER_PROMPT_COOLDOWN_S,
    POST_BENCH_COOLDOWN_S,
    PRE_BENCH_COOLDOWN_S,
    PROMPT_SET,
    RUNS_PER_PROMPT,
    WARMUPS,
    bench_ddtree,
    bench_mlx_plain,
    cooldown,
    get_timestamp_iso8601,
    load_ddtree_runtime,
    make_results_payload,
    resolve_draft_ref,
    run_prompt_suite,
    write_results,
)

TARGET = os.environ.get("TARGET", "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ")
DRAFT = os.environ.get("DRAFT")
TREE_BUDGET = int(os.environ.get("TREE_BUDGET", str(DEFAULT_TREE_BUDGET)))

_LABEL_MAP = {
    "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ": "Qwen3.6-35B-MoE    [MLX-int4-DWQ]",
}
TARGET_LABEL = _LABEL_MAP.get(TARGET, TARGET)


def run_ddtree_suite(label, target, tok, draft, stop, prompt_text):
    accept_runs: list[float] = []

    def timed():
        metrics, acc = bench_ddtree(
            target, tok, draft, stop, prompt_text, tree_budget=TREE_BUDGET
        )
        if acc == acc:
            accept_runs.append(acc)
        return metrics

    suite = run_prompt_suite(
        label,
        bench_fn=timed,
        warmup_fn=lambda wp: bench_ddtree(
            target, tok, draft, stop, wp, tree_budget=TREE_BUDGET
        )[0],
    )
    suite["avg_acceptance"] = (
        sum(accept_runs) / len(accept_runs) if accept_runs else None
    )
    return suite


def write_method_results(ts: str, method: str, results_dict: dict, loaded_draft_ref: str | None):
    payload = make_results_payload(
        ts=ts,
        method=method,
        model_label=TARGET_LABEL,
        model_ref=TARGET,
        drafter_ref=loaded_draft_ref if method == "ddtree-mlx" else None,
        tree_budget=TREE_BUDGET if method == "ddtree-mlx" else None,
        results=[
            {"prompt": prompt_name, **results_dict[prompt_name]}
            for prompt_name, _ in CODING_PROMPTS
        ],
        sampling=dict(DEFAULT_SAMPLING),
        warmups=WARMUPS,
        runs_per_prompt=RUNS_PER_PROMPT,
        prompt_set=PROMPT_SET,
    )
    path = write_results(payload)
    print(f"Wrote {method} results to {path}", flush=True)


resolved_draft = resolve_draft_ref(TARGET, DRAFT)
print(f"Loading {TARGET} + drafter {resolved_draft}...", flush=True)
target_model, tokenizer, draft_model, loaded_draft_ref, stop_ids = load_ddtree_runtime(
    TARGET, DRAFT
)
print(f"Loaded drafter: {loaded_draft_ref}\n", flush=True)

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool")

ts = get_timestamp_iso8601()
results_by_method: dict[str, dict] = {}

for method_name, run_fn in [
    (
        "plain-mlx",
        lambda pname, ptext: run_prompt_suite(
            f"plain-mlx · {TARGET_LABEL} [{pname}]",
            bench_fn=lambda pt=ptext: bench_mlx_plain(target_model, tokenizer, pt),
            warmup_fn=lambda wp: bench_mlx_plain(target_model, tokenizer, wp),
        ),
    ),
    (
        "ddtree-mlx",
        lambda pname, ptext: run_ddtree_suite(
            f"ddtree-mlx · {TARGET_LABEL} [{pname}]",
            target_model,
            tokenizer,
            draft_model,
            stop_ids,
            ptext,
        ),
    ),
]:
    print(f"\n{'='*70}", flush=True)
    print(f"  {method_name} · {TARGET_LABEL}", flush=True)
    print(f"{'='*70}", flush=True)

    results_this_method = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        results_this_method[prompt_name] = {
            "avg_acceptance": None,
            **run_fn(prompt_name, prompt_text),
        }
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {method_name}")

    results_by_method[method_name] = results_this_method
    write_method_results(ts, method_name, results_this_method, loaded_draft_ref)
    cooldown(INTER_CONFIG_COOLDOWN_S, f"between methods")

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

print(f"\n\n{'='*70}")
print(f"  SUMMARY  ({WARMUPS} warmups · {RUNS_PER_PROMPT} runs · tree_budget={TREE_BUDGET})")
print(f"  Model: {TARGET_LABEL}  Draft: {loaded_draft_ref}")
print(f"{'='*70}")
baseline_decode = results_by_method["plain-mlx"][CODING_PROMPTS[0][0]]["decode_tps_median"]
for method in ("plain-mlx", "ddtree-mlx"):
    print(f"\n  {method}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}  {'vs plain':>10}")
    for prompt_name, _ in CODING_PROMPTS:
        m = results_by_method[method][prompt_name]
        ratio = m["decode_tps_median"] / baseline_decode if baseline_decode > 0 else 1.0
        acc = m.get("avg_acceptance")
        acc_str = f"  accept {acc:.2f} tok/cycle" if acc is not None else ""
        print(
            f"  {prompt_name:<12}  {m['ttft_ms_median']:>10.1f}"
            f"  {m['decode_tps_median']:>15.1f}  {ratio:>9.2f}×{acc_str}"
        )
print()
