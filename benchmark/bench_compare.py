#!/usr/bin/env python3
"""
Benchmark: Ollama models vs plain mlx_lm vs DFlash+DDTree.

Prefer `benchmark.bench_session` for fair single-session comparisons.
This script remains for partial re-runs (BENCH_PHASE=ollama|mlx|all).

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_compare
    BENCH_PHASE=ollama|mlx|all
"""
import gc
import os
import subprocess

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
    bench_ollama_chat,
    cooldown,
    ensure_ollama_model,
    get_timestamp_iso8601,
    load_ddtree_runtime,
    load_mlx_target,
    make_results_payload,
    ollama_unload,
    resolve_draft_ref,
    run_prompt_suite,
    wait_for_ollama_server,
    write_results,
)
from benchmark.models import MODEL_FAMILIES, family_label

BENCH_PHASE = os.environ.get("BENCH_PHASE", "all").lower()
assert BENCH_PHASE in ("all", "ollama", "mlx"), f"BENCH_PHASE must be all|ollama|mlx (got {BENCH_PHASE!r})"
RUN_OLLAMA = BENCH_PHASE in ("all", "ollama")
RUN_MLX = BENCH_PHASE in ("all", "mlx")
SESSION_ID = os.environ.get("BENCH_SESSION_ID")

# MoE family only (legacy compare scope).
_COMPARE_FAMILY = next(f for f in MODEL_FAMILIES if f.id == "3.6-35b-moe")
TREE_BUDGET = DEFAULT_TREE_BUDGET


def ensure_ollama_server() -> None:
    try:
        wait_for_ollama_server(timeout_s=5)
        return
    except RuntimeError:
        print("Starting ollama serve...", flush=True)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_for_ollama_server(timeout_s=30)


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


def write_method_results(
    ts: str,
    method: str,
    results_dict: dict,
    *,
    drafter_ref: str | None = None,
    tree_budget: int | None = None,
):
    family = _COMPARE_FAMILY
    payload = make_results_payload(
        ts=ts,
        session_id=SESSION_ID,
        family_id=family.id,
        method=method,
        model_label=family_label(family),
        model_ref=family.mlx_ref if method != "ollama" else family.ollama_ref,
        drafter_ref=drafter_ref,
        tree_budget=tree_budget,
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


results_by_method_model: dict = {}
ts = get_timestamp_iso8601()
family = _COMPARE_FAMILY

if RUN_OLLAMA and family.ollama_ref:
    print("=" * 70, flush=True)
    print("  PHASE 1: Ollama models (/api/chat, think=false)", flush=True)
    print("=" * 70, flush=True)

    ensure_ollama_server()
    cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before Ollama phase")

    ollama_model = family.ollama_ref
    label = family_label(family)
    print(f"\nBenchmarking Ollama {label}", flush=True)
    ensure_ollama_model(ollama_model)
    cooldown(INTER_CONFIG_COOLDOWN_S, "between configs: thermal reset before new model")

    results_this_model = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        suite = run_prompt_suite(
            f"Ollama {label} [{prompt_name}]",
            bench_fn=lambda pt=prompt_text: bench_ollama_chat(ollama_model, pt),
            warmup_fn=lambda wp, m=ollama_model: bench_ollama_chat(m, wp),
        )
        results_this_model[prompt_name] = {"avg_acceptance": None, **suite}
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of ollama/{label}")

    results_by_method_model[("ollama", label, ollama_model)] = results_this_model
    write_method_results(ts, "ollama", results_this_model)

    print(f"\nUnloading {ollama_model}...", end=" ", flush=True)
    ollama_unload(ollama_model)
    print("done", flush=True)

if RUN_MLX and family.mlx_ref:
    print(f"\n{'='*70}", flush=True)
    print("  PHASE 2: MLX models (plain target only, then target+drafter)", flush=True)
    print(f"{'='*70}", flush=True)

    cooldown(INTER_CONFIG_COOLDOWN_S, "before MLX phase: reset after Ollama")

    mlx_ref = family.mlx_ref
    label = family_label(family)
    draft_ref = family.draft_ref or resolve_draft_ref(mlx_ref)

    print(f"\nLoading plain MLX target {mlx_ref}...", flush=True)
    target, tok = load_mlx_target(mlx_ref)

    results_plain = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        suite = run_prompt_suite(
            f"Plain mlx_lm {label} [{prompt_name}]",
            bench_fn=lambda pt=prompt_text: bench_mlx_plain(target, tok, pt),
            warmup_fn=lambda wp: bench_mlx_plain(target, tok, wp),
        )
        results_plain[prompt_name] = {"avg_acceptance": None, **suite}
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of plain-mlx/{label}")

    results_by_method_model[("plain-mlx", label, mlx_ref)] = results_plain
    write_method_results(ts, "plain-mlx", results_plain)

    del target, tok
    gc.collect()
    cooldown(INTER_CONFIG_COOLDOWN_S, "unload plain target before DDTree load")

    if draft_ref:
        print(f"\nLoading {mlx_ref} + drafter {draft_ref}...", flush=True)
        target, tok, draft, loaded_draft_ref, stop = load_ddtree_runtime(mlx_ref, draft_ref)
        print(f"Loaded drafter: {loaded_draft_ref}", flush=True)

        results_ddtree = {}
        for prompt_name, prompt_text in CODING_PROMPTS:
            suite = run_ddtree_suite(
                f"DFlash+DDTree {label} [{prompt_name}]",
                target, tok, draft, stop, prompt_text,
            )
            results_ddtree[prompt_name] = suite
            cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of ddtree-mlx/{label}")

        results_by_method_model[("ddtree-mlx", label, mlx_ref)] = results_ddtree
        write_method_results(
            ts,
            "ddtree-mlx",
            results_ddtree,
            drafter_ref=loaded_draft_ref,
            tree_budget=TREE_BUDGET,
        )

        del target, tok, draft, stop
        gc.collect()

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

print(f"\n\n{'='*70}")
print(f"  COMPARISON SUMMARY")
print(f"  {WARMUPS} warmups · {RUNS_PER_PROMPT} runs/prompt")
print(f"{'='*70}")

for (method, label, _), results_dict in sorted(results_by_method_model.items()):
    print(f"\n  {method} {label}")
    print(f"  {'-'*60}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}")
    for prompt_name, _ in CODING_PROMPTS:
        m = results_dict[prompt_name]
        print(f"  {prompt_name:<12}  {m['ttft_ms_median']:>10.1f}  {m['decode_tps_median']:>15.1f}")
print()
