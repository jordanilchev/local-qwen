#!/usr/bin/env python3
"""
Plain MLX benchmark for targets without a DFlash drafter (no DDTree path).

Currently: Qwen3.6-27B-dense (uniform int4) and Qwen3.6-27B-dense (OptiQ 4-bit).

Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_extras
"""
import gc
import os

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    CODING_PROMPTS,
    DEFAULT_SAMPLING,
    INTER_CONFIG_COOLDOWN_S,
    INTER_PROMPT_COOLDOWN_S,
    POST_BENCH_COOLDOWN_S,
    PRE_BENCH_COOLDOWN_S,
    PROMPT_SET,
    RUNS_PER_PROMPT,
    WARMUPS,
    bench_mlx_plain,
    cooldown,
    get_timestamp_iso8601,
    make_results_payload,
    run_prompt_suite,
    write_results,
)

TARGETS = [
    ("Qwen3.6-27B-dense  [MLX-int4]", "mlx-community/Qwen3.6-27B-4bit"),
    ("Qwen3.6-27B-dense  [MLX-OptiQ-4bit]", "mlx-community/Qwen3.6-27B-OptiQ-4bit"),
]

from mlx_lm import load

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before extras")

for target_idx, (label, model_ref) in enumerate(TARGETS):
    if target_idx > 0:
        cooldown(INTER_CONFIG_COOLDOWN_S, "between extras targets")

    print(f"\nLoading {model_ref}...", flush=True)
    model, tokenizer = load(model_ref)
    print("Loaded.", flush=True)

    results = []
    for p_idx, (prompt_name, prompt_text) in enumerate(CODING_PROMPTS):
        if p_idx > 0:
            cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {label}")

        suite = run_prompt_suite(
            f"Plain mlx_lm {label} [{prompt_name}]",
            bench_fn=lambda pt=prompt_text: bench_mlx_plain(model, tokenizer, pt),
            warmup_fn=lambda wp: bench_mlx_plain(model, tokenizer, wp),
        )
        results.append({"prompt": prompt_name, "avg_acceptance": None, **suite})

    payload = make_results_payload(
        ts=get_timestamp_iso8601(),
        method="plain-mlx",
        model_label=label,
        model_ref=model_ref,
        results=results,
        sampling=dict(DEFAULT_SAMPLING),
        warmups=WARMUPS,
        runs_per_prompt=RUNS_PER_PROMPT,
        prompt_set=PROMPT_SET,
    )
    path = write_results(payload)
    print(f"\nWrote {label} results to {path}", flush=True)

    del model, tokenizer
    gc.collect()

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")
