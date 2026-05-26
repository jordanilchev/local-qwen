#!/usr/bin/env python3
"""
Benchmark: vllm-mlx (EngineCore) on MLX targets.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_vllm
    VLLM_ONLY=35b HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_vllm
"""
import gc
import os
import time

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
    bench_vllm_engine,
    cooldown,
    get_timestamp_iso8601,
    make_results_payload,
    run_prompt_suite,
    write_results,
)

ALL_MLX_MODELS = [
    ("Qwen3.6-27B-dense  [MLX-int4]", "mlx-community/Qwen3.6-27B-4bit"),
    ("Qwen3.6-35B-MoE    [MLX-int4-DWQ]", "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ"),
]
_only = os.environ.get("VLLM_ONLY", "").lower()
if _only:
    MLX_MODELS = [(l, r) for l, r in ALL_MLX_MODELS if _only in l.lower() or _only in r.lower()]
    assert MLX_MODELS, f"VLLM_ONLY={_only!r} matched no models"
else:
    MLX_MODELS = ALL_MLX_MODELS

print("=" * 70, flush=True)
print("  PHASE: vllm-mlx (EngineCore)", flush=True)
print("=" * 70, flush=True)

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before vllm-mlx phase")

from vllm_mlx import EngineCore, EngineConfig
from vllm_mlx.utils.tokenizer import load_model_with_fallback

results_by_model = {}

for label, mlx_ref in MLX_MODELS:
    print(f"\nLoading {mlx_ref} via vllm_mlx...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load_model_with_fallback(mlx_ref)
    engine = EngineCore(model, tokenizer, EngineConfig(model_name=mlx_ref, gpu_memory_utilization=0.7))
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.", flush=True)

    results_this_model = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        suite = run_prompt_suite(
            f"vllm-mlx {label} [{prompt_name}]",
            bench_fn=lambda pt=prompt_text: bench_vllm_engine(engine, tokenizer, pt),
            warmup_fn=lambda wp: bench_vllm_engine(engine, tokenizer, wp),
        )
        results_this_model[prompt_name] = {"avg_acceptance": None, **suite}
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of vllm-mlx/{label}")

    results_by_model[(label, mlx_ref)] = results_this_model

    del engine, model, tokenizer
    gc.collect()
    cooldown(INTER_CONFIG_COOLDOWN_S, "between models: thermal reset + memory release")

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

ts = get_timestamp_iso8601()
for (label, model_ref), results_dict in results_by_model.items():
    payload = make_results_payload(
        ts=ts,
        method="vllm-mlx",
        model_label=label,
        model_ref=model_ref,
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
    print(f"Wrote vllm-mlx {label} results to {path}", flush=True)

print(f"\n\n{'='*70}")
print(f"  vllm-mlx SUMMARY")
print(f"  {WARMUPS} warmups · {RUNS_PER_PROMPT} runs per prompt · median")
print(f"{'='*70}")
for (label, _), results_dict in sorted(results_by_model.items()):
    print(f"\n  vllm-mlx {label}")
    print(f"  {'-'*60}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}")
    for prompt_name, _ in CODING_PROMPTS:
        m = results_dict[prompt_name]
        print(f"  {prompt_name:<12}  {m['ttft_ms_median']:>10.1f}  {m['decode_tps_median']:>15.1f}")
print()
