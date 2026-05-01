#!/usr/bin/env python3
"""
Benchmark: vllm-mlx (EngineCore) vs the same MLX targets used in bench_compare.py.

Multi-prompt run (code, prose, JSON). TTFT + decode tok/s, cool-down discipline,
JSON output written under benchmark/results/. Format matches bench_compare.py
so the result files plug straight into the same summary tables.

Timing convention (matches plain-MLX in bench_compare.py):
  - TTFT = wall time from the first scheduler.step to the first emitted token
  - Decode = (tokens - 1) / (elapsed time after the first token)

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_vllm
    VLLM_ONLY=35b HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_vllm
"""
import os, time, gc, statistics, uuid
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    PROMPTS, HOST, cooldown,
    PRE_BENCH_COOLDOWN_S, INTER_PROMPT_COOLDOWN_S, INTER_CONFIG_COOLDOWN_S, POST_BENCH_COOLDOWN_S,
    write_results, get_timestamp_iso8601,
)

WARMUPS = 2
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200

# Same model list as bench_compare.py MLX_MODELS.
# Filter via VLLM_ONLY="35b" or "27b" to bench a single model (cuts wall time / peak RAM).
ALL_MLX_MODELS = [
    ("Qwen3.5-27B-dense  [MLX-int4]",     "mlx-community/Qwen3.5-27B-4bit"),
    ("Qwen3.6-27B-dense  [MLX-int4]",     "mlx-community/Qwen3.6-27B-4bit"),
    ("Qwen3.6-35B-MoE    [MLX-int4-DWQ]", "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ"),
]
_only = os.environ.get("VLLM_ONLY", "").lower()
if _only:
    MLX_MODELS = [(l, r) for l, r in ALL_MLX_MODELS if _only in l.lower() or _only in r.lower()]
    assert MLX_MODELS, f"VLLM_ONLY={_only!r} matched no models in {[l for l,_ in ALL_MLX_MODELS]}"
else:
    MLX_MODELS = ALL_MLX_MODELS

# ── vllm-mlx single-prompt bench ──────────────────────────────────────────────

def bench_vllm(engine, tokenizer, prompt: str) -> tuple[float, float]:
    """Run vllm-mlx EngineCore on a single prompt, return (ttft_ms, decode_tps).

    Drives engine.scheduler directly (sync) so we can take a precise wall-clock
    reading at the moment the first token is emitted. SamplingParams forces
    greedy (temperature=0) to match plain-MLX/DDTree fairness.
    """
    from vllm_mlx import Request, SamplingParams
    from vllm_mlx.mlx_streams import bind_generation_streams

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, top_p=1.0, top_k=0)
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

    rid = str(uuid.uuid4())
    request = Request(request_id=rid, prompt=prompt_str, sampling_params=sp)

    bind_generation_streams()
    engine.scheduler.add_request(request)

    t0_start = time.perf_counter()
    t0_first = None
    count = 0
    ttft_ms = 0.0
    try:
        while engine.scheduler.has_requests():
            out = engine.scheduler.step()
            for ro in out.outputs:
                if ro.request_id != rid:
                    continue
                n = len(ro.new_token_ids)
                if n > 0 and t0_first is None:
                    t0_first = time.perf_counter()
                    ttft_ms = (t0_first - t0_start) * 1000.0
                count += n
    finally:
        engine.scheduler.remove_finished_request(rid)

    if t0_first is None or count <= 1:
        return ttft_ms, 0.0

    elapsed_after_first = time.perf_counter() - t0_first
    decode_tps = (count - 1) / elapsed_after_first if elapsed_after_first > 0 else 0.0
    return ttft_ms, decode_tps

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_suite(label: str, bench_fn) -> dict:
    print(f"\n{'='*70}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*70}", flush=True)
    for w in range(1, WARMUPS + 1):
        print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
        bench_fn()
        print("done", flush=True)

    ttft_runs, decode_runs = [], []
    for i in range(1, RUNS_PER_PROMPT + 1):
        print(f"  [run {i}/{RUNS_PER_PROMPT}] ", end="", flush=True)
        ttft_ms, decode_tps = bench_fn()
        ttft_runs.append(ttft_ms)
        decode_runs.append(decode_tps)
        print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s", flush=True)

    return {"ttft_ms_runs": ttft_runs, "decode_tps_runs": decode_runs}

# ── Main ──────────────────────────────────────────────────────────────────────

results_by_model = {}

print("=" * 70, flush=True)
print("  PHASE: vllm-mlx (EngineCore)", flush=True)
print("=" * 70, flush=True)

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before vllm-mlx phase")

from vllm_mlx import EngineCore, EngineConfig
from vllm_mlx.utils.tokenizer import load_model_with_fallback

for label, mlx_ref in MLX_MODELS:
    print(f"\nLoading {mlx_ref} via vllm_mlx...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load_model_with_fallback(mlx_ref)
    # gpu_memory_utilization=0.7 leaves headroom on a 32 GB M4 — vllm-mlx
    # default 0.9 plus a 21 GB MoE plus baseline pressure has OOM'd before.
    engine = EngineCore(model, tokenizer, EngineConfig(model_name=mlx_ref, gpu_memory_utilization=0.7))
    print(f"Loaded in {time.perf_counter()-t0:.1f}s.", flush=True)

    results_this_model = {}
    for prompt_name, prompt_text in PROMPTS:
        suite_result = run_suite(
            f"vllm-mlx {label} [{prompt_name}]",
            lambda e=engine, t=tokenizer, pr=prompt_text: bench_vllm(e, t, pr),
        )
        ttft_runs = suite_result["ttft_ms_runs"]
        decode_runs = suite_result["decode_tps_runs"]
        results_this_model[prompt_name] = {
            "ttft_ms_runs": ttft_runs,
            "decode_tps_runs": decode_runs,
            "avg_acceptance": None,
            "ttft_ms_median": statistics.median(ttft_runs),
            "decode_tps_median": statistics.median(decode_runs),
        }
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of vllm-mlx/{label}")

    results_by_model[(label, mlx_ref)] = results_this_model

    # Free before next model
    del engine, model, tokenizer
    gc.collect()
    cooldown(INTER_CONFIG_COOLDOWN_S, "between models: thermal reset + memory release")

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

# ── Write JSON outputs ─────────────────────────────────────────────────────────

ts = get_timestamp_iso8601()

for (label, model_ref), results_dict in results_by_model.items():
    payload = {
        "ts": ts,
        "host": HOST,
        "method": "vllm-mlx",
        "model_label": label,
        "model_ref": model_ref,
        "drafter_ref": None,
        "tree_budget": None,
        "sampling": {"temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS},
        "warmups": WARMUPS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "results": [
            {"prompt": prompt_name, **results_dict[prompt_name]}
            for prompt_name, _ in PROMPTS
        ],
    }
    path = write_results(payload)
    print(f"Wrote vllm-mlx {label} results to {path}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print(f"  vllm-mlx SUMMARY")
print(f"  {WARMUPS} warmups · {RUNS_PER_PROMPT} runs per prompt · median · {MAX_TOKENS} max tok")
print(f"{'='*70}")

for (label, _), results_dict in sorted(results_by_model.items()):
    print(f"\n  vllm-mlx {label}")
    print(f"  {'-'*60}")
    print(f"  {'Prompt':<12}  {'TTFT (ms)':>10}  {'Decode (tok/s)':>15}")
    for prompt_name, _ in PROMPTS:
        ttft_med = results_dict[prompt_name]["ttft_ms_median"]
        decode_med = results_dict[prompt_name]["decode_tps_median"]
        print(f"  {prompt_name:<12}  {ttft_med:>10.1f}  {decode_med:>15.1f}")

print()
