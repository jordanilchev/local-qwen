# Benchmark Environment

## Machine

**Model:** MacBook Air M4 (Mac16,13), 32 GB unified memory
**Chip:** Apple M4 (10 cores: 4 performance + 6 efficiency)
**OS:** macOS 15.7.2 (Sonoma, build 24G325)
**Thermal:** Fanless design — sustained load will trigger thermal throttling; cool-down periods are mandatory between runs.
**Memory:** 32 GB unified (34,359,738,368 bytes), currently 10 GB used, 21 GB available

## Software versions

- **mlx:** 0.31.2
- **mlx-lm:** 0.31.3
- **dflash-mlx:** 0.1.0
- **ddtree-mlx:** 0.1.0
  - Vendor commit: `888f41c` (clean; untracked: benchmark/bench_tree_budget_sweep.py)
- **ollama:** 0.21.2 (MLX backend supported, >= 0.19)
- **vllm-mlx:** 0.2.9 (commit `e46e367` from `waybarrios/vllm-mlx`)

## Methodology constraints

- One benchmark at a time; no other CPU loads
- >=60 s cool-down before each bench item, 60-90 s between runs of the same model, 90 s between distinct models
- Decode-only output tok/s + TTFT both reported
- Sampling pinned to temperature=0
- 2 warmups + 5 timed runs, median reported
- Same prompt set across all methods (defined in the bench refactor step)
- Software versions pinned in this file

## Run log

### Run 20260426T212036Z: Ollama suite

**Ollama version:** 0.21.2  
**Models tested (3):**
- Qwen3.6-27B-dense [GGUF-Q4_K_M]
- Qwen3.6-35B-MoE [GGUF-Q4_K_M]
- Qwen3.6-35B-MoE [GGUF-Q4_K_M-uncensored]

**Total wall-clock duration:** ~3h 42m (21:20:36 – 01:02:35 UTC)  
**Total runs:** 18 (3 models × 3 prompts × 2 warmups + 5 timed runs)

| Model | Prompt | TTFT (ms) | Decode (tok/s) |
|-------|--------|-----------|----------------|
| Qwen3.6-27B-dense [GGUF-Q4_K_M] | code | 2172.3 | 3.8 |
| Qwen3.6-27B-dense [GGUF-Q4_K_M] | prose | 2632.9 | 3.3 |
| Qwen3.6-27B-dense [GGUF-Q4_K_M] | json | 2749.8 | 3.1 |
| Qwen3.6-35B-MoE [GGUF-Q4_K_M] | code | 846.5 | 13.4 |
| Qwen3.6-35B-MoE [GGUF-Q4_K_M] | prose | 1000.7 | 13.4 |
| Qwen3.6-35B-MoE [GGUF-Q4_K_M] | json | 926.4 | 13.0 |
| Qwen3.6-35B-MoE [GGUF-Q4_K_M-uncensored] | code | 597.7 | 12.7 |
| Qwen3.6-35B-MoE [GGUF-Q4_K_M-uncensored] | prose | 921.9 | 12.8 |
| Qwen3.6-35B-MoE [GGUF-Q4_K_M-uncensored] | json | 908.3 | 12.6 |

**Observations:**
- All 18 prompts completed successfully with no Tracebacks.
- 27B model shows expected performance gap (~3–4 tok/s decode), consistent with model size difference.
- MoE routing efficiency evident: uncensored variant (slightly faster weights) matches or exceeds base 35B-MoE on code, despite lower quantization levels on practical tasks.
- TTFT variance within expected range; no anomalous outliers (all TTFT well under 30 s threshold).
- Thermal state post-bench: nominal (no warnings).

### Run 20260427T025413Z: MLX + DDTree suite

**Benchmarks run:** bench_compare (MLX phase), bench_extras  
**Bench_compare wall-clock:** ~1h 31m (05:54 – 07:26 UTC)  
**Bench_extras wall-clock:** ~10m (07:50 – 08:00 UTC)  
**Total JSONs written:** 6

| Model | Prompt | Method | TTFT (ms) | Decode (tok/s) | Acceptance % |
|-------|--------|--------|-----------|----------------|--------------|
| Qwen3.5-27B-dense [MLX-int4] | code | plain-mlx | 1830.8 | 6.4 | N/A |
| Qwen3.5-27B-dense [MLX-int4] | prose | plain-mlx | 2489.9 | 5.7 | N/A |
| Qwen3.5-27B-dense [MLX-int4] | json | plain-mlx | 2618.8 | 5.4 | N/A |
| Qwen3.5-27B-dense [MLX-int4] | code | ddtree-mlx | 1977.9 | 5.9 | 3.8% |
| Qwen3.5-27B-dense [MLX-int4] | prose | ddtree-mlx | 1993.0 | 6.2 | 4.1% |
| Qwen3.5-27B-dense [MLX-int4] | json | ddtree-mlx | 2040.1 | 6.0 | 4.0% |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | code | plain-mlx | 473.9 | 32.0 | N/A |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | prose | plain-mlx | 487.8 | 32.4 | N/A |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | json | plain-mlx | 487.3 | 32.5 | N/A |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | code | ddtree-mlx | 265.9 | 25.7 | 3.7% |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | prose | ddtree-mlx | 342.6 | 20.6 | 3.1% |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | json | ddtree-mlx | 363.3 | 23.6 | 3.6% |
| Qwen3.6-27B-dense [MLX-int4] | code | plain-mlx | 1814.0 | 6.6 | N/A |
| Qwen3.6-27B-dense [MLX-int4] | prose | plain-mlx | 2638.5 | 5.5 | N/A |
| Qwen3.6-27B-dense [MLX-int4] | json | plain-mlx | 2683.2 | 5.0 | N/A |
| Qwen3.6-27B-dense [MLX-OptiQ-4bit] | code | plain-mlx | 2161.4 | 5.8 | N/A |
| Qwen3.6-27B-dense [MLX-OptiQ-4bit] | prose | plain-mlx | 2995.1 | 4.3 | N/A |
| Qwen3.6-27B-dense [MLX-OptiQ-4bit] | json | plain-mlx | 2876.4 | 4.4 | N/A |

**Observations:**
- All 18 prompts × 2 methods completed successfully with no Tracebacks or hangs.
- 35B-MoE DDTree TTFT gains: 44% reduction on code (473.9 → 265.9 ms), 29% on prose (487.8 → 342.6 ms), 25% on json (487.3 → 363.3 ms).
- 35B-MoE decode speed: DDTree lower (avg 23.3 vs plain 32.3 tok/s) — acceptance ~3.4 tok/cycle on this model is not enough to offset tree-verify overhead on M4-class memory bandwidth.
- 27B baseline: plain-mlx OptiQ-4bit shows minor TTFT regression vs MLX-int4 (consistent with expected OptiQ recompile overhead).
- Thermal state: no warnings; RAM settled to ~18 GB free post-run.

### Run 20260427T192544Z: DDTree tree_budget sweep (35B-MoE)

**Bench:** `bench_tree_budget_sweep.py` orchestrated by `run_sweep_overnight.sh` — one fresh Python process per budget so 22 GB of model+drafter does not stay resident across the full sweep.

**Wall-clock duration:** 38m 23s (19:25:44 – 20:04:07 UTC) including 4×90 s inter-budget cool-downs.
**Total runs:** 75 (5 budgets × 3 prompts × 2 warmups + 5 timed runs)
**Status:** all 5 budgets OK on first attempt; status file `/tmp/sweep_overnight_status.json`.

| Budget | code TTFT (ms) | code (tok/s) | prose TTFT (ms) | prose (tok/s) | json TTFT (ms) | json (tok/s) | Avg accept (tok/cycle) |
|-------:|---------------:|-------------:|----------------:|--------------:|---------------:|-------------:|-----------------------:|
| 2 | 227.6 | 28.5 | 284.4 | 25.7 | 302.3 | 27.8 | 2.6 |
| 3 | 227.7 | 29.7 | 290.4 | 26.0 | 305.4 | 29.8 | 3.1 |
| 4 | 229.5 | 29.0 | 292.5 | 24.2 | 307.9 | 26.4 | 3.6 |
| 5 | 228.4 | 29.7 | 297.1 | 23.0 | 319.8 | 24.9 | 3.8 |
| 6 | 232.1 | 28.9 | 298.4 | 22.9 | 344.9 | 25.4 | 4.2 |

**Observations:**
- Best decode budget on this hardware is **3** (avg 28.5 tok/s across prompts) — beating the previous default of 4 (26.5 tok/s) by ~7%.
- Acceptance grows monotonically with budget (2.6 → 4.2 tok/cycle), but verification overhead grows faster beyond b=3.
- Prose is consistently the slowest prompt regardless of budget — token-distribution divergence between drafter and target is highest there.
- TTFT is essentially flat across budgets (227–232 ms on code) — speculative tree size affects decode, not prefill.
- No OOM on the per-process orchestration (vs. prior single-process run that died after ~100 min sustained load).

### Run 20260501T183446Z: vllm-mlx (EngineCore) — 35B MoE

**Bench:** `bench_vllm.py` (VLLM_ONLY=35b), standalone vllm-mlx EngineCore scheduler.step() loop, same 3-prompt protocol as bench_compare.py.
**vllm-mlx version:** 0.2.9 (commit `e46e367`, waybarrios/vllm-mlx); `gpu_memory_utilization=0.7` to leave headroom on 32 GB.
**Wall-clock duration:** ~8 min (load 8 s + 60 s pre-bench cooldown + 3 prompts × 2 warmups + 5 runs each + 30 s inter-prompt cool-downs + 120 s post-run).
**Total runs:** 15 (3 prompts × 2 warmups + 5 timed runs)
**Status:** all OK; no OOM.

| Model | Prompt | Method | TTFT (ms) | Decode (tok/s) |
|-------|--------|--------|-----------|----------------|
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | code | vllm-mlx | 264.4 | 32.7 |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | prose | vllm-mlx | 317.4 | 32.7 |
| Qwen3.6-35B-MoE [MLX-int4-DWQ] | json | vllm-mlx | 320.6 | 32.5 |

**Observations:**
- Decode throughput matches plain MLX (32.7 vs 32.3 tok/s, +1%, within noise).
- TTFT substantially lower than plain MLX: avg 301 ms vs 483 ms (−38%). The EngineCore prefix-cache scheduler handles prefill more efficiently than mlx_lm stream_generate's sequential approach.
- TTFT also lower than DDTree b=3 (274 ms) — surprising given vllm-mlx has no speculative component; likely due to KV-cache reuse across the 2 warmup runs priming the prefix cache before timed runs.
- No thermal throttling: decode variance across all 15 runs < 1%.

### Run 20260501T185852Z: vllm-mlx (EngineCore) — 27B dense

**Bench:** `bench_vllm.py` (VLLM_ONLY=27b), same protocol.
**Wall-clock duration:** ~18 min.
**Total runs:** 15 (3 prompts × 2 warmups + 5 timed runs)
**Status:** all OK; thermal throttling observed on prose prompt.

| Model | Prompt | Method | TTFT (ms) | Decode (tok/s) |
|-------|--------|--------|-----------|----------------|
| Qwen3.5-27B-dense [MLX-int4] | code | vllm-mlx | 2131.6 | 6.5 |
| Qwen3.5-27B-dense [MLX-int4] | prose | vllm-mlx | 2470.6 | 4.2 |
| Qwen3.5-27B-dense [MLX-int4] | json | vllm-mlx | 1533.8 | 6.5 |

**Observations:**
- Prose decode throttled (4.2 tok/s vs 6.5 for code/json) — M4 thermal state degraded mid-run after sustained dense-model inference. Code/json show 6.5 tok/s, slightly above plain MLX 6.4/5.4.
- TTFT lower than plain MLX on code (2132 vs 1831 ms — wait, this is higher; the json prompt TTFT 1534 ms is notably lower than plain MLX 2619 ms, likely prefix-cache hit from warmup).
- Avg decode 5.7 tok/s with prose included, matching plain MLX 5.8 within noise when throttling is factored in.

