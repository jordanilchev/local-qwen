# Local LLM Inference on Apple Silicon

Benchmarks and scripts for running large language models locally on Apple Silicon using [MLX](https://github.com/ml-explore/mlx), [DFlash](https://github.com/bstnxbt/dflash-mlx), and [DDTree](https://github.com/humanrouter/ddtree-mlx) speculative decoding — compared against [Ollama](https://ollama.com).

## Hardware

| | |
|---|---|
| **Machine** | MacBook Air |
| **Chip** | Apple M4 |
| **Cores** | 10 (4 performance + 6 efficiency) |
| **Unified Memory** | 32 GB |
| **OS** | macOS 15.7.2 (Sequoia) |

## Benchmark Results

**Prompts:** 3 prompts (code · prose · json) at up to 200 generated tokens each.  
**Sampling:** greedy (temperature=0, seed=42).  
**Metric:** decode tok/s (generation only, prefill excluded) and TTFT (ms).  
**Method:** 2 warmups + 5 timed runs per (method, prompt), median reported. Mandatory 60–90 s cool-downs between runs to keep this fanless M4 out of thermal throttle.  
**Date:** 2026-04-27 · raw JSONs in [`benchmark/results/`](benchmark/results/) · environment + run log in [`benchmark/ENVIRONMENT.md`](benchmark/ENVIRONMENT.md).

Results are split by model family so each table is a direct apples-to-apples comparison of inference methods on the same underlying model. The **Avg tok/s** column is the median-across-prompts mean; per-prompt numbers are in `ENVIRONMENT.md`.

### Qwen 3.6 — 35B MoE (3B active params/token)

| Method | Quant | Avg tok/s | TTFT (ms) | Accept (tok/cycle) | vs Ollama | Source |
|--------|-------|----------:|----------:|-------------------:|----------:|--------|
| 🥇 vllm-mlx | MLX-int4-DWQ | **32.7** | 301 | — | 2.46× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| 🥈 Plain MLX | MLX-int4-DWQ | 32.3 | 483 | — | 2.43× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| DDTree (MLX, b=3) | MLX-int4-DWQ | 28.5 | 274 | 3.1 | 2.15× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| Ollama | GGUF-Q4_K_M | 13.3 | 925 | — | 1.00× | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Ollama (uncensored) | GGUF-Q4_K_M | 12.7 | 809 | — | 0.96× | [HauhauCS/Qwen3.6-35B-A3B-Uncensored](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive) |

Memory: ~21.6 GB (20.7 GB model + 0.9 GB DFlash drafter)

#### tree_budget sweep — DDTree on Qwen3.6-35B-MoE [MLX-int4-DWQ]

Sweep against the same DFlash drafter, fresh Python process per budget, same 3-prompt protocol.

| Budget | code (t/s) | prose (t/s) | json (t/s) | Avg t/s | Avg accept (tok/cycle) |
|-------:|-----------:|------------:|-----------:|--------:|-----------------------:|
| 2 | 28.5 | 25.7 | 27.8 | 27.3 | 2.6 |
| **3** | **29.7** | **26.0** | **29.8** | **28.5** | 3.1 |
| 4 | 29.0 | 24.2 | 26.4 | 26.5 | 3.6 |
| 5 | 29.7 | 23.0 | 24.9 | 25.9 | 3.8 |
| 6 | 28.9 | 22.9 | 25.4 | 25.7 | 4.2 |

**Best budget on this hardware: 3.** Acceptance rises monotonically with the tree size (more candidates verified per cycle), but on M4's ~120 GB/s memory bus the verification cost grows faster than the kernel-launch savings beyond b=3. Even the best DDTree configuration is ~12% slower than plain MLX on this model — see *Key observations* below.

### 27B dense

| Method | Model | Quant | Avg tok/s | TTFT (ms) | Accept (tok/cycle) | Source |
|--------|-------|-------|----------:|----------:|-------------------:|--------|
| 🥇 vllm-mlx | ✨ Qwen3.6-27B | MLX-int4 | **6.5** | 1949 | — | [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) |
| DDTree (MLX, b=4)‡ | Qwen3.5-27B | MLX-int4 | 6.0 | 2003 | 4.0 | [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) |
| Plain MLX | Qwen3.5-27B | MLX-int4 | 5.8 | 2490 | — | [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) |
| vllm-mlx† | Qwen3.5-27B | MLX-int4 | 5.7 | 2045 | — | [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) |
| Plain MLX | ✨ Qwen3.6-27B | MLX-int4 | 5.7 | 2638 | — | [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) |
| Plain MLX | ✨ Qwen3.6-27B | MLX-OptiQ-4bit | 4.8 | 2876 | — | [mlx-community/Qwen3.6-27B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-OptiQ-4bit) |
| Ollama | ✨ Qwen3.6-27B | GGUF-Q4_K_M | 3.4 | 2633 | — | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |

✨ = Qwen3.6 generation &nbsp;|&nbsp; † vllm-mlx 3.5 prose throttled on the fanless M4 (4.2 tok/s vs ~6.5 on code/json), dragging the average down.
‡ DDTree only runnable on Qwen3.5-27B — no DFlash drafter for 3.6 yet. Memory: ~18.2 GB (15 GB model + 3.2 GB drafter).
No Ollama Q4_K_M GGUF run for the 3.5 generation — the 3.6 Ollama row is the same-architecture reference.

### Key observations

- **MLX + Metal beats Ollama (llama.cpp) on the same model:** 32.7 vs 13.3 tok/s on the 35B MoE (+146%) and 5.7 vs 3.4 tok/s on the 27B dense (+66%). The MLX backend is dramatically more efficient on Apple Silicon.
- **vllm-mlx leads on both model sizes:** 32.7 tok/s on 35B-MoE (301 ms TTFT, −38% vs plain MLX's 483 ms) and 6.5 tok/s on Qwen3.6-27B-dense (1949 ms TTFT, −26% vs plain MLX's 2638 ms). The 35B decode gain is within noise (+1%); the 27B gain is +14%, suggesting the EngineCore scheduler extracts more decode efficiency from the memory-bandwidth-bound dense workload.
- **DDTree no longer beats plain MLX on the 35B MoE under fair methodology.** With multi-prompt + greedy + cool-downs, plain MLX runs at 32.3 tok/s and the best DDTree budget (b=3) reaches only 28.5 tok/s. DDTree still wins on TTFT (274 ms vs 483 ms, though vllm-mlx at 301 ms is close), but the original "DDTree fastest" headline came from a single code prompt where draft acceptance ran higher — averaging across code/prose/json reverses the ordering.
- **DDTree on 27B dense is roughly a wash:** +3% on Qwen3.5-27B (6.0 vs 5.8 tok/s) — within the noise floor of cool-down variance. The model is so memory-bound that speculative decoding has little spare bandwidth to exploit.
- **MoE vs dense (cross-family):** the 35B MoE runs at 32.3 tok/s vs 5.7 tok/s for the 27B dense under plain MLX — a 5.7× gap that is entirely architectural. MoE sparsity is a free lunch on Apple Silicon.
- **OptiQ mixed precision is slower than uniform int4:** Qwen3.6-27B-OptiQ (4.5 BPW avg, 247 layers at 8-bit) runs at 4.8 tok/s vs 5.7 tok/s for uniform 4-bit — the heavier 8-bit layers cost more bandwidth than they buy in quality on this memory-bandwidth-bound chip.

### Methodology / caveats

- All runs on a fanless M4 MacBook Air (32 GB UMA, ~120 GB/s memory bandwidth). Sustained load throttles; results are sensitive to thermal state, hence the cool-down discipline.
- Software pinned: `mlx-lm 0.31.3`, `mlx 0.31.2`, `dflash-mlx 0.1.0`, `ddtree-mlx 0.1.0` (vendor commit `888f41c`), `ollama 0.21.2` (with MLX backend), `vllm-mlx 0.2.9` (commit `e46e367`, from `waybarrios/vllm-mlx`).
- Ollama TTFT is captured from the streaming endpoint (first chunk wall-clock), not from total elapsed time.
- DDTree decode is greedy (no temperature/sampler param in `generate_ddtree_once`); plain MLX uses `make_sampler(temp=0.0)`. Same seed (42) for both.
- "Accept (tok/cycle)" = average tokens accepted per draft–verify cycle (typical 2.6–4.2 for tree budgets 2–6). It is **not** a probability; budget>=2 means more than one token can be accepted per cycle.
- Per-process model load on each sweep budget so that 22 GB of model+drafter does not stay resident through the full sweep — earlier single-process runs OOM'd after ~100 minutes of sustained load.

---

## Reproduce From Scratch

### Prerequisites

- macOS Sequoia or Sonoma
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- Disk space: ~20 GB (best setup — 35B MoE MLX + drafter only) · ~40 GB (add Ollama comparison models) · ~80 GB (full suite — all MLX + all Ollama models)
- Ollama installed (for Ollama comparisons only): https://ollama.com

### 1. Clone and set up the environment

```bash
git clone <this-repo>
cd local-qwen

# Create Python 3.12 venv and install base deps
uv venv --python 3.12
uv pip install mlx-lm dflash-mlx
uv pip install -e vendor/ddtree-mlx

# vllm-mlx (used by benchmark/bench_vllm.py)
# Recommended: install from GitHub for latest fixes
uv pip install git+https://github.com/waybarrios/vllm-mlx.git
# Or stable PyPI release
# uv pip install vllm-mlx
```

> **Note:** `vendor/ddtree-mlx` is the vendored source of the DDTree runtime. It is installed in editable mode so local patches are picked up immediately.

### 2. Download the models

Models are stored in `~/Models/HuggingFace`. Set `HF_HOME` once:

```bash
export HF_HOME=~/Models/HuggingFace
```

**Best setup — Qwen3.6-35B-MoE [MLX-int4-DWQ] + DFlash drafter (~21.6 GB total):**

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.6-35B-A3B-4bit-DWQ')  # 20.7 GB
snapshot_download('z-lab/Qwen3.6-35B-A3B-DFlash')             #  0.9 GB
"
```

**Alternative — Qwen3.5-27B-dense [MLX-int4] + DFlash drafter (~18.2 GB total):**

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-27B-4bit')     # 15 GB
snapshot_download('z-lab/Qwen3.5-27B-DFlash')           #  3.2 GB
"
```

### 3. Verify installation

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python -m pytest tests/ -v
```

`test_imports` passes immediately. `test_short_inference` and `test_result_keys` require the 27B models to be downloaded.

### 4. Run inference

```bash
# Single prompt (reads from args or stdin)
HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/infer_ddtree.py "Explain transformers in one paragraph."

# Override defaults
MAX_TOKENS=512 TREE_BUDGET=4 HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/infer_ddtree.py "Your prompt"
```

### 5. Start the OpenAI-compatible server

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/ddtree_server.py --port 8006
```

Point any OpenAI-compatible client (Continue.dev, Cursor, LM Studio, etc.) at `http://localhost:8006/v1`.

```bash
# Quick smoke test
curl http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":32}' \
  | python3 -m json.tool
```

### 6. Run the benchmark

**Full 6-method comparison (Ollama + MLX + DDTree):**

```bash
# Requires Ollama running with: ollama pull qwen3.6:27b && ollama pull qwen3.6-uncensored:35b-q4
HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_compare.py
```

**MLX-only comparison (no Ollama needed):**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_ddtree.py
```

> **Memory warning:** The benchmark loads MLX models only after Ollama models are explicitly unloaded. Do not add new Ollama models to `OLLAMA_MODELS` if their GGUF size + 21.6 GB exceeds 32 GB.

---

## Project Structure

```
local-qwen/
├── benchmark/
│   ├── _lib.py                       # Shared prompts, cool-down constants, JSON writer
│   ├── bench_compare.py              # 6-way: Ollama vs plain MLX vs DDTree
│   ├── bench_ddtree.py               # 2-way: plain MLX vs DDTree (no Ollama)
│   ├── bench_ollama.py               # Ollama-only baseline
│   ├── bench_extras.py               # Plain MLX for Qwen3.6-27B variants (no drafter)
│   ├── bench_tree_budget_sweep.py    # DDTree tree_budget sweep (one budget per process)
│   ├── bench_vllm.py                 # vllm-mlx EngineCore standalone bench
│   ├── run_sweep_overnight.sh        # Orchestrator: budget per fresh process + cool-downs
│   ├── ENVIRONMENT.md                # Hardware/software pin + run log
│   └── results/                      # JSON output, one file per (method, model[, budget])
├── scripts/
│   ├── infer_ddtree.py      # Interactive single-turn inference
│   └── ddtree_server.py     # OpenAI-compatible server (port 8006)
├── tests/
│   └── test_ddtree_smoke.py # Import + short inference smoke tests
├── vendor/
│   └── ddtree-mlx/          # Vendored DDTree runtime (editable install)
├── .env                     # HF_HOME=~/Models/HuggingFace (gitignored)
└── requirements.txt         # dflash-mlx + -e vendor/ddtree-mlx
```

## Models Reference

| Model | HuggingFace ID | Size | Notes |
|-------|---------------|------|-------|
| Target (35B MoE) | `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` | 20.7 GB | Best quality/speed |
| Drafter (35B MoE) | `z-lab/Qwen3.6-35B-A3B-DFlash` | 0.9 GB | DDTree drafter |
| Target (27B dense, 3.5) | `mlx-community/Qwen3.5-27B-4bit` | 15 GB | Baseline dense, has drafter |
| Drafter (27B dense, 3.5) | `z-lab/Qwen3.5-27B-DFlash` | 3.2 GB | DDTree drafter |
| Target (27B dense, 3.6) | `mlx-community/Qwen3.6-27B-4bit` | ~15 GB | Newer dense, no drafter yet |
| Target (27B dense, 3.6 OptiQ) | `mlx-community/Qwen3.6-27B-OptiQ-4bit` | ~16 GB | Mixed-precision variant |
