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

**Prompts:** 3 coding prompts (`code-algo` · `code-async` · `code-cache`) at up to 200 generated tokens each.  
**Sampling:** greedy (temperature=0, seed=42 where supported), `enable_thinking=False`.  
**Metric:** decode tok/s (wall clock after first token) and TTFT (ms); per-run token counts in JSON.  
**Method:** 2 warmups on distinct prompts + 5 timed runs per (method, prompt), **60 s between timed runs**, median reported. Cool-downs per `benchmark/_lib.py`.  
**Date:** 2026-05-25 · raw JSONs in [`benchmark/results/`](benchmark/results/) · environment + run log in [`benchmark/ENVIRONMENT.md`](benchmark/ENVIRONMENT.md).

Results are split by model family so each table is a direct apples-to-apples comparison of inference methods on the same underlying model. The **Avg tok/s** column is the mean of per-prompt medians.

### Qwen 3.6 — 35B MoE (3B active params/token)

| Method | Quant | Avg tok/s | TTFT (ms) | Accept | vs Ollama | Source |
|--------|-------|----------:|----------:|-------:|----------:|--------|
| 🥇 Plain MLX | MLX-int4-DWQ | **32.9** | 723 | — | 1.98× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| 🥈 vllm-mlx | MLX-int4-DWQ | 32.8 | 599 | — | 1.97× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| DDTree (MLX, b=3) | MLX-int4-DWQ | 32.1 | 362 | 3.3 tok/cycle | 1.93× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| Ollama | GGUF-Q4_K_M | 16.6 | 686 | — | 1.00× | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |

Memory: ~21.6 GB (20.7 GB model + 0.9 GB DFlash drafter)

#### tree_budget sweep — DDTree on Qwen3.6-35B-MoE [MLX-int4-DWQ]

Prior sweep (2026-04-27) used mixed `code/prose/json` prompts; not re-run under the current `coding` protocol.

| Budget | code (t/s) | prose (t/s) | json (t/s) | Avg t/s | Avg accept (tok/cycle) |
|-------:|-----------:|------------:|-----------:|--------:|-----------------------:|
| 2 | 28.5 | 25.7 | 27.8 | 27.3 | 2.6 |
| **3** | **29.7** | **26.0** | **29.8** | **28.5** | 3.1 |
| 4 | 29.0 | 24.2 | 26.4 | 26.5 | 3.6 |
| 5 | 29.7 | 23.0 | 24.9 | 25.9 | 3.8 |
| 6 | 28.9 | 22.9 | 25.4 | 25.7 | 4.2 |

**Default budget: 3.** Under the current coding-prompt run, DDTree at b=3 reaches 32.1 tok/s — within ~2% of plain MLX (32.9).

### Qwen 3.6 — 27B dense

| Method | Quant | Avg tok/s | TTFT (ms) | Accept | vs Ollama | Source |
|--------|-------|----------:|----------:|-------:|----------:|--------|
| 🥇 llama.cpp MTP (n=2) | GGUF-Q4_K_XL | **6.0** | 1897 | 90% draft | 1.36× | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) |
| Plain MLX | MLX-int4 | 5.5 | 2131 | — | 1.25× | [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) |
| vllm-mlx | MLX-int4 | 5.4 | 1929 | — | 1.23× | [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) |
| llama.cpp baseline | GGUF-Q4_K_XL | 5.1 | 1811 | — | 1.16× | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) |
| Ollama | GGUF-Q4_K_M | 4.4 | 1503 | — | 1.00× | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Plain MLX (OptiQ) | MLX-OptiQ-4bit | 2.6 | 3227 | — | 0.59× | [mlx-community/Qwen3.6-27B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-OptiQ-4bit) |

† Plain MLX and vllm-mlx show thermal throttling on `code-cache` (3.2–3.3 tok/s vs ~6.6 on other prompts) on this fanless M4.

### Key observations

- **MLX + Metal beats Ollama on the same model:** 32.9 vs 16.6 tok/s on 35B MoE (+98%) and 5.5 vs 4.4 tok/s on 27B dense (+25%).
- **DDTree on 35B MoE is now competitive with plain MLX** under the unified methodology (32.1 vs 32.9 tok/s at b=3), while cutting TTFT roughly in half (362 ms vs 723 ms).
- **vllm-mlx matches plain MLX decode on 35B** (32.8 vs 32.9 tok/s) with lower TTFT (599 ms). On 27B, decode is in the same band but thermally noisy.
- **llama.cpp MTP gives +18% over its own baseline** (6.0 vs 5.1 tok/s) at `--spec-draft-n-max 2` with ~90% draft acceptance — modest but real on this memory-bound chip.
- **OptiQ mixed precision is slower than uniform int4:** 2.6 vs 5.5 tok/s on 27B — bandwidth cost of 8-bit layers dominates on M4.
- **MoE vs dense:** 35B MoE plain MLX at 32.9 tok/s vs 27B at 5.5 tok/s — a 6× architectural gap on identical silicon.

### Methodology / caveats

- All runs on a fanless M4 MacBook Air (32 GB UMA, ~120 GB/s memory bandwidth). Sustained load throttles; results are sensitive to thermal state, hence the cool-down discipline.
- Software pinned: `mlx-lm 0.31.3`, `mlx 0.31.2`, `dflash-mlx 0.1.0`, `ddtree-mlx 0.1.0` (vendor commit `888f41c`), `vllm-mlx 0.2.9` (commit `e46e367`), Ollama 0.24.0, llama.cpp ≥ b9180 for MTP.
- Ollama uses `/api/chat` with `think: false`; same user messages as MLX (`enable_thinking=False`).
- DDTree default `tree_budget=3`; TTFT from `prefill_us`, decode from wall clock after TTFT.
- DDTree decode is greedy (no temperature/sampler param in `generate_ddtree_once`); plain MLX uses `make_sampler(temp=0.0)`.
- Published tables before 2026-05-24 used mixed `code/prose/json` prompts — not comparable to current `coding` prompt set.
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
MAX_TOKENS=512 TREE_BUDGET=3 HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/infer_ddtree.py "Your prompt"
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

**Full suite (recommended):**

```bash
./benchmark/run_all_benches.sh
```

Order: llama.cpp MTP → plain MLX extras → DDTree → vllm-mlx → Ollama + MLX compare.

**Ollama-only compare** (requires `ollama pull qwen3.6:27b && ollama pull qwen3.6:35b`):

```bash
BENCH_PHASE=ollama HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_compare
```

**MLX-only comparison (no Ollama needed):**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_ddtree
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
│   ├── bench_llamacpp_mtp.py         # llama.cpp baseline vs MTP speculative decoding (coding prompts)
│   ├── run_all_benches.sh            # Full suite orchestrator
│   ├── rerun_compare_ollama.sh       # Ollama phase only (27b + 35b)
│   ├── rerun_ddtree_after_main.sh    # DDTree re-run helper
│   ├── rerun_llamacpp_mtp_after_ollama.sh
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
| GGUF baseline (27B, 3.6) | `unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL` | ~15 GB | llama.cpp 4-bit baseline |
| GGUF + MTP heads (27B, 3.6) | `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` | ~15 GB | Bundled MTP prediction heads for speculative decoding |
