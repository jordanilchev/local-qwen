# Local LLM Inference on Apple Silicon

Benchmarks and scripts for running large language models locally on Apple Silicon using [MLX](https://github.com/ml-explore/mlx), [DFlash](https://github.com/humanrouter/dflash-mlx), and [DDTree](https://github.com/humanrouter/ddtree-mlx) speculative decoding — compared against [Ollama](https://ollama.com).

## Hardware

| | |
|---|---|
| **Machine** | MacBook Air |
| **Chip** | Apple M4 |
| **Cores** | 10 (4 performance + 6 efficiency) |
| **Unified Memory** | 32 GB |
| **OS** | macOS 15.7.2 (Sequoia) |

## Benchmark Results

**Task:** Generate up to 200 tokens for a red-black tree implementation in Python.  
**Metric:** Output tok/s — generation time only, prefill excluded.  
**Method:** 2 warmup runs + 5 timed runs, median reported.  
**Date:** 2026-04-26

Results are split by model family so each table is a direct apples-to-apples comparison of inference methods on the same underlying model.

### Qwen 3.6 — 35B MoE (3B active params/token)

| Method | Quant | tok/s | vs Ollama | Source |
|--------|-------|------:|----------:|--------|
| 🥇 DDTree (MLX) | MLX-int4-DWQ | **28.7** | 2.33× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| 🥈 Plain MLX | MLX-int4-DWQ | 26.9 | 2.19× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| Ollama | GGUF-Q4_K_P | 12.3 | 1.00× | [HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive) |

Memory: ~21.6 GB (20.7 GB model + 0.9 GB DFlash drafter)

### Qwen 3.6 — 27B dense

| Method | Quant | tok/s | vs Ollama | Source |
|--------|-------|------:|----------:|--------|
| DDTree (MLX) | TBD | TBD | TBD | [still wip 26.April](https://x.com/zhijianliu_/status/2048093433680859246?s=20) |
| 🥇 Plain MLX | MLX-int4 | **6.7** | 1.86× | [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) |
| Ollama | GGUF-Q4_K_M | 3.6 | 1.00× | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |

Memory: TBD

### Qwen 3.5 — 27B dense

| Method | Quant | tok/s | vs Ollama | Source |
|--------|-------|------:|----------:|--------|
| 🥇 DDTree (MLX) | MLX-int4 | **5.5** | 1.45× | [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) |
| 🥈 Plain MLX | MLX-int4 | 4.9 | 1.29× | [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) |
| Ollama | GGUF-Q4_K_M | 3.8 | 1.00× | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |

Memory: ~18.2 GB (15 GB model + 3.2 GB DFlash drafter)

### Key observations

- **MLX vs Ollama:** MLX + Metal is up to ~2.2× faster than Ollama + llama.cpp on the same model (26.9 vs 12.3 tok/s on 35B MoE; 4.9 vs 3.8 tok/s on 27B dense).
- **DDTree on top of MLX:** adds ~7% on the 35B MoE and ~12% on the 27B dense — smaller gain on MoE because generation is already fast. DDTree acceptance rate on 35B MoE: **369%** (3.7 draft tokens accepted per cycle).
- **MoE vs dense (cross-family):** the 35B MoE runs at 26.9 tok/s vs 4.9 tok/s for the 27B dense under plain MLX — a 5.5× gap that is entirely architectural. MoE sparsity is a free lunch on Apple Silicon.

---

## Reproduce From Scratch

### Prerequisites

- macOS Sequoia or Sonoma
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- ~40 GB free disk space (models)
- Ollama installed (for Ollama comparisons only): https://ollama.com

### 1. Clone and set up the environment

```bash
git clone <this-repo>
cd local-qwen

# Create Python 3.12 venv and install base deps
uv venv --python 3.12
uv pip install mlx-lm dflash-mlx
uv pip install -e vendor/ddtree-mlx
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
│   ├── bench_compare.py     # 6-way: Ollama vs plain MLX vs DDTree
│   ├── bench_ddtree.py      # 2-way: plain MLX vs DDTree (no Ollama)
│   └── bench_ollama.py      # Ollama-only baseline
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
| Target (27B dense) | `mlx-community/Qwen3.5-27B-4bit` | 15 GB | Baseline dense |
| Drafter (27B dense) | `z-lab/Qwen3.5-27B-DFlash` | 3.2 GB | DDTree drafter |
