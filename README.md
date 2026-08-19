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

**Session:** `20260819T044632Z` (2026-08-19, one `session_id` for all families).  
**Prompts:** 3 coding prompts (`code-algo` · `code-async` · `code-cache`) at up to 200 generated tokens each.  
**Sampling:** greedy (temperature=0, seed=42 where supported), `enable_thinking=False`.  
**Metric:** decode tok/s (wall clock after first token) and TTFT (ms); per-run token counts in JSON.  
**Method:** `benchmark.bench_session` runs every backend in **one thermal session** per model family, **MLX first**: plain-mlx → vllm-mlx → ddtree → ollama → llama.cpp. 2 warmups + 5 timed runs, 60 s between runs, median reported. **Avg tok/s** is the mean of per-prompt medians.  
**Software this session:** mlx 0.32.1 · mlx-lm 0.31.3 · Ollama 0.32.14 · vllm-mlx 0.2.9.

Summarize: `.venv/bin/python -m benchmark.summarize --session-id 20260819T044632Z`

### Qwen 3.8 — 27B dense

| Method | Quant | Avg tok/s | TTFT (ms) | vs Ollama | Source |
|--------|-------|----------:|----------:|----------:|--------|
| 🥇 Ollama | NVFP4 (`qwen3.8:27b-mlx`)† | **17.2** | 527 | 1.00× | `qwen3.8:27b-mlx` (same digest as `27b-nvfp4`) |
| Plain MLX | MLX-int4 | 6.6 | 1658 | 0.39× | [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) |
| vllm-mlx | MLX-int4 | 6.6 | 1621 | 0.38× | same weights |
| llama.cpp | GGUF-Q4_K_XL | 4.4 | 1887 | 0.26× | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| DDTree | DFlash2 | — | — | skipped | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) downloaded; `ddtree-mlx` returned `draft_model=None` |

Token parity vs plain-mlx: ✓ for MLX/vllm/llama.cpp. † Session `20260819T044632Z` used Ollama `27b-mlx`, which is **NVFP4**, not mlx-community int4 and not GGUF Q4.

#### Fair 4-bit + official DFlash2 — session `20260819T192413Z`

`dflash generate mlx` path (`dflash.model_mlx.stream_generate`, `--draft-bits 4`, `block_size` capped at 5 for int4 matmul). Same `session_id` for both rows. Do not mix with the NVFP4 Ollama row above.

| Method | Quant | Avg tok/s | TTFT (ms) | vs Q4 Ollama | Source |
|--------|-------|----------:|----------:|-------------:|--------|
| 🥇 DFlash2 MLX | MLX-int4 + draft 4-bit | **16.6** | 1595 | 2.91× | [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) + [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| Ollama | GGUF-Q4_K_M | 5.7 | 755 | 1.00× | `qwen3.8:27b-q4_K_M` (full 27.3B) |

Acceptance ~4.2 tokens/block. ~2.5× the 6.6 tok/s plain-MLX baseline from session `044632Z` (different thermal window; same MLX-int4 file). Ollama Q4 is in the same ballpark as 3.6-27B Q4 (5.6). NVFP4 `27b-mlx` (17.2) is a different quant — not this comparison.

### Qwen 3.6 — 35B MoE

| Method | Quant | Avg tok/s | TTFT (ms) | vs Ollama | Source |
|--------|-------|----------:|----------:|----------:|--------|
| 🥇 Plain MLX | MLX-int4-DWQ | **32.8** | 702 | 1.10× | [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ) |
| vllm-mlx | MLX-int4-DWQ | 32.6 | 723 | 1.09× | same weights |
| Ollama | GGUF-Q4_K_M | 29.9 | 287 | 1.00× | `qwen3.6:35b` |
| DDTree | DFlash | — | — | skipped | drafter on disk; `ddtree-mlx` returned `draft_model=None` |
| llama.cpp | — | — | — | skipped | no GGUF in this family |

Memory: ~21.6 GB (20.7 GB model + 0.9 GB DFlash drafter) when DDTree loads.

Ollama 35B decode jumped vs the May 2025 mixed-session tables (16.6 → 29.9 tok/s) after the Ollama 0.21 → 0.32 upgrade. MLX’s lead is now ~10%, not ~2×. Quant still differs (DWQ vs Q4_K_M).

#### tree_budget sweep — DDTree on Qwen3.6-35B-MoE [MLX-int4-DWQ]

Prior sweep (2026-04-27) used mixed `code/prose/json` prompts; **not re-run** under the current `coding` protocol or this session (35B DDTree skipped).

| Budget | code (t/s) | prose (t/s) | json (t/s) | Avg t/s | Avg accept (tok/cycle) |
|-------:|-----------:|------------:|-----------:|--------:|-----------------------:|
| 2 | 28.5 | 25.7 | 27.8 | 27.3 | 2.6 |
| **3** | **29.7** | **26.0** | **29.8** | **28.5** | 3.1 |
| 4 | 29.0 | 24.2 | 26.4 | 26.5 | 3.6 |
| 5 | 29.7 | 23.0 | 24.9 | 25.9 | 3.8 |
| 6 | 28.9 | 22.9 | 25.4 | 25.7 | 4.2 |

**Default budget: 3.**

### Qwen 3.6 — 27B dense

| Method | Quant | Avg tok/s | TTFT (ms) | Accept | vs Ollama | Source |
|--------|-------|----------:|----------:|-------:|----------:|--------|
| 🥇 DDTree (MLX, b=3) | MLX-int4 | **11.0** | 1243 | 3.4 tok/cycle | 1.95× | target + [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| Plain MLX | MLX-int4 | 6.6 | 1671 | — | 1.18× | [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) |
| vllm-mlx | MLX-int4 | 6.6 | 1637 | — | 1.17× | same weights |
| Ollama | GGUF-Q4_K_M | 5.6 | 759 | — | 1.00× | `qwen3.6:27b` |
| llama.cpp MTP (n=2) | GGUF-Q4_K_XL | 5.3 | 1956 | 90% draft | 0.94× | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) |
| llama.cpp baseline | GGUF-Q4_K_XL | 5.1 | 1783 | — | 0.90× | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) |
| Plain MLX (OptiQ)† | MLX-OptiQ-4bit | 5.6 | 1798 | — | 0.99× | [mlx-community/Qwen3.6-27B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-OptiQ-4bit) |

† OptiQ is an extras-only plain-MLX run in the same session (`family_id` unset in JSON). vs Ollama is vs the 3.6-27b Ollama row.

DDTree token parity vs plain-mlx: ✗ on `code-algo` and `code-async`, ✓ on `code-cache`. Treat the 1.95× as speed, not identical output.

No thermal split this session: plain MLX and vllm-mlx held ~6.6 tok/s on all three prompts (unlike the May 27B run, which throttled on `code-cache`).

### Key observations

- **Official DFlash2 is the 3.8 MLX speed path:** 16.6 tok/s (`dflash2-mlx`, session `20260819T192413Z`) vs 5.7 tok/s Ollama Q4_K_M and 6.6 tok/s plain MLX. Ollama cannot load DFlash2; its 17.2 tok/s `27b-mlx` row is **NVFP4**, not int4.
- **Ollama caught up on 35B MoE:** 29.9 tok/s vs MLX 32.8 (MLX +10%). The old “MLX ~2× Ollama” claim was against Ollama 0.21 / mixed sessions.
- **DDTree wins 3.6-27B dense:** 11.0 vs plain MLX 6.6 (+65%) at ~3.4 accepted tokens/cycle. 35B DDTree did **not** run session `044632Z` (`draft_model=None`). 3.8 uses official DFlash2 (`dflash2-mlx`), not DDTree tree-search, in session `192413Z`.
- **vllm-mlx matches plain MLX decode** on every family that both ran (35B 32.6 vs 32.8; both 27B dense models 6.6).
- **llama.cpp MTP is a small bump over its own baseline** (5.3 vs 5.1, ~90% draft accept) and is behind Ollama and MLX on 3.6-27B. No MTP GGUF was benched for 3.8.
- **OptiQ is no longer a disaster:** 5.6 vs uniform int4 6.6 tok/s (−15%), vs 2.6 tok/s in the May mixed-session tables.
- **MoE vs dense (plain MLX):** 32.8 tok/s (35B) vs 6.6 tok/s (both 27B dense models) — still ~5× on this chip.

### Methodology / caveats

- **Single-session rule:** published comparisons must share the same `session_id` in JSON (set automatically by `run_all_benches.sh`). Cross-family tables are `20260819T044632Z`; the 3.8 DFlash2 vs Q4 Ollama table is `20260819T192413Z`.
- All runs on a fanless M4 MacBook Air (32 GB UMA). 60 s between timed runs; 90 s thermal floor between methods; 180 s between families.
- Software this session: mlx 0.32.1, mlx-lm 0.31.3, Ollama 0.32.14, vllm-mlx 0.2.9. Older pins and May 2025 mixed-session JSONs live in [`benchmark/ENVIRONMENT.md`](benchmark/ENVIRONMENT.md).
- Ollama uses `/api/chat` with `think: false` (localhost HTTP overhead). 3.8 fair 4-bit tag is `qwen3.8:27b-q4_K_M`; `27b-mlx` is NVFP4 (same digest as `27b-nvfp4`). 3.6 tags are GGUF-Q4_K_M.
- DDTree TTFT: wall-clock proportional to `prefill_us/elapsed_us`; internal value stored as `ttft_prefill_us_ms`.
- Plain MLX: `mx.random.seed(42)` + greedy sampler; Ollama/llama.cpp: `seed=42`. Plain MLX loads **target only** (no drafter resident).
- Token parity: first 32 output token IDs vs plain-mlx (may differ across quants and speculative paths).
- Quant mismatch: MLX DWQ/int4 vs Ollama Q4_K_M vs llama.cpp Q4_K_XL — decode ratios are approximate except where both sides are the same MLX 4bit file.

---

## Reproduce From Scratch

### Prerequisites

- macOS Sequoia or Sonoma
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- Disk space: ~21 GB (35B MoE MLX + drafter) · ~80 GB+ (full 3.6+3.8 suite: MLX, DFlash, Unsloth GGUF, Ollama)
- Ollama 0.32+ (Qwen 3.8 library tags): https://ollama.com

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

**Full fair suite (recommended — MLX first, single session_id):**

```bash
./benchmark/run_all_benches.sh
# Summary: /tmp/run_all_benches_summary_<session_id>.log
```

**One model family:**

```bash
BENCH_FAMILY=3.8-27b HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_session
BENCH_FAMILY=3.6-35b-moe HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_session
```

**Summarize a session:**

```bash
.venv/bin/python -m benchmark.summarize --session-id 20260819T044632Z
```

> **Memory warning:** The benchmark loads MLX models only after Ollama models are explicitly unloaded. Do not add new Ollama models to `OLLAMA_MODELS` if their GGUF size + 21.6 GB exceeds 32 GB.

---

## Project Structure

```
local-qwen/
├── benchmark/
│   ├── models.py                     # Model family registry (3.6 + 3.8)
│   ├── _lib.py                       # Shared prompts, timing, parity, JSON writer
│   ├── bench_session.py              # Fair unified session (MLX first) ★
│   ├── summarize.py                  # Tables from one session_id
│   ├── bench_compare.py              # Legacy partial: Ollama vs MLX/DDTree (35B)
│   ├── bench_ddtree.py               # Legacy: plain MLX vs DDTree
│   ├── bench_ollama.py               # Ollama-only probe
│   ├── bench_extras.py               # Plain MLX OptiQ variant
│   ├── bench_tree_budget_sweep.py    # DDTree tree_budget sweep
│   ├── bench_vllm.py                 # vllm-mlx standalone
│   ├── bench_llamacpp_mtp.py         # llama.cpp baseline vs MTP
│   ├── complete_unattended.sh        # Downloads + run_all_benches (retries, caffeinate)
│   ├── run_all_benches.sh            # Orchestrator → bench_session per family
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
| Target (27B dense, 3.8) | `mlx-community/Qwen3.8-27B-4bit` | ~16 GB | Newest dense; mlx-lm ~6.6 tok/s this session |
| Drafter (27B dense, 3.8) | `z-lab/Qwen3.8-27B-DFlash2` | ~3.6 GB | Official `dflash2-mlx` + DDTree drafter; not an Ollama target |
| GGUF (27B, 3.8) | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` | ~15 GB | llama.cpp 4-bit |
| Ollama (27B, 3.8, 4-bit) | `qwen3.8:27b-q4_K_M` | ~17 GB | Full 27.3B Q4_K_M; 5.7 tok/s in session `192413Z` |
| Ollama (27B, 3.8, NVFP4) | `qwen3.8:27b-mlx` | ~18 GB | Same digest as `27b-nvfp4`; 17.2 tok/s in session 044632Z |
| Target (35B MoE) | `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` | 20.7 GB | Best MLX decode (32.8 tok/s) |
| Drafter (35B MoE) | `z-lab/Qwen3.6-35B-A3B-DFlash` | 0.9 GB | On disk; DDTree skipped this session |
| Target (27B dense, 3.6) | `mlx-community/Qwen3.6-27B-4bit` | ~15 GB | Uniform int4 |
| Drafter (27B dense, 3.6) | `z-lab/Qwen3.6-27B-DFlash` | — | DDTree worked this session (11.0 tok/s) |
| Target (27B dense, 3.6 OptiQ) | `mlx-community/Qwen3.6-27B-OptiQ-4bit` | ~16 GB | Mixed-precision; extras-only |
| GGUF baseline (27B, 3.6) | `unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL` | ~15 GB | llama.cpp 4-bit baseline |
| GGUF + MTP heads (27B, 3.6) | `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` | ~15 GB | Bundled MTP prediction heads |
| Target (27B dense, 3.5) | `mlx-community/Qwen3.5-27B-4bit` | 15 GB | Older dense; not in this session |
| Drafter (27B dense, 3.5) | `z-lab/Qwen3.5-27B-DFlash` | 3.2 GB | DDTree drafter for 3.5 |
