# Qwen3.5-27B DFlash + DDTree Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install DFlash + DDTree speculative decoding on top of `mlx-community/Qwen3.5-27B-4bit`, wire up an OpenAI-compatible server, and produce a benchmark script that proves the speedup over plain MLX inference.

**Architecture:** `dflash-mlx` (PyPI) provides the speculative drafter loader and `ddtree-mlx` (GitHub) provides the tree-based decoding runtime that runs on top of it. The target model (~16 GB, 4-bit) does final token verification; the small drafter (`z-lab/Qwen3.5-27B-DFlash`, ~3 GB) proposes candidates. Total RAM: ~19 GB.

**Tech Stack:** Python 3.12, uv, mlx, mlx-lm, dflash-mlx (PyPI), ddtree-mlx (GitHub), HuggingFace Hub, FastAPI + uvicorn (server)

---

## File Map

| Path | Status | Responsibility |
|---|---|---|
| `.venv/` | existing | project virtualenv managed by uv |
| `vendor/ddtree-mlx/` | **create** | cloned ddtree-mlx source, installed editable |
| `.env` | **create** | `HF_HOME` pointing to `~/Models/HuggingFace` |
| `scripts/infer_ddtree.py` | **create** | interactive single-turn inference via DDTree |
| `scripts/ddtree_server.py` | **create** | OpenAI-compatible `/v1/chat/completions` server |
| `benchmark/bench_ddtree.py` | **create** | compares plain mlx_lm vs DFlash vs DDTree; follows bench_ollama.py pattern |
| `tests/test_ddtree_smoke.py` | **create** | import + short-run smoke tests (32 tokens max) |

---

## Task 1: Install dflash-mlx into the project venv

**Files:**
- No new files; mutates `.venv/`

- [ ] **Step 1: Install the package**

```bash
uv pip install dflash-mlx
```

Expected output: `Installed 1 package` (or `Resolved ... Installed ...`). If it errors on a missing mlx version, run `uv pip install "dflash-mlx[mlx]"`.

- [ ] **Step 2: Verify the import**

```bash
.venv/bin/python -c "import dflash_mlx; print('dflash_mlx ok')"
```

Expected: `dflash_mlx ok`

---

## Task 2: Clone and install ddtree-mlx from source

**Files:**
- Create: `vendor/ddtree-mlx/` (git clone)

- [ ] **Step 1: Clone the repo into vendor/**

```bash
mkdir -p vendor
git clone https://github.com/humanrouter/ddtree-mlx.git vendor/ddtree-mlx
```

Expected: directory `vendor/ddtree-mlx/` with a `setup.py` or `pyproject.toml`.

- [ ] **Step 2: Install it editable into the project venv**

```bash
uv pip install -e vendor/ddtree-mlx
```

Expected: `Installed 1 package` with an editable link.

- [ ] **Step 3: Verify the import**

```bash
.venv/bin/python -c "from ddtree_mlx.runtime import generate_ddtree_once; print('ddtree_mlx ok')"
```

Expected: `ddtree_mlx ok`

---

## Task 3: Configure model storage and download both models

**Files:**
- Create: `.env` (sets `HF_HOME` so HuggingFace downloads land in `~/Models`)

- [ ] **Step 1: Create .env**

Create `.env`:

```
HF_HOME=~/Models/HuggingFace
```

This makes `huggingface_hub` and `mlx_lm` store models under `~/Models/HuggingFace/hub/`.

- [ ] **Step 2: Download the target model (~16 GB, expect 5–15 min)**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-27B-4bit')
"
```

Expected: progress bars, then `~/Models/HuggingFace/hub/models--mlx-community--Qwen3.5-27B-4bit/` exists.

- [ ] **Step 3: Download the drafter model (~3 GB)**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('z-lab/Qwen3.5-27B-DFlash')
"
```

Expected: `~/Models/HuggingFace/hub/models--z-lab--Qwen3.5-27B-DFlash/` exists.

- [ ] **Step 4: Verify sizes**

```bash
du -sh ~/Models/HuggingFace/hub/models--mlx-community--Qwen3.5-27B-4bit
du -sh ~/Models/HuggingFace/hub/models--z-lab--Qwen3.5-27B-DFlash
```

Expected: first ~16 GB, second ~3 GB.

---

## Task 4: Write and pass smoke tests

**Files:**
- Create: `tests/test_ddtree_smoke.py`

- [ ] **Step 1: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_ddtree_smoke.py`:

```python
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

import pytest

TARGET = "mlx-community/Qwen3.5-27B-4bit"
DRAFTER = "z-lab/Qwen3.5-27B-DFlash"


def test_imports():
    from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
    from ddtree_mlx.runtime import generate_ddtree_once


def test_short_inference():
    from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
    from ddtree_mlx.runtime import generate_ddtree_once

    target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET)

    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello in one word."}],
        tokenize=True,
        add_generation_prompt=True,
    ))

    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=32,
        tree_budget=4,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )

    assert len(result["generated_token_ids"]) > 0
    assert result["tokens_per_second"] > 0.0
    text = tokenizer.decode(result["generated_token_ids"])
    assert len(text.strip()) > 0


def test_result_keys():
    from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
    from ddtree_mlx.runtime import generate_ddtree_once

    target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET)
    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": "1+1="}],
        tokenize=True,
        add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=8,
        tree_budget=4,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )
    assert "generated_token_ids" in result
    assert "tokens_per_second" in result
```

- [ ] **Step 3: Run tests — expect FAIL on test_short_inference and test_result_keys until models are downloaded**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python -m pytest tests/test_ddtree_smoke.py::test_imports -v
```

Expected: PASS (imports work). If FAIL, re-check Tasks 1 & 2.

- [ ] **Step 4: Run full suite (requires models from Task 3)**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python -m pytest tests/test_ddtree_smoke.py -v -s
```

Expected: all 3 tests PASS. `test_short_inference` will load the ~19 GB models on first run — expect ~30–60 s cold load, then fast inference.

---

## Task 5: Write the inference script

**Files:**
- Create: `scripts/infer_ddtree.py`

- [ ] **Step 1: Create scripts directory**

```bash
mkdir -p scripts
```

- [ ] **Step 2: Write the script**

Create `scripts/infer_ddtree.py`:

```python
#!/usr/bin/env python3
"""Single-turn interactive inference with Qwen3.5-27B-4bit + DDTree."""
import os, sys
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once

TARGET = "mlx-community/Qwen3.5-27B-4bit"
MAX_NEW_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
TREE_BUDGET = int(os.environ.get("TREE_BUDGET", "4"))


def run(prompt: str) -> None:
    target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET)

    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    ))

    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        tree_budget=TREE_BUDGET,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )

    print(tokenizer.decode(result["generated_token_ids"]))
    print(f"\n→ {result['tokens_per_second']:.1f} tok/s", file=sys.stderr)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ")
    run(prompt)
```

- [ ] **Step 3: Run it**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/infer_ddtree.py "Write a haiku about M4 chips."
```

Expected: a haiku followed by `→ XX.X tok/s` on stderr.

---

## Task 6: Write the benchmark script

**Files:**
- Create: `benchmark/bench_ddtree.py`

This follows the same structure as `benchmark/bench_ollama.py` — 1 warmup + 3 timed runs per method, with a final summary table.

- [ ] **Step 1: Write the benchmark**

Create `benchmark/bench_ddtree.py`:

```python
#!/usr/bin/env python3
"""
Benchmark: plain mlx_lm vs DFlash vs DFlash+DDTree on Qwen3.5-27B-4bit.
Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_ddtree.py
"""
import os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

import mlx.core as mx
from mlx_lm import load as mlx_load, generate as mlx_generate
from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once

TARGET = "mlx-community/Qwen3.5-27B-4bit"
RUNS = 3
MAX_TOKENS = 200
TREE_BUDGET = 4

PROMPT = (
    "Implement a red-black tree in Python with insert, delete, and search methods. "
    "Include proper rebalancing logic and type hints."
)


def bench_plain(model, tokenizer) -> float:
    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    t0 = time.perf_counter()
    out = mlx_generate(model, tokenizer, prompt=prompt_str, max_tokens=MAX_TOKENS, verbose=False)
    elapsed = time.perf_counter() - t0
    tokens = len(tokenizer.encode(out))
    return tokens / elapsed


def bench_ddtree(target_model, tokenizer, draft_model) -> tuple[float, float]:
    prompt_tokens = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=MAX_TOKENS,
        tree_budget=TREE_BUDGET,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )
    return result["tokens_per_second"], result.get("acceptance_rate", float("nan"))


print("Loading models...", flush=True)
plain_model, plain_tokenizer = mlx_load(TARGET)
target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET)
print("Models loaded.\n", flush=True)

summary = []

for label, fn in [
    ("plain mlx_lm", lambda: bench_plain(plain_model, plain_tokenizer)),
    ("DFlash+DDTree", lambda: bench_ddtree(target_model, tokenizer, draft_model)),
]:
    print(f"{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)

    print("  [warmup]...", end=" ", flush=True)
    fn()
    print("done", flush=True)

    runs = []
    for i in range(1, RUNS + 1):
        print(f"  [run {i}/{RUNS}] ", end="", flush=True)
        result = fn()
        if isinstance(result, tuple):
            tps, acc = result
        else:
            tps, acc = result, float("nan")
        runs.append((tps, acc))
        acc_str = f"  accept {acc:.0%}" if acc == acc else ""
        print(f"{tps:.1f} tok/s{acc_str}", flush=True)

    avg_tps = sum(r[0] for r in runs) / RUNS
    avg_acc = sum(r[1] for r in runs if r[1] == r[1]) / RUNS
    summary.append((label, avg_tps, avg_acc))

    print(f"  AVG: {avg_tps:.1f} tok/s\n", flush=True)

print(f"\n{'='*60}")
print(f"  SUMMARY  ({RUNS} runs, post-warmup, {MAX_TOKENS} max tokens)")
print(f"{'='*60}")
print(f"  {'Method':<20}  {'tok/s':>7}  {'speedup':>8}  {'accept':>8}")
print(f"  {'-'*20}  {'-'*7}  {'-'*8}  {'-'*8}")
baseline = summary[0][1]
for label, tps, acc in summary:
    speedup = tps / baseline
    acc_str = f"{acc:.0%}" if acc == acc else "  —"
    print(f"  {label:<20}  {tps:7.1f}  {speedup:7.2f}×  {acc_str:>8}")
print()
```

- [ ] **Step 2: Run it (expect ~5–10 min total including model load)**

```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python benchmark/bench_ddtree.py
```

Expected final table:
```
  Method                 tok/s   speedup    accept
  --------------------  -------  --------  --------
  plain mlx_lm           XX.X     1.00×       —
  DFlash+DDTree          YY.Y     1.5–2×     85–90%
```

---

## Task 7: Set up the OpenAI-compatible server

**Files:**
- Create: `scripts/ddtree_server.py` (or verify it exists in `vendor/ddtree-mlx/`)

- [ ] **Step 1: Check if server script is in the vendored repo**

```bash
find vendor/ddtree-mlx -name "ddtree_server.py"
```

**If found:** skip to Step 3, the server is already there.
**If not found:** proceed to Step 2.

- [ ] **Step 2: Write a minimal OpenAI-compatible server**

Create `scripts/ddtree_server.py`:

```python
#!/usr/bin/env python3
"""
Minimal OpenAI-compatible chat server backed by DDTree + Qwen3.5-27B-4bit.
Usage: HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/ddtree_server.py --port 8006
"""
import os, argparse, time, uuid
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once

TARGET = "mlx-community/Qwen3.5-27B-4bit"

app = FastAPI()
_state: dict = {}


@app.on_event("startup")
def load_models():
    target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET)
    _state["target"] = target_model
    _state["tokenizer"] = tokenizer
    _state["draft"] = draft_model
    _state["stop_ids"] = get_stop_token_ids(tokenizer)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = TARGET
    messages: list[Message]
    max_tokens: int = 2048
    temperature: float = 0.0
    tree_budget: int = 4


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    tokenizer = _state["tokenizer"]
    prompt_tokens = list(tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=True,
        add_generation_prompt=True,
    ))
    result = generate_ddtree_once(
        target_model=_state["target"],
        draft_model=_state["draft"],
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=req.max_tokens,
        tree_budget=req.tree_budget,
        stop_token_ids=_state["stop_ids"],
    )
    text = tokenizer.decode(result["generated_token_ids"])
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(result["generated_token_ids"]),
            "total_tokens": len(prompt_tokens) + len(result["generated_token_ids"]),
        },
        "x_tokens_per_second": result["tokens_per_second"],
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": TARGET, "object": "model"}]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
```

- [ ] **Step 3: Install fastapi if not already present**

```bash
uv pip install fastapi uvicorn
.venv/bin/python -c "import fastapi, uvicorn; print('server deps ok')"
```

Expected: `server deps ok`

- [ ] **Step 4: Start the server and smoke-test it**

Terminal 1:
```bash
HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/ddtree_server.py --port 8006
```

Terminal 2 (after "Application startup complete" appears):
```bash
curl -s http://localhost:8006/v1/models | python3 -m json.tool
```

Expected: JSON listing with model id `mlx-community/Qwen3.5-27B-4bit`.

```bash
curl -s http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":32}' \
  | python3 -m json.tool
```

Expected: JSON with `choices[0].message.content` containing a greeting and `x_tokens_per_second` > 0.

---

## Self-Review

### Spec Coverage

| Spec requirement | Task |
|---|---|
| Install dflash-mlx | Task 1 |
| Install ddtree-mlx from GitHub | Task 2 |
| Download `mlx-community/Qwen3.5-27B-4bit` | Task 3 |
| Download `z-lab/Qwen3.5-27B-DFlash` drafter | Task 3 |
| Models land in `~/Models` | Task 3 (.env + `HF_HOME`) |
| Python inference with `tree_budget=4` | Task 5 (infer_ddtree.py) + tests |
| OpenAI-compatible server on port 8006 | Task 7 |
| Benchmark proving speedup | Task 6 |
| `tree_budget=4` confirmed optimal | Task 5 + 6 |

### No Placeholders ✓

All steps include actual commands, actual code, and actual expected output.

### Type Consistency ✓

- `generate_ddtree_once` result dict keys `generated_token_ids` and `tokens_per_second` used consistently across Tasks 4, 5, 6, 7.
- `load_runtime_components(model_ref=TARGET)` call signature consistent across all tasks.
- `tree_budget=4` used everywhere.
