#!/usr/bin/env python3
"""
Shared utilities for benchmark scripts: prompts, cooldown, JSON output, greedy sampler.
Fanless M4 MacBook Air requires strict thermal discipline.
"""
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Callable

# ── Constants ──────────────────────────────────────────────────────────────────

HOST = "M4-MBA-32GB"

PRE_BENCH_COOLDOWN_S = 60
INTER_PROMPT_COOLDOWN_S = 30
INTER_CONFIG_COOLDOWN_S = 60
POST_BENCH_COOLDOWN_S = 60

# Multi-prompt set: code, prose, JSON.
PROMPT_CODE = (
    "Implement a red-black tree in Python with insert, delete, and search methods. "
    "Include proper rebalancing logic and type hints."
)

PROMPT_PROSE = (
    "Write a short essay (about 200 words) explaining why memory bandwidth, not raw compute, "
    "is the bottleneck for transformer inference on Apple Silicon. Use vivid analogies and "
    "avoid technical jargon."
)

PROMPT_JSON = (
    "Return a JSON array of 5 fictional novels. Each entry must have keys: title, author, year, "
    "genre, summary (2 sentences). Output a single valid JSON array. No commentary, no markdown fences."
)

PROMPTS = [
    ("code", PROMPT_CODE),
    ("prose", PROMPT_PROSE),
    ("json", PROMPT_JSON),
]

# ── Cooldown helper ────────────────────────────────────────────────────────────

def cooldown(seconds: int, reason: str) -> None:
    """Sleep with reason logged to stdout, for thermal management on fanless M4 Air."""
    print(f"[cooldown {seconds}s] {reason}", flush=True)
    time.sleep(seconds)

# ── Greedy sampler factory ─────────────────────────────────────────────────────

def make_greedy_sampler() -> Callable:
    """Return a greedy (temperature=0) sampler for mlx_lm.stream_generate / generate_step.

    Canonical pattern:
        from benchmark._lib import make_greedy_sampler
        sampler = make_greedy_sampler()
        for token in stream_generate(model, tokenizer, sampler=sampler, ...):
            ...
    """
    from mlx_lm.sample_utils import make_sampler
    return make_sampler(temp=0.0)

# ── JSON output ────────────────────────────────────────────────────────────────

def write_results(payload: dict) -> Path:
    """Write benchmark results to JSON under benchmark/results/<method>_<modelSlug>_<utc-iso>.json.

    Args:
        payload: Dict with keys: ts, host, method, model_label, model_ref, drafter_ref,
                 tree_budget, sampling, warmups, runs_per_prompt, results.

    Returns:
        Path to written file.
    """
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Construct filename: method_modelSlug_timestamp.json
    # modelSlug: model_ref with / → -, remove -4bit, etc. Keep it short.
    method = payload["method"]
    model_ref = payload["model_ref"]
    model_slug = model_ref.replace("/", "-")  # mlx-community/Qwen3.5-27B-4bit → mlx-community-Qwen3.5-27B-4bit

    ts = payload["ts"]  # e.g. "20260426T220000Z"
    filename = f"{method}_{model_slug}_{ts}.json"
    path = results_dir / filename

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path

def get_timestamp_iso8601() -> str:
    """Return current UTC time as ISO 8601 string without colons: 20260426T220000Z."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")
