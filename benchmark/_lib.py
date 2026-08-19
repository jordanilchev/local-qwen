#!/usr/bin/env python3
"""
Shared utilities for benchmark scripts: prompts, cooldown, JSON output, timing helpers.
Fanless M4 MacBook Air requires strict thermal discipline.

Timing convention (all backends):
  TTFT   = wall time from request/call start to first generated token (or prefill end for batch paths)
  Decode = (completion_tokens - 1) / wall time after TTFT

DDTree has no streaming hook; TTFT is wall-clock proportional to internal prefill_us / elapsed_us.
Internal prefill_us is stored separately as ttft_prefill_us_ms for diagnostics.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# ── Constants ──────────────────────────────────────────────────────────────────

HOST = "M4-MBA-32GB"

PRE_BENCH_COOLDOWN_S = 60
INTER_RUN_COOLDOWN_S = 60
INTER_PROMPT_COOLDOWN_S = 60
INTER_CONFIG_COOLDOWN_S = 90
POST_BENCH_COOLDOWN_S = 60
THERMAL_LOAD_MAX = 1.5
THERMAL_MAX_WAIT_S = 600

WARMUPS = 2
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200
DEFAULT_TREE_BUDGET = 3  # best on M4 per bench_tree_budget_sweep (was 4)

# Disable thinking mode so Qwen3.x models match across MLX / Ollama / llama.cpp paths.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

DEFAULT_SAMPLING: dict[str, Any] = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": MAX_TOKENS,
    "enable_thinking": False,
}

PROMPT_SET = "coding"

# Legacy mixed prompt set (kept for comparing against older JSON results).
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

PROMPT_CODE_ASYNC = (
    "Write a Python async HTTP client class using aiohttp with automatic retry "
    "(exponential backoff on 5xx and 429 errors, max 3 attempts), a configurable "
    "per-request timeout, and a context manager interface. Include type hints."
)
PROMPT_CODE_CACHE = (
    "Implement a thread-safe LRU cache in Python using collections.OrderedDict. "
    "Support get(key, default=None), put(key, value), and delete(key) with O(1) "
    "amortized complexity and a max_size constructor parameter. Include type hints."
)
CODING_PROMPTS = [
    ("code-algo", PROMPT_CODE),
    ("code-async", PROMPT_CODE_ASYNC),
    ("code-cache", PROMPT_CODE_CACHE),
]

# Targets with official MLX DFlash drafters (dflash_mlx registry + local fallback).
DEFAULT_DRAFT_BY_TARGET: dict[str, str] = {
    "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "mlx-community/Qwen3.6-27B-4bit": "z-lab/Qwen3.6-27B-DFlash",
    "mlx-community/Qwen3.5-27B-4bit": "z-lab/Qwen3.5-27B-DFlash",
    "mlx-community/Qwen3.8-27B-4bit": "z-lab/Qwen3.8-27B-DFlash2",
}

# Warmup prompts differ from timed prompts so prefix-cache priming does not skew TTFT.
WARMUP_PROMPTS = [
    "List three primary colors.",
    "What is 17 + 25? Reply with the number only.",
    "Name one planet in our solar system.",
]


OUTPUT_TOKEN_SAMPLE = 32


@dataclass(frozen=True)
class RunMetrics:
    ttft_ms: float
    decode_tps: float
    completion_tokens: int
    total_ms: float
    ttft_prefill_us_ms: float | None = None
    output_token_ids: tuple[int, ...] = ()


# ── Chat formatting ────────────────────────────────────────────────────────────

def chat_messages(user_text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": user_text}]


def apply_chat_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        chat_messages(user_text),
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )


def apply_chat_prompt_tokens(tokenizer, user_text: str) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            chat_messages(user_text),
            tokenize=True,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
    )


def resolve_draft_ref(model_ref: str, draft_ref: Optional[str] = None) -> Optional[str]:
    """Resolve drafter HF ref: explicit override → dflash registry → local fallback map."""
    if draft_ref:
        return draft_ref
    try:
        from dflash_mlx.generate import resolve_optional_draft_ref

        resolved = resolve_optional_draft_ref(model_ref, None)
        if resolved:
            return resolved
    except ImportError:
        pass
    return DEFAULT_DRAFT_BY_TARGET.get(model_ref)


def load_ddtree_runtime(model_ref: str, draft_ref: Optional[str] = None):
    """Load target + drafter for DDTree; fail fast if the drafter cannot be loaded."""
    from dflash_mlx.generate import get_stop_token_ids, load_runtime_components

    from benchmark.dflash2 import is_dflash2_ref, load_dflash2_draft

    resolved = resolve_draft_ref(model_ref, draft_ref)
    if not resolved:
        raise RuntimeError(
            f"DDTree requires a drafter but none is registered for target {model_ref!r}. "
            f"Set DRAFT= explicitly or add benchmark._lib.DEFAULT_DRAFT_BY_TARGET[{model_ref!r}]."
        )

    if is_dflash2_ref(resolved):
        from mlx_lm import load as mlx_load

        target, tokenizer = mlx_load(model_ref)
        draft = load_dflash2_draft(resolved)
        draft.bind(target)
        return target, tokenizer, draft, resolved, get_stop_token_ids(tokenizer)

    target, tokenizer, draft, loaded_ref = load_runtime_components(
        model_ref=model_ref,
        draft_ref=resolved,
    )
    if draft is None:
        raise RuntimeError(
            f"DDTree requires a drafter but draft_model is None for target {model_ref!r} "
            f"(resolved draft ref: {resolved!r}). dflash-mlx returned None "
            f"(exceptions are swallowed in load_runtime_components)."
        )
    return target, tokenizer, draft, loaded_ref, get_stop_token_ids(tokenizer)


# ── Timing helpers ─────────────────────────────────────────────────────────────

def set_benchmark_seed(seed: int | None = None) -> None:
    """Best-effort deterministic sampling on MLX backends (no seed param in make_sampler)."""
    import mlx.core as mx

    mx.random.seed(seed if seed is not None else DEFAULT_SAMPLING["seed"])


def wall_ttft_from_ddtree_phases(total_ms: float, prefill_us: float, elapsed_us: float) -> float:
    """Map DDTree internal phase split onto wall clock for fair TTFT vs streaming backends."""
    if elapsed_us <= 0:
        return total_ms
    return total_ms * (prefill_us / elapsed_us)


def sample_output_tokens(token_ids: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(token_ids[:OUTPUT_TOKEN_SAMPLE])


def decode_tps_from_wallclock(completion_tokens: int, ttft_ms: float, total_ms: float) -> float:
    """Decode tok/s excluding TTFT; returns 0 when fewer than 2 tokens."""
    if completion_tokens <= 1:
        return 0.0
    decode_s = max((total_ms - ttft_ms) / 1000.0, 0.0)
    return (completion_tokens - 1) / decode_s if decode_s > 0 else 0.0


def median_run_field(runs: list[RunMetrics], field: str) -> float:
    return statistics.median(getattr(r, field) for r in runs)


def runs_to_result_dict(runs: list[RunMetrics], extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ttft_ms_runs": [r.ttft_ms for r in runs],
        "decode_tps_runs": [r.decode_tps for r in runs],
        "completion_tokens_runs": [r.completion_tokens for r in runs],
        "total_ms_runs": [r.total_ms for r in runs],
        "ttft_ms_median": median_run_field(runs, "ttft_ms"),
        "decode_tps_median": median_run_field(runs, "decode_tps"),
        "completion_tokens_median": int(median_run_field(runs, "completion_tokens")),
        "total_ms_median": median_run_field(runs, "total_ms"),
        "output_token_ids_sample_runs": [list(r.output_token_ids) for r in runs],
    }
    prefill_runs = [r.ttft_prefill_us_ms for r in runs if r.ttft_prefill_us_ms is not None]
    if prefill_runs:
        payload["ttft_prefill_us_ms_runs"] = prefill_runs
    if extra:
        payload.update(extra)
    return payload


# ── Cooldown + suite runner ────────────────────────────────────────────────────

def _pmset_therm_line() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        text = (out.stdout or "") + (out.stderr or "")
        return " ".join(text.split())[:200]
    except Exception as exc:
        return f"pmset unavailable ({exc})"


def wait_for_thermal_idle(reason: str, *, min_s: int | None = None) -> None:
    """Sleep until 1-min loadavg is cool enough, after a minimum cooldown (fanless M4)."""
    floor = INTER_CONFIG_COOLDOWN_S if min_s is None else min_s
    cooldown(floor, f"thermal floor before {reason}")
    deadline = time.time() + THERMAL_MAX_WAIT_S
    while time.time() < deadline:
        load1, _, _ = os.getloadavg()
        therm = _pmset_therm_line()
        print(f"[thermal] load1={load1:.2f} therm={therm!r}", flush=True)
        if load1 <= THERMAL_LOAD_MAX and "CPU_Speed_Limit" not in therm:
            return
        time.sleep(15)
    print(f"[thermal] max wait {THERMAL_MAX_WAIT_S}s reached; continuing ({reason})", flush=True)


def cooldown(seconds: int, reason: str) -> None:
    print(f"[cooldown {seconds}s] {reason}", flush=True)
    time.sleep(seconds)


def run_prompt_suite(
    label: str,
    bench_fn: Callable[[], RunMetrics],
    warmup_fn: Callable[[str], None],
    warmup_prompts: list[str] | None = None,
    warmups: int = WARMUPS,
    runs: int = RUNS_PER_PROMPT,
) -> dict[str, Any]:
    """Warm up on distinct prompts, then timed runs with inter-run cooldown."""
    warmups_list = warmup_prompts or WARMUP_PROMPTS

    print(f"\n{'='*70}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*70}", flush=True)

    for w in range(1, warmups + 1):
        wp = warmups_list[(w - 1) % len(warmups_list)]
        print(f"  [warmup {w}/{warmups}]...", end=" ", flush=True)
        warmup_fn(wp)
        print("done", flush=True)

    timed_runs: list[RunMetrics] = []
    for i in range(1, runs + 1):
        if i > 1:
            cooldown(INTER_RUN_COOLDOWN_S, f"between timed runs ({label})")
        print(f"  [run {i}/{runs}] ", end="", flush=True)
        metrics = bench_fn()
        timed_runs.append(metrics)
        print(
            f"TTFT {metrics.ttft_ms:.1f}ms  decode {metrics.decode_tps:.1f} tok/s"
            f"  ({metrics.completion_tokens} tok)",
            flush=True,
        )

    return runs_to_result_dict(timed_runs)


# ── MLX plain stream_generate ──────────────────────────────────────────────────

def load_mlx_target(model_ref: str):
    """Load MLX target only (no drafter resident — fair plain-MLX baseline)."""
    from mlx_lm import load

    return load(model_ref)


def make_greedy_sampler() -> Callable:
    from mlx_lm.sample_utils import make_sampler

    set_benchmark_seed()
    return make_sampler(temp=0.0)


def bench_ddtree(
    target_model,
    tokenizer,
    draft_model,
    stop_ids,
    prompt_text: str,
    tree_budget: int = DEFAULT_TREE_BUDGET,
    max_tokens: int = MAX_TOKENS,
) -> tuple[RunMetrics, float]:
    """Returns (metrics, avg_acceptance). TTFT from prefill_us; decode uses wall clock after TTFT."""
    from ddtree_mlx.runtime import generate_ddtree_once

    prompt_tokens = apply_chat_prompt_tokens(tokenizer, prompt_text)
    t0 = time.perf_counter()
    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_tokens,
        tree_budget=tree_budget,
        stop_token_ids=stop_ids,
    )
    t_end = time.perf_counter()

    gen_tokens = int(result["generation_tokens"])
    prefill_us = float(result["prefill_us"])
    elapsed_us = float(result["elapsed_us"])
    total_ms = (t_end - t0) * 1000.0
    ttft_ms = wall_ttft_from_ddtree_phases(total_ms, prefill_us, elapsed_us)
    decode_tps = decode_tps_from_wallclock(gen_tokens, ttft_ms, total_ms)
    avg_acceptance = float(result.get("avg_acceptance", float("nan")))
    out_ids = sample_output_tokens(result.get("generated_token_ids", []))
    return (
        RunMetrics(
            ttft_ms,
            decode_tps,
            gen_tokens,
            total_ms,
            ttft_prefill_us_ms=prefill_us / 1000.0,
            output_token_ids=out_ids,
        ),
        avg_acceptance,
    )


def bench_vllm_engine(
    engine, tokenizer, prompt_text: str, max_tokens: int = MAX_TOKENS
) -> RunMetrics:
    import uuid
    from vllm_mlx import Request, SamplingParams
    from vllm_mlx.mlx_streams import bind_generation_streams

    set_benchmark_seed()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0, top_k=0)
    prompt_str = apply_chat_prompt(tokenizer, prompt_text)
    rid = str(uuid.uuid4())
    request = Request(request_id=rid, prompt=prompt_str, sampling_params=sp)

    bind_generation_streams()
    engine.scheduler.add_request(request)

    t0 = time.perf_counter()
    t_first: float | None = None
    count = 0
    try:
        while engine.scheduler.has_requests():
            out = engine.scheduler.step()
            for ro in out.outputs:
                if ro.request_id != rid:
                    continue
                n = len(ro.new_token_ids)
                if n > 0 and t_first is None:
                    t_first = time.perf_counter()
                count += n
    finally:
        engine.scheduler.remove_finished_request(rid)

    t_end = time.perf_counter()
    if t_first is None:
        return RunMetrics(0.0, 0.0, count, (t_end - t0) * 1000.0, output_token_ids=())

    ttft_ms = (t_first - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    decode_tps = decode_tps_from_wallclock(count, ttft_ms, total_ms)
    return RunMetrics(ttft_ms, decode_tps, count, total_ms, output_token_ids=())


def bench_mlx_plain(model, tokenizer, prompt_text: str, max_tokens: int = MAX_TOKENS) -> RunMetrics:
    from mlx_lm import stream_generate

    prompt_str = apply_chat_prompt(tokenizer, prompt_text)
    sampler = make_greedy_sampler()

    t0 = time.perf_counter()
    t_first: float | None = None
    count = 0
    token_ids: list[int] = []
    for response in stream_generate(
        model, tokenizer, prompt=prompt_str, max_tokens=max_tokens, sampler=sampler
    ):
        if t_first is None:
            t_first = time.perf_counter()
        count += 1
        token_ids.append(int(response.token))

    t_end = time.perf_counter()
    if t_first is None:
        return RunMetrics(0.0, 0.0, 0, (t_end - t0) * 1000.0, output_token_ids=())

    ttft_ms = (t_first - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    decode_tps = decode_tps_from_wallclock(count, ttft_ms, total_ms)
    return RunMetrics(
        ttft_ms,
        decode_tps,
        count,
        total_ms,
        output_token_ids=sample_output_tokens(token_ids),
    )


def bench_dflash2_mlx(
    model,
    tokenizer,
    draft,
    prompt_text: str,
    *,
    block_size: int,
    max_tokens: int = MAX_TOKENS,
) -> tuple[RunMetrics, float]:
    """Official DFlash2 linear+selector path (`dflash.model_mlx.stream_generate`)."""
    from dflash.model_mlx import stream_generate

    set_benchmark_seed()
    prompt_str = apply_chat_prompt(tokenizer, prompt_text)

    t0 = time.perf_counter()
    t_first: float | None = None
    token_ids: list[int] = []
    accepted_chunks: list[float] = []
    for response in stream_generate(
        model,
        draft,
        tokenizer,
        prompt_str,
        block_size=block_size,
        max_tokens=max_tokens,
        temperature=0.0,
    ):
        if not response.tokens:
            continue
        if t_first is None:
            t_first = time.perf_counter()
        token_ids.extend(int(t) for t in response.tokens)
        if response.accepted is not None:
            accepted_chunks.append(float(response.accepted))

    t_end = time.perf_counter()
    count = len(token_ids)
    if t_first is None:
        return (
            RunMetrics(0.0, 0.0, 0, (t_end - t0) * 1000.0, output_token_ids=()),
            float("nan"),
        )

    ttft_ms = (t_first - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    decode_tps = decode_tps_from_wallclock(count, ttft_ms, total_ms)
    avg_acceptance = (
        statistics.mean(accepted_chunks) if accepted_chunks else float("nan")
    )
    return (
        RunMetrics(
            ttft_ms,
            decode_tps,
            count,
            total_ms,
            output_token_ids=sample_output_tokens(token_ids),
        ),
        avg_acceptance,
    )


# ── Ollama chat streaming ──────────────────────────────────────────────────────

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def wait_for_ollama_server(timeout_s: int = 60) -> None:
    url = f"{OLLAMA_HOST}/api/tags"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception as exc:
            last_err = exc
            time.sleep(1)
    raise RuntimeError(f"Ollama not reachable at {OLLAMA_HOST} after {timeout_s}s: {last_err}")


def list_ollama_models() -> set[str]:
    wait_for_ollama_server()
    with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=10) as resp:
        data = json.loads(resp.read())
    names: set[str] = set()
    for entry in data.get("models", []):
        name = entry.get("name", "")
        if name:
            names.add(name)
            if ":" in name:
                names.add(name.split(":", 1)[0])
    return names


def _ollama_model_present(model: str, available: set[str]) -> bool:
    if model in available:
        return True
    base = model.split(":", 1)[0]
    tag = model.split(":", 1)[1] if ":" in model else "latest"
    return model in available or f"{base}:{tag}" in available or (
        tag == "latest" and base in available
    )


def ensure_ollama_model(model: str) -> None:
    """Pull the model if it is not already present locally."""
    import subprocess

    wait_for_ollama_server()
    available = list_ollama_models()
    if _ollama_model_present(model, available):
        print(f"Ollama model ready: {model}", flush=True)
        return

    print(f"Pulling Ollama model {model} (first run may take a while)...", flush=True)
    result = subprocess.run(
        ["ollama", "pull", model],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama pull {model!r} failed (rc={result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    print(f"Pulled {model}", flush=True)


def bench_ollama_chat(model: str, prompt_text: str, max_tokens: int = MAX_TOKENS) -> RunMetrics:
    """Stream /api/chat; wall-clock TTFT + decode using final eval_count."""
    payload = json.dumps({
        "model": model,
        "messages": chat_messages(prompt_text),
        "stream": True,
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0, "seed": 42},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    eval_count = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                if not raw.strip():
                    continue
                chunk = json.loads(raw)
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama chat error for {model!r}: {chunk['error']}")
                msg = chunk.get("message") or {}
                content = msg.get("content") or msg.get("thinking") or ""
                if content and t_first is None:
                    t_first = time.perf_counter()
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama /api/chat HTTP {exc.code} for model {model!r}: {body}"
        ) from exc

    t_end = time.perf_counter()
    if t_first is None:
        return RunMetrics(0.0, 0.0, eval_count, (t_end - t0) * 1000.0, output_token_ids=())

    ttft_ms = (t_first - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    tokens = eval_count if eval_count > 0 else 1
    decode_tps = decode_tps_from_wallclock(tokens, ttft_ms, total_ms)
    return RunMetrics(ttft_ms, decode_tps, tokens, total_ms, output_token_ids=())


def ollama_unload(model: str) -> None:
    payload = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception:
        pass


# ── JSON output ────────────────────────────────────────────────────────────────

REQUIRED_RESULT_KEYS = {
    "ts", "host", "method", "model_label", "model_ref", "sampling",
    "warmups", "runs_per_prompt", "prompt_set", "results",
}
REQUIRED_PROMPT_RESULT_KEYS = {
    "prompt",
    "ttft_ms_runs",
    "decode_tps_runs",
    "completion_tokens_runs",
    "ttft_ms_median",
    "decode_tps_median",
    "output_token_ids_sample_runs",
}


def validate_results_payload(payload: dict) -> None:
    missing = REQUIRED_RESULT_KEYS - payload.keys()
    if missing:
        raise ValueError(f"results payload missing keys: {sorted(missing)}")
    for entry in payload["results"]:
        missing_entry = REQUIRED_PROMPT_RESULT_KEYS - entry.keys()
        if missing_entry:
            raise ValueError(f"prompt result missing keys: {sorted(missing_entry)}")


def write_results(payload: dict) -> Path:
    validate_results_payload(payload)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    method = payload["method"]
    model_ref = payload["model_ref"]
    model_slug = model_ref.replace("/", "-").replace(":", "-")

    ts = payload["ts"]
    filename = f"{method}_{model_slug}_{ts}.json"
    path = results_dir / filename

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def get_timestamp_iso8601() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def check_output_token_parity(
    reference: tuple[int, ...],
    candidate: tuple[int, ...],
    *,
    method: str,
    prompt: str,
) -> dict[str, Any]:
    """Compare first-token samples; greedy backends should match when quant-compatible."""
    if not reference or not candidate:
        return {"parity_checked": False, "parity_match": None, "parity_method": method}
    n = min(len(reference), len(candidate))
    match = reference[:n] == candidate[:n]
    if not match:
        print(
            f"[parity] {method} [{prompt}] first {n} tokens differ from plain-mlx reference",
            flush=True,
        )
    return {
        "parity_checked": True,
        "parity_match": match,
        "parity_method": method,
        "parity_compared_tokens": n,
    }


def make_results_payload(
    *,
    ts: str,
    method: str,
    model_label: str,
    model_ref: str,
    results: list[dict[str, Any]],
    drafter_ref: Optional[str] = None,
    tree_budget: Optional[int] = None,
    sampling: Optional[dict[str, Any]] = None,
    warmups: int = WARMUPS,
    runs_per_prompt: int = RUNS_PER_PROMPT,
    prompt_set: str = PROMPT_SET,
    session_id: Optional[str] = None,
    family_id: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ts": ts,
        "session_id": session_id,
        "family_id": family_id,
        "host": HOST,
        "method": method,
        "model_label": model_label,
        "model_ref": model_ref,
        "drafter_ref": drafter_ref,
        "tree_budget": tree_budget,
        "sampling": sampling or dict(DEFAULT_SAMPLING),
        "warmups": warmups,
        "runs_per_prompt": runs_per_prompt,
        "prompt_set": prompt_set,
        "results": results,
    }
    if extra:
        payload.update(extra)
    return payload
