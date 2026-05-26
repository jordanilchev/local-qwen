#!/usr/bin/env python3
"""
Benchmark: Qwen3.6-27B Q4_K_XL (llama.cpp) — baseline vs MTP speculative decoding.

Usage:
    LLAMACPP_SERVER=~/llama.cpp/build/bin/llama-server python -m benchmark.bench_llamacpp_mtp
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from benchmark._lib import (
    CODING_PROMPTS,
    DEFAULT_SAMPLING,
    INTER_CONFIG_COOLDOWN_S,
    INTER_PROMPT_COOLDOWN_S,
    POST_BENCH_COOLDOWN_S,
    PRE_BENCH_COOLDOWN_S,
    PROMPT_SET,
    RUNS_PER_PROMPT,
    RunMetrics,
    WARMUP_PROMPTS,
    WARMUPS,
    chat_messages,
    cooldown,
    decode_tps_from_wallclock,
    get_timestamp_iso8601,
    make_results_payload,
    run_prompt_suite,
    write_results,
)

MAX_TOKENS = DEFAULT_SAMPLING["max_tokens"]
MTP_DRAFT_N_MAX = 2
SERVER_PORT = 8989

LLAMACPP_SERVER = Path(os.path.expanduser(
    os.environ.get("LLAMACPP_SERVER", "/opt/homebrew/bin/llama-server")
))
LLAMA_CACHE = os.path.expanduser(os.environ.get("LLAMA_CACHE", "~/Models/llamacpp"))

if not LLAMACPP_SERVER.exists():
    print(f"ERROR: llama-server not found at {LLAMACPP_SERVER}", file=sys.stderr)
    sys.exit(1)

ALL_CONFIGS = [
    {
        "label": "Qwen3.6-27B-dense [GGUF-Q4_K_XL]",
        "model_hf": "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
        "method": "llamacpp",
        "mtp": False,
        "drafter": None,
    },
    {
        "label": "Qwen3.6-27B-MTP   [GGUF-Q4_K_XL]",
        "model_hf": "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL",
        "method": "llamacpp-mtp",
        "mtp": True,
        "drafter": "unsloth/Qwen3.6-27B-MTP-GGUF (bundled heads)",
    },
]

_only = os.environ.get("LLAMACPP_ONLY", "").lower()
CONFIGS = (
    [c for c in ALL_CONFIGS if _only in c["label"].lower() or _only in c["method"]]
    if _only
    else ALL_CONFIGS
)
if _only:
    assert CONFIGS, f"LLAMACPP_ONLY={_only!r} matched nothing"


def start_server(cfg: dict) -> subprocess.Popen:
    cmd = [
        str(LLAMACPP_SERVER),
        "-hf", cfg["model_hf"],
        "--host", "127.0.0.1", "--port", str(SERVER_PORT),
        "--ctx-size", "8192",
        "-ngl", "99",
        "-fa", "on",
        "-np", "1",
    ]
    if cfg["mtp"]:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_DRAFT_N_MAX)]
    return subprocess.Popen(cmd, env={**os.environ, "LLAMA_CACHE": LLAMA_CACHE})


def wait_for_server(proc: subprocess.Popen, timeout_s: int = 2400) -> None:
    url = f"http://127.0.0.1:{SERVER_PORT}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"llama-server did not start within {timeout_s}s")


def bench_one(prompt: str) -> tuple[RunMetrics, float | None]:
    payload = json.dumps({
        "messages": chat_messages(prompt),
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": DEFAULT_SAMPLING["seed"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    completion_tokens = 0
    draft_n = 0
    draft_n_accepted = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            timings = chunk.get("timings")
            if timings and "draft_n" in timings:
                draft_n = timings.get("draft_n", 0)
                draft_n_accepted = timings.get("draft_n_accepted", 0)
            if "usage" in chunk and not chunk.get("choices"):
                completion_tokens = chunk["usage"].get("completion_tokens", 0)
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            token_text = delta.get("content") or delta.get("reasoning_content") or ""
            if token_text and t_first is None:
                t_first = time.perf_counter()

    t_end = time.perf_counter()
    accept_ratio = (draft_n_accepted / draft_n) if draft_n > 0 else None

    if t_first is None:
        total_ms = (t_end - t0) * 1000.0
        return RunMetrics(0.0, 0.0, completion_tokens, total_ms), accept_ratio

    ttft_ms = (t_first - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    tokens = completion_tokens if completion_tokens > 0 else 1
    decode_tps = decode_tps_from_wallclock(tokens, ttft_ms, total_ms)
    return RunMetrics(ttft_ms, decode_tps, tokens, total_ms), accept_ratio


def run_llamacpp_suite(label: str, prompt_text: str):
    accept_runs: list[float] = []

    def timed():
        metrics, accept = bench_one(prompt_text)
        if accept is not None:
            accept_runs.append(accept)
        return metrics

    suite = run_prompt_suite(
        label,
        bench_fn=timed,
        warmup_fn=lambda wp: bench_one(wp)[0],
        warmup_prompts=WARMUP_PROMPTS,
    )
    suite["avg_acceptance"] = statistics.median(accept_runs) if accept_runs else None
    if accept_runs:
        suite["accept_ratio_runs"] = accept_runs
        suite["acceptance_unit"] = "draft_ratio"
    return suite


all_results: dict[str, dict] = {}

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before llama.cpp phase")

for cfg_idx, cfg in enumerate(CONFIGS):
    if cfg_idx > 0:
        cooldown(INTER_CONFIG_COOLDOWN_S, "between configs: thermal reset + server teardown")

    label = cfg["label"]
    print(f"\n{'='*70}", flush=True)
    print(f"  Config: {label}  (MTP={'on, n=' + str(MTP_DRAFT_N_MAX) if cfg['mtp'] else 'off'})", flush=True)
    print(f"{'='*70}\n", flush=True)

    proc = start_server(cfg)
    results_per_prompt: dict[str, dict] = {}
    try:
        print("  Waiting for server...", flush=True)
        wait_for_server(proc)
        print("  Server ready.\n", flush=True)

        for p_idx, (pname, ptext) in enumerate(CODING_PROMPTS):
            if p_idx > 0:
                cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts [{label}]")
            results_per_prompt[pname] = run_llamacpp_suite(f"{label} [{pname}]", ptext)

        all_results[label] = results_per_prompt
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        print(f"\n  Stopped server for {label}.", flush=True)

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

ts = get_timestamp_iso8601()
for cfg in CONFIGS:
    label = cfg["label"]
    rpp = all_results[label]
    payload = make_results_payload(
        ts=ts,
        method=cfg["method"],
        model_label=label,
        model_ref=cfg["model_hf"],
        drafter_ref=cfg["drafter"],
        tree_budget=MTP_DRAFT_N_MAX if cfg["mtp"] else None,
        results=[{"prompt": pname, **rpp[pname]} for pname, _ in CODING_PROMPTS],
        sampling=dict(DEFAULT_SAMPLING),
        warmups=WARMUPS,
        runs_per_prompt=RUNS_PER_PROMPT,
        prompt_set=PROMPT_SET,
    )
    path = write_results(payload)
    print(f"Wrote {label} → {path}", flush=True)

print(f"\n\n{'='*70}")
print(f"  llama.cpp MTP SUMMARY — {PROMPT_SET} prompts")
print(f"{'='*70}")
for cfg in CONFIGS:
    label = cfg["label"]
    rpp = all_results[label]
    print(f"\n  {label}")
    print(f"  {'Prompt':<14}  {'TTFT (ms)':>10}  {'Decode tok/s':>13}  {'Accept':>7}")
    for pname, _ in CODING_PROMPTS:
        t = rpp[pname]
        a = t.get("avg_acceptance")
        a_str = f"{a * 100:.0f}%" if a is not None else "—"
        print(
            f"  {pname:<14}  {t['ttft_ms_median']:>10.1f}"
            f"  {t['decode_tps_median']:>13.1f}  {a_str:>7}"
        )
print()
