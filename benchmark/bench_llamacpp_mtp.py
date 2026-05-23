#!/usr/bin/env python3
"""
Benchmark: Qwen3.6-27B Q4_K_XL (llama.cpp) — baseline vs MTP speculative decoding.

Uses CODING_PROMPTS (algo / async / cache) for coding-task throughput measurement.
The MTP model bundles prediction heads in one GGUF; llama.cpp verifies draft tokens
from those heads in parallel (--spec-type draft-mtp).

Optimal MTP config per Unsloth: --spec-draft-n-max 2 (83% acceptance at temperature).
At temperature=0 (greedy) acceptance is higher, so n=2 is a conservative lower bound.

Models (auto-downloaded on first run via -hf flag):
    Baseline:  unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL       (~15 GB, 4-bit)
    MTP:       unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL    (~15 GB + bundled MTP heads)

Install llama.cpp on macOS (brew bottle ships with Metal + MTP since b9180):
    brew install llama.cpp     # or: brew upgrade llama.cpp
Or build from source for the latest master:
    git clone https://github.com/ggml-org/llama.cpp
    cmake llama.cpp -B llama.cpp/build
    cmake --build llama.cpp/build --config Release -j --target llama-server

Requires llama.cpp >= b9180 (MTP merged 2026-05-16, PR #22673).

Env vars:
    LLAMACPP_SERVER   path to llama-server binary   [/opt/homebrew/bin/llama-server]
    LLAMA_CACHE       HuggingFace model cache dir   [~/Models/llamacpp]
    LLAMACPP_ONLY     "baseline" or "mtp" to bench one config (cuts wall time)

Usage:
    LLAMACPP_SERVER=~/llama.cpp/build/bin/llama-server \
    python -m benchmark.bench_llamacpp_mtp
"""
import json, os, statistics, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

from benchmark._lib import (
    CODING_PROMPTS, HOST, cooldown,
    PRE_BENCH_COOLDOWN_S, INTER_PROMPT_COOLDOWN_S, INTER_CONFIG_COOLDOWN_S,
    POST_BENCH_COOLDOWN_S, write_results, get_timestamp_iso8601,
)

WARMUPS = 2
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200
MTP_DRAFT_N_MAX = 2
SERVER_PORT = 8989

LLAMACPP_SERVER = Path(os.path.expanduser(
    os.environ.get("LLAMACPP_SERVER", "/opt/homebrew/bin/llama-server")
))
LLAMA_CACHE = os.path.expanduser(
    os.environ.get("LLAMA_CACHE", "~/Models/llamacpp")
)

if not LLAMACPP_SERVER.exists():
    print(f"ERROR: llama-server not found at {LLAMACPP_SERVER}", file=sys.stderr)
    print("Install: brew install llama.cpp  (needs >= b9180 for MTP)", file=sys.stderr)
    sys.exit(1)

ALL_CONFIGS = [
    {
        "label":    "Qwen3.6-27B-dense [GGUF-Q4_K_XL]",
        "model_hf": "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
        "method":   "llamacpp",
        "mtp":      False,
        "drafter":  None,
    },
    {
        "label":    "Qwen3.6-27B-MTP   [GGUF-Q4_K_XL]",
        "model_hf": "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL",
        "method":   "llamacpp-mtp",
        "mtp":      True,
        "drafter":  "unsloth/Qwen3.6-27B-MTP-GGUF (bundled heads)",
    },
]

_only = os.environ.get("LLAMACPP_ONLY", "").lower()
if _only:
    CONFIGS = [c for c in ALL_CONFIGS if _only in c["label"].lower() or _only in c["method"]]
    assert CONFIGS, f"LLAMACPP_ONLY={_only!r} matched nothing"
else:
    CONFIGS = ALL_CONFIGS


# ── Server management ──────────────────────────────────────────────────────────

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
    env = {**os.environ, "LLAMA_CACHE": LLAMA_CACHE}
    return subprocess.Popen(cmd, env=env)


def wait_for_server(proc: subprocess.Popen, timeout_s: int = 2400) -> None:
    """Poll /health until the server responds or timeout. Fast-fail if proc dies."""
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


# ── Single-inference bench ─────────────────────────────────────────────────────

def bench_one(prompt: str) -> tuple[float, float, float | None]:
    """Stream one chat completion, return (ttft_ms, decode_tps, accept_ratio).

    TTFT  = wall time to first content token.
    Decode = (completion_tokens - 1) / elapsed_after_first_token.
    accept_ratio = draft_n_accepted / draft_n from the final chunk's `timings`
                   field; None when the server emits no draft tokens (baseline).
    Uses stream_options.include_usage for exact token count; falls back to
    content-chunk counting if the server version omits the usage chunk.
    """
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    ttft_ms = 0.0
    n_chunks = 0
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
            # Cumulative spec-decoding timings ride along the SSE deltas; the
            # final chunk carries the totals (server-task.cpp:927). Track the
            # latest values we see so we end with the full-response numbers.
            timings = chunk.get("timings")
            if timings and "draft_n" in timings:
                draft_n = timings.get("draft_n", 0)
                draft_n_accepted = timings.get("draft_n_accepted", 0)
            # usage chunk (stream_options.include_usage=True)
            if "usage" in chunk and not chunk.get("choices"):
                completion_tokens = chunk["usage"].get("completion_tokens", 0)
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            # Qwen3.6 is a thinking model: generated tokens appear in
            # `reasoning_content` (thinking phase) before any `content` is
            # emitted.  At max_tokens=200 the response may consist entirely of
            # thinking tokens, so we must treat either field as a real token
            # for TTFT and decode-rate measurement.
            token_text = delta.get("content") or delta.get("reasoning_content") or ""
            if token_text:
                if t_first is None:
                    t_first = time.perf_counter()
                    ttft_ms = (t_first - t0) * 1000.0
                n_chunks += 1

    accept_ratio = (draft_n_accepted / draft_n) if draft_n > 0 else None

    if t_first is None:
        return 0.0, 0.0, accept_ratio

    n_decode = completion_tokens if completion_tokens > 1 else n_chunks
    elapsed = time.perf_counter() - t_first
    decode_tps = (n_decode - 1) / elapsed if elapsed > 0 and n_decode > 1 else 0.0
    return ttft_ms, decode_tps, accept_ratio


# ── Main ───────────────────────────────────────────────────────────────────────

all_results: dict[str, dict] = {}

cooldown(PRE_BENCH_COOLDOWN_S, "pre-bench: let chip cool before llama.cpp phase")

for cfg_idx, cfg in enumerate(CONFIGS):
    label = cfg["label"]
    if cfg_idx > 0:
        cooldown(INTER_CONFIG_COOLDOWN_S, f"between configs: thermal reset + server teardown")

    print(f"\n{'='*70}", flush=True)
    print(f"  Config: {label}  (MTP={'on, n=' + str(MTP_DRAFT_N_MAX) if cfg['mtp'] else 'off'})", flush=True)
    print(f"{'='*70}\n", flush=True)

    proc = start_server(cfg)
    results_per_prompt: dict[str, dict] = {}
    try:
        print("  Waiting for server (downloads + load can take 30+ min on first run)...", flush=True)
        wait_for_server(proc)
        print("  Server ready.\n", flush=True)

        for p_idx, (pname, ptext) in enumerate(CODING_PROMPTS):
            if p_idx > 0:
                cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts [{label}]")

            print(f"  Prompt: {pname}", flush=True)
            for w in range(1, WARMUPS + 1):
                print(f"  [warmup {w}/{WARMUPS}]...", end=" ", flush=True)
                bench_one(ptext)
                print("done", flush=True)

            ttft_runs, decode_runs, accept_runs = [], [], []
            for i in range(1, RUNS_PER_PROMPT + 1):
                print(f"  [run {i}/{RUNS_PER_PROMPT}] ", end="", flush=True)
                ttft_ms, decode_tps, accept = bench_one(ptext)
                ttft_runs.append(ttft_ms)
                decode_runs.append(decode_tps)
                if accept is not None:
                    accept_runs.append(accept)
                accept_str = f"  accept {accept:.0%}" if accept is not None else ""
                print(f"TTFT {ttft_ms:.1f}ms  decode {decode_tps:.1f} tok/s{accept_str}", flush=True)

            results_per_prompt[pname] = {
                "prompt": pname,
                "ttft_ms_runs": ttft_runs,
                "decode_tps_runs": decode_runs,
                "accept_ratio_runs": accept_runs,
                "avg_acceptance": statistics.median(accept_runs) if accept_runs else None,
                "ttft_ms_median": statistics.median(ttft_runs),
                "decode_tps_median": statistics.median(decode_runs),
            }

        all_results[label] = results_per_prompt

    finally:
        proc.terminate()
        proc.wait(timeout=30)
        print(f"\n  Stopped server for {label}.", flush=True)

cooldown(POST_BENCH_COOLDOWN_S, "post-bench: final cooldown")

# ── Write JSON outputs ─────────────────────────────────────────────────────────

ts = get_timestamp_iso8601()
for cfg in CONFIGS:
    label = cfg["label"]
    rpp = all_results[label]
    payload = {
        "ts": ts,
        "host": HOST,
        "method": cfg["method"],
        "model_label": label,
        "model_ref": cfg["model_hf"],
        "drafter_ref": cfg["drafter"],
        "tree_budget": MTP_DRAFT_N_MAX if cfg["mtp"] else None,
        "sampling": {"temperature": 0, "seed": None, "max_tokens": MAX_TOKENS},
        "warmups": WARMUPS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "results": [rpp[pname] for pname, _ in CODING_PROMPTS],
    }
    path = write_results(payload)
    print(f"Wrote {label} → {path}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print(f"  llama.cpp MTP SUMMARY — coding tasks")
print(f"  {WARMUPS} warmups · {RUNS_PER_PROMPT} runs/prompt · median · {MAX_TOKENS} max tok")
print(f"{'='*70}")
for cfg in CONFIGS:
    label = cfg["label"]
    rpp = all_results[label]
    print(f"\n  {label}")
    print(f"  {'Prompt':<14}  {'TTFT (ms)':>10}  {'Decode tok/s':>13}  {'Accept':>7}")
    for pname, _ in CODING_PROMPTS:
        t = rpp[pname]["ttft_ms_median"]
        d = rpp[pname]["decode_tps_median"]
        a = rpp[pname]["avg_acceptance"]
        a_str = f"{a*100:.0f}%" if a is not None else "—"
        print(f"  {pname:<14}  {t:>10.1f}  {d:>13.1f}  {a_str:>7}")
print()
