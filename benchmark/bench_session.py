#!/usr/bin/env python3
"""
Unified single-session benchmark: all backends for one model family, MLX first.

Every method in a session shares the same `session_id` and `ts` so published tables
never mix results from different thermal windows.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_session
    BENCH_FAMILY=3.8-27b HF_HOME=~/Models/HuggingFace .venv/bin/python -m benchmark.bench_session
    BENCH_METHODS=dflash2-mlx,ollama  # optional subset of DEFAULT_SESSION_METHODS
    BENCH_SESSION_ID=20260818T120000Z  # optional fixed session id
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark._lib import (
    CODING_PROMPTS,
    DEFAULT_SAMPLING,
    DEFAULT_TREE_BUDGET,
    INTER_CONFIG_COOLDOWN_S,
    INTER_PROMPT_COOLDOWN_S,
    POST_BENCH_COOLDOWN_S,
    PRE_BENCH_COOLDOWN_S,
    PROMPT_SET,
    RUNS_PER_PROMPT,
    WARMUPS,
    bench_ddtree,
    bench_dflash2_mlx,
    bench_mlx_plain,
    bench_ollama_chat,
    bench_vllm_engine,
    check_output_token_parity,
    cooldown,
    wait_for_thermal_idle,
    ensure_ollama_model,
    get_timestamp_iso8601,
    load_ddtree_runtime,
    load_mlx_target,
    make_results_payload,
    ollama_unload,
    resolve_draft_ref,
    run_prompt_suite,
    wait_for_ollama_server,
    write_results,
)
from benchmark.dflash2 import (
    DEFAULT_DFLASH2_DRAFT_BITS,
    is_dflash2_ref,
    load_dflash2_mlx_runtime,
)
from benchmark.models import MODEL_FAMILIES, ModelFamily, family_label, get_family

DEFAULT_SESSION_METHODS = (
    "plain-mlx",
    "vllm-mlx",
    "dflash2-mlx",
    "ddtree-mlx",
    "ollama",
    "llamacpp",
)

BENCH_FAMILY = os.environ.get("BENCH_FAMILY", "all").lower()
SESSION_ID = os.environ.get("BENCH_SESSION_ID") or get_timestamp_iso8601()
VLLM_GPU_MEM = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95"))
LLAMACPP_SERVER = Path(os.path.expanduser(
    os.environ.get("LLAMACPP_SERVER", "/opt/homebrew/bin/llama-server")
))
LLAMA_CACHE = os.path.expanduser(os.environ.get("LLAMA_CACHE", "~/Models/llamacpp"))
LLAMACPP_PORT = 8989
MTP_DRAFT_N_MAX = 2

# Reference token samples from plain-mlx first timed run per prompt (parity checks).
_reference_tokens: dict[str, tuple[int, ...]] = {}


def _families_to_run() -> list[ModelFamily]:
    if BENCH_FAMILY == "all":
        return list(MODEL_FAMILIES)
    return [get_family(BENCH_FAMILY)]


def methods_to_run() -> list[str]:
    raw = os.environ.get("BENCH_METHODS", "").strip()
    if not raw:
        return list(DEFAULT_SESSION_METHODS)
    requested = [m.strip() for m in raw.split(",") if m.strip()]
    known = set(DEFAULT_SESSION_METHODS)
    unknown = [m for m in requested if m not in known]
    if unknown:
        raise ValueError(
            f"Unknown BENCH_METHODS {unknown}; known: {', '.join(DEFAULT_SESSION_METHODS)}"
        )
    return requested


def _write_method(
    *,
    ts: str,
    method: str,
    family: ModelFamily,
    model_ref: str,
    results_dict: dict[str, dict[str, Any]],
    drafter_ref: str | None = None,
    tree_budget: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    label = family_label(family)
    payload = make_results_payload(
        ts=ts,
        session_id=SESSION_ID,
        family_id=family.id,
        method=method,
        model_label=label,
        model_ref=model_ref,
        drafter_ref=drafter_ref,
        tree_budget=tree_budget,
        results=[
            {"prompt": prompt_name, **results_dict[prompt_name]}
            for prompt_name, _ in CODING_PROMPTS
        ],
        sampling=dict(DEFAULT_SAMPLING),
        warmups=WARMUPS,
        runs_per_prompt=RUNS_PER_PROMPT,
        prompt_set=PROMPT_SET,
        extra=extra,
    )
    return write_results(payload)


def _run_prompt_suites_for_method(
    method_label: str,
    bench_fn_factory: Callable[[str], Callable[[], Any]],
    warmup_fn_factory: Callable[[], Callable[[str], None]],
    *,
    is_reference_method: bool = False,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for prompt_name, prompt_text in CODING_PROMPTS:
        suite = run_prompt_suite(
            f"{method_label} [{prompt_name}]",
            bench_fn=bench_fn_factory(prompt_text),
            warmup_fn=warmup_fn_factory(),
        )

        if suite.get("output_token_ids_sample_runs"):
            sample = tuple(suite["output_token_ids_sample_runs"][0])
            if is_reference_method:
                _reference_tokens[prompt_name] = sample
            elif prompt_name in _reference_tokens:
                suite.update(
                    check_output_token_parity(
                        _reference_tokens[prompt_name],
                        sample,
                        method=method_label,
                        prompt=prompt_name,
                    )
                )

        suite.setdefault("avg_acceptance", None)
        results[prompt_name] = suite
        cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {method_label}")
    return results


def _run_plain_mlx(family: ModelFamily, ts: str) -> None:
    if not family.mlx_ref:
        return
    print(f"\n{'='*70}\n  plain-mlx · {family_label(family)}\n{'='*70}", flush=True)
    model, tokenizer = load_mlx_target(family.mlx_ref)
    try:
        results = _run_prompt_suites_for_method(
            f"plain-mlx · {family_label(family)}",
            lambda pt: lambda: bench_mlx_plain(model, tokenizer, pt),
            lambda: (lambda wp: bench_mlx_plain(model, tokenizer, wp)),
            is_reference_method=True,
        )
        path = _write_method(
            ts=ts, method="plain-mlx", family=family, model_ref=family.mlx_ref, results_dict=results
        )
        print(f"Wrote plain-mlx → {path}", flush=True)
    finally:
        del model, tokenizer
        gc.collect()


def _run_vllm_mlx(family: ModelFamily, ts: str) -> None:
    if not family.mlx_ref:
        return
    from vllm_mlx import EngineCore, EngineConfig
    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    print(f"\n{'='*70}\n  vllm-mlx · {family_label(family)}\n{'='*70}", flush=True)
    model, tokenizer = load_model_with_fallback(family.mlx_ref)
    engine = EngineCore(
        model,
        tokenizer,
        EngineConfig(model_name=family.mlx_ref, gpu_memory_utilization=VLLM_GPU_MEM),
    )
    try:
        results = _run_prompt_suites_for_method(
            f"vllm-mlx · {family_label(family)}",
            lambda pt: lambda: bench_vllm_engine(engine, tokenizer, pt),
            lambda: (lambda wp: bench_vllm_engine(engine, tokenizer, wp)),
        )
        path = _write_method(
            ts=ts,
            method="vllm-mlx",
            family=family,
            model_ref=family.mlx_ref,
            results_dict=results,
            extra={"vllm_gpu_memory_utilization": VLLM_GPU_MEM},
        )
        print(f"Wrote vllm-mlx → {path}", flush=True)
    finally:
        del engine, model, tokenizer
        gc.collect()


def _run_dflash2_mlx(family: ModelFamily, ts: str) -> None:
    draft_ref = family.draft_ref or resolve_draft_ref(family.mlx_ref or "")
    if not family.mlx_ref or not draft_ref or not is_dflash2_ref(draft_ref):
        print(f"Skipping dflash2-mlx for {family.id} (no DFlash2 drafter)", flush=True)
        return

    print(f"\n{'='*70}\n  dflash2-mlx · {family_label(family)}\n{'='*70}", flush=True)
    try:
        model, tokenizer, draft, block_size = load_dflash2_mlx_runtime(
            family.mlx_ref, draft_ref
        )
    except Exception as exc:
        print(f"Skipping dflash2-mlx for {family.id}: {exc}", flush=True)
        return

    try:
        results: dict[str, dict[str, Any]] = {}
        for prompt_name, prompt_text in CODING_PROMPTS:
            accept_runs: list[float] = []

            def _timed(pt=prompt_text):
                metrics, acc = bench_dflash2_mlx(
                    model, tokenizer, draft, pt, block_size=block_size
                )
                if acc == acc:
                    accept_runs.append(acc)
                return metrics

            suite = run_prompt_suite(
                f"dflash2-mlx · {family_label(family)} [{prompt_name}]",
                bench_fn=_timed,
                warmup_fn=lambda wp: bench_dflash2_mlx(
                    model, tokenizer, draft, wp, block_size=block_size
                )[0],
            )
            if suite.get("output_token_ids_sample_runs") and prompt_name in _reference_tokens:
                sample = tuple(suite["output_token_ids_sample_runs"][0])
                suite.update(
                    check_output_token_parity(
                        _reference_tokens[prompt_name],
                        sample,
                        method="dflash2-mlx",
                        prompt=prompt_name,
                    )
                )
            suite["avg_acceptance"] = (
                statistics.mean(accept_runs) if accept_runs else None
            )
            results[prompt_name] = suite
            cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of dflash2-mlx")
        path = _write_method(
            ts=ts,
            method="dflash2-mlx",
            family=family,
            model_ref=family.mlx_ref,
            results_dict=results,
            drafter_ref=draft_ref,
            extra={
                "block_size": block_size,
                "draft_bits": DEFAULT_DFLASH2_DRAFT_BITS,
                "spec_path": "dflash.model_mlx.stream_generate",
            },
        )
        print(f"Wrote dflash2-mlx → {path}", flush=True)
    finally:
        del model, tokenizer, draft
        gc.collect()


def _run_ddtree(family: ModelFamily, ts: str) -> None:
    draft_ref = family.draft_ref or resolve_draft_ref(family.mlx_ref or "")
    if not family.mlx_ref or not draft_ref:
        print(f"Skipping DDTree for {family.id} (no drafter)", flush=True)
        return

    print(f"\n{'='*70}\n  ddtree-mlx · {family_label(family)}\n{'='*70}", flush=True)
    try:
        target, tokenizer, draft, loaded_draft, stop = load_ddtree_runtime(family.mlx_ref, draft_ref)
    except Exception as exc:
        print(f"Skipping DDTree for {family.id}: {exc}", flush=True)
        return

    try:
        results: dict[str, dict[str, Any]] = {}
        for prompt_name, prompt_text in CODING_PROMPTS:
            accept_runs: list[float] = []

            def _timed(pt=prompt_text):
                metrics, acc = bench_ddtree(
                    target, tokenizer, draft, stop, pt, tree_budget=DEFAULT_TREE_BUDGET
                )
                if acc == acc:
                    accept_runs.append(acc)
                return metrics

            suite = run_prompt_suite(
                f"ddtree-mlx · {family_label(family)} [{prompt_name}]",
                bench_fn=_timed,
                warmup_fn=lambda wp: bench_ddtree(
                    target, tokenizer, draft, stop, wp, tree_budget=DEFAULT_TREE_BUDGET
                )[0],
            )
            if suite.get("output_token_ids_sample_runs") and prompt_name in _reference_tokens:
                sample = tuple(suite["output_token_ids_sample_runs"][0])
                suite.update(
                    check_output_token_parity(
                        _reference_tokens[prompt_name],
                        sample,
                        method="ddtree-mlx",
                        prompt=prompt_name,
                    )
                )
            suite["avg_acceptance"] = (
                statistics.mean(accept_runs) if accept_runs else None
            )
            results[prompt_name] = suite
            cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of ddtree-mlx")
        path = _write_method(
            ts=ts,
            method="ddtree-mlx",
            family=family,
            model_ref=family.mlx_ref,
            results_dict=results,
            drafter_ref=loaded_draft,
            tree_budget=DEFAULT_TREE_BUDGET,
        )
        print(f"Wrote ddtree-mlx → {path}", flush=True)
    finally:
        del target, tokenizer, draft, stop
        gc.collect()


def _ensure_ollama_server() -> None:
    try:
        wait_for_ollama_server(timeout_s=5)
        return
    except RuntimeError:
        print("Starting ollama serve...", flush=True)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_for_ollama_server(timeout_s=30)


def _run_ollama(family: ModelFamily, ts: str) -> None:
    if not family.ollama_ref:
        return
    _ensure_ollama_server()
    print(f"\n{'='*70}\n  ollama · {family_label(family)} ({family.ollama_ref})\n{'='*70}", flush=True)
    ensure_ollama_model(family.ollama_ref)
    results = _run_prompt_suites_for_method(
        f"ollama · {family_label(family)}",
        lambda pt: lambda: bench_ollama_chat(family.ollama_ref, pt),
        lambda: (lambda wp: bench_ollama_chat(family.ollama_ref, wp)),
    )
    path = _write_method(
        ts=ts,
        method="ollama",
        family=family,
        model_ref=family.ollama_ref,
        results_dict=results,
        extra={"ollama_quant": family.ollama_quant, "includes_http_overhead": True},
    )
    print(f"Wrote ollama → {path}", flush=True)
    print(f"Unloading {family.ollama_ref}...", flush=True)
    ollama_unload(family.ollama_ref)


def _bench_llamacpp_one(prompt: str, port: int) -> tuple[Any, float | None]:
    from benchmark._lib import RunMetrics, chat_messages, decode_tps_from_wallclock

    payload = json.dumps({
        "messages": chat_messages(prompt),
        "max_tokens": DEFAULT_SAMPLING["max_tokens"],
        "temperature": 0.0,
        "seed": DEFAULT_SAMPLING["seed"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
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
        return RunMetrics(0.0, 0.0, completion_tokens, (t_end - t0) * 1000.0), accept_ratio

    ttft_ms = (t_first - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    tokens = completion_tokens if completion_tokens > 0 else 1
    decode_tps = decode_tps_from_wallclock(tokens, ttft_ms, total_ms)
    return RunMetrics(ttft_ms, decode_tps, tokens, total_ms), accept_ratio


def _run_llamacpp_config(
    family: ModelFamily,
    ts: str,
    *,
    model_hf: str,
    method: str,
    mtp: bool,
    drafter: str | None,
) -> None:
    if not LLAMACPP_SERVER.exists():
        print(f"Skipping {method}: llama-server not found at {LLAMACPP_SERVER}", flush=True)
        return

    label = f"{family.label} [{family.gguf_quant}]"
    local_model = os.environ.get("LLAMACPP_MODEL", "").strip()
    cmd = [
        str(LLAMACPP_SERVER),
        "--host", "127.0.0.1", "--port", str(LLAMACPP_PORT),
        "--ctx-size", "8192",
        "-ngl", "99",
        "-fa", "on",
        "-np", "1",
    ]
    if local_model:
        cmd += ["--model", local_model]
        mmproj = os.environ.get("LLAMACPP_MMPROJ", "").strip()
        if mmproj:
            cmd += ["--mmproj", mmproj]
        print(f"Using local GGUF: {local_model}", flush=True)
    else:
        cmd[1:1] = ["-hf", model_hf]
    if mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_DRAFT_N_MAX)]

    print(f"\n{'='*70}\n  {method} · {label}\n{'='*70}", flush=True)
    proc = subprocess.Popen(cmd, env={**os.environ, "LLAMA_CACHE": LLAMA_CACHE})
    results: dict[str, dict[str, Any]] = {}
    try:
        deadline = time.time() + 2400
        health = f"http://127.0.0.1:{LLAMACPP_PORT}/health"
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited rc={proc.returncode}")
            try:
                with urllib.request.urlopen(health, timeout=2):
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("llama-server startup timeout")

        for prompt_name, prompt_text in CODING_PROMPTS:
            accept_runs: list[float] = []

            def _timed(pt=prompt_text):
                metrics, acc = _bench_llamacpp_one(pt, LLAMACPP_PORT)
                if acc is not None:
                    accept_runs.append(acc)
                return metrics

            suite = run_prompt_suite(
                f"{method} · {label} [{prompt_name}]",
                bench_fn=_timed,
                warmup_fn=lambda wp: _bench_llamacpp_one(wp, LLAMACPP_PORT)[0],
            )
            suite["avg_acceptance"] = statistics.median(accept_runs) if accept_runs else None
            if accept_runs:
                suite["accept_ratio_runs"] = accept_runs
                suite["acceptance_unit"] = "draft_ratio"
            results[prompt_name] = suite
            cooldown(INTER_PROMPT_COOLDOWN_S, f"between prompts of {method}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    path = _write_method(
        ts=ts,
        method=method,
        family=family,
        model_ref=model_hf,
        results_dict=results,
        drafter_ref=drafter,
        tree_budget=MTP_DRAFT_N_MAX if mtp else None,
        extra={"gguf_quant": family.gguf_quant, "includes_http_overhead": True},
    )
    print(f"Wrote {method} → {path}", flush=True)


def _run_llamacpp(family: ModelFamily, ts: str) -> None:
    if family.llamacpp_hf:
        cooldown(INTER_CONFIG_COOLDOWN_S, "before llama.cpp baseline")
        _run_llamacpp_config(
            family, ts, model_hf=family.llamacpp_hf, method="llamacpp", mtp=False, drafter=None
        )
    if family.llamacpp_mtp_hf:
        cooldown(INTER_CONFIG_COOLDOWN_S, "before llama.cpp MTP")
        _run_llamacpp_config(
            family,
            ts,
            model_hf=family.llamacpp_mtp_hf,
            method="llamacpp-mtp",
            mtp=True,
            drafter=f"{family.llamacpp_mtp_hf} (bundled MTP heads)",
        )


def run_family_session(family: ModelFamily) -> None:
    global _reference_tokens
    _reference_tokens = {}
    ts = get_timestamp_iso8601()

    print(f"\n{'#'*70}", flush=True)
    print(f"  SESSION {SESSION_ID} · family {family.id} · ts {ts}", flush=True)
    print(f"  Order: {' → '.join(methods_to_run())}", flush=True)
    if family.notes:
        print(f"  Note: {family.notes}", flush=True)
    print(f"{'#'*70}", flush=True)

    cooldown(PRE_BENCH_COOLDOWN_S, f"pre-bench: {family.id}")
    wait_for_thermal_idle(f"family {family.id} start")

    step_fns: dict[str, Callable[[], None]] = {
        "plain-mlx": lambda: _run_plain_mlx(family, ts),
        "vllm-mlx": lambda: _run_vllm_mlx(family, ts),
        "dflash2-mlx": lambda: _run_dflash2_mlx(family, ts),
        "ddtree-mlx": lambda: _run_ddtree(family, ts),
        "ollama": lambda: _run_ollama(family, ts),
        "llamacpp": lambda: _run_llamacpp(family, ts),
    }
    for step_name in methods_to_run():
        try:
            step_fns[step_name]()
        except Exception as exc:
            print(f"ERROR {step_name} {family.id}: {exc}", flush=True)
        wait_for_thermal_idle(f"after {step_name}")


def main() -> None:
    families = _families_to_run()
    print(f"Benchmark session_id={SESSION_ID} families={[f.id for f in families]}", flush=True)
    for idx, family in enumerate(families):
        if idx > 0:
            cooldown(INTER_CONFIG_COOLDOWN_S, f"between families ({family.id})")
        run_family_session(family)

    summary_path = Path(__file__).parent / "results" / f"session_{SESSION_ID}.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "session_id": SESSION_ID,
                "families": [fam.id for fam in families],
                "host": "M4-MBA-32GB",
                "prompt_set": PROMPT_SET,
            },
            f,
            indent=2,
        )
    print(f"\nSession complete. Marker → {summary_path}", flush=True)
    print(f"Summarize: .venv/bin/python -m benchmark.summarize --session-id {SESSION_ID}", flush=True)


if __name__ == "__main__":
    main()
