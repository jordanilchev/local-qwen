#!/usr/bin/env python3
"""
Legacy Ollama probe — prefer benchmark/bench_compare.py for published results.

Uses the same chat API and timing helpers as the main benchmark suite.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from benchmark._lib import (
    DEFAULT_SAMPLING,
    MAX_TOKENS,
    PROMPT_CODE,
    WARMUP_PROMPTS,
    bench_ollama_chat,
    chat_messages,
    cooldown,
    INTER_RUN_COOLDOWN_S,
)

RUNS = 3


def get_ollama_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return [line.split()[0] for line in result.stdout.strip().split("\n")[1:] if line.strip()]


class ResourceMonitor:
    def __init__(self):
        self.samples_cpu: list[float] = []
        self.samples_mem_gb: list[float] = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples_cpu = []
        self.samples_mem_gb = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        if not HAS_PSUTIL:
            return
        procs = [
            p for p in psutil.process_iter(["pid", "name", "cmdline"])
            if "ollama" in (p.info["name"] or "").lower()
            or "ollama" in " ".join(p.info["cmdline"] or []).lower()
        ]
        for p in procs:
            try:
                p.cpu_percent(interval=None)
            except Exception:
                pass
        time.sleep(0.15)
        while not self._stop.is_set():
            cpu_total = mem_total = 0
            for p in procs:
                try:
                    cpu_total += p.cpu_percent(interval=None)
                    mem_total += p.memory_info().rss
                except Exception:
                    pass
            self.samples_cpu.append(cpu_total)
            self.samples_mem_gb.append(mem_total / (1024 ** 3))
            time.sleep(0.25)

    def stats(self):
        if not self.samples_cpu:
            return 0.0, 0.0, 0.0
        return (
            sum(self.samples_cpu) / len(self.samples_cpu),
            max(self.samples_cpu),
            max(self.samples_mem_gb) if self.samples_mem_gb else 0.0,
        )


MODELS = get_ollama_models()
if not MODELS:
    print("No models found — is ollama running?")
    sys.exit(1)

print(f"Models: {', '.join(MODELS)}", flush=True)
print(f"Timed prompt: coding (red-black tree)", flush=True)
print(f"Runs per model: {RUNS}\n", flush=True)

monitor = ResourceMonitor()
summary = []

for model in MODELS:
    print(f"\n{'='*75}", flush=True)
    print(f"  {model}", flush=True)
    print(f"{'='*75}", flush=True)

    print("  [warmup]...", end=" ", flush=True)
    bench_ollama_chat(model, WARMUP_PROMPTS[0], max_tokens=64)
    print("done", flush=True)

    runs = []
    for i in range(1, RUNS + 1):
        if i > 1:
            cooldown(INTER_RUN_COOLDOWN_S, f"between runs ({model})")
        print(f"  [run {i}/{RUNS}]", end=" ", flush=True)
        monitor.start()
        m = bench_ollama_chat(model, PROMPT_CODE, max_tokens=MAX_TOKENS)
        monitor.stop()
        avg_cpu, peak_cpu, peak_mem = monitor.stats()
        runs.append((m.ttft_ms, m.decode_tps, m.completion_tokens, avg_cpu, peak_mem))
        print(
            f"TTFT {m.ttft_ms:6.0f} ms  |  out {m.decode_tps:5.1f} t/s"
            f"  ({m.completion_tokens} tok)  |  CPU avg {avg_cpu:5.1f}%  |  Mem {peak_mem:.1f} GB",
            flush=True,
        )

    n = len(runs)
    avg = tuple(sum(r[k] for r in runs) / n for k in range(5))
    print(
        f"  {'AVG':>8}  TTFT {avg[0]:6.0f} ms  |  out {avg[1]:5.1f} t/s"
        f"  ({avg[2]:.0f} tok)  |  CPU avg {avg[3]:5.1f}%  |  Mem {avg[4]:.1f} GB",
        flush=True,
    )
    summary.append((model, *avg))

print(f"\n\n{'='*75}")
print(f"  BENCHMARK SUMMARY  (avg of {RUNS} runs, /api/chat, seed={DEFAULT_SAMPLING['seed']})")
print(f"{'='*75}")
print(f"  {'Model':<32}  {'TTFT':>8}  {'Out t/s':>7}  {'Tokens':>7}  {'CPU%':>6}  {'Mem GB':>7}")
for model, ttft, out, tokens, avg_cpu, peak_mem in summary:
    print(f"  {model:<32}  {ttft:6.0f} ms  {out:7.1f}  {tokens:7.0f}  {avg_cpu:6.1f}  {peak_mem:7.1f}")
print()
