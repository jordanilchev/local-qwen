#!/usr/bin/env python3
import json, urllib.request, threading, time, subprocess, sys

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not available — CPU/mem monitoring disabled", flush=True)

PROMPT = (
    "Implement a red-black tree in Python with insert, delete, and search methods. "
    "Include proper rebalancing logic and type hints."
)
RUNS = 3


def get_ollama_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    models = []
    for line in result.stdout.strip().split("\n")[1:]:
        if line.strip():
            models.append(line.split()[0])
    return models


def get_ollama_procs():
    if not HAS_PSUTIL:
        return []
    procs = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = p.info["name"] or ""
            cmd = " ".join(p.info["cmdline"] or [])
            if "ollama" in name.lower() or "ollama" in cmd.lower():
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs


class ResourceMonitor:
    def __init__(self):
        self.samples_cpu = []
        self.samples_mem_gb = []
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
        procs = get_ollama_procs()
        if not procs:
            return
        for p in procs:
            try:
                p.cpu_percent(interval=None)
            except Exception:
                pass
        time.sleep(0.15)

        while not self._stop.is_set():
            cpu_total = 0.0
            mem_total = 0
            for p in procs:
                try:
                    cpu_total += p.cpu_percent(interval=None)
                    mem_total += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.samples_cpu.append(cpu_total)
            self.samples_mem_gb.append(mem_total / (1024 ** 3))
            time.sleep(0.25)

    def stats(self):
        if not self.samples_cpu:
            return 0.0, 0.0, 0.0
        avg_cpu = sum(self.samples_cpu) / len(self.samples_cpu)
        peak_cpu = max(self.samples_cpu)
        peak_mem = max(self.samples_mem_gb) if self.samples_mem_gb else 0.0
        return avg_cpu, peak_cpu, peak_mem


def call(model, num_predict):
    payload = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


MODELS = get_ollama_models()
if not MODELS:
    print("No models found — is ollama running?")
    sys.exit(1)

print(f"Models: {', '.join(MODELS)}", flush=True)
print(f"Runs per model: {RUNS} (+ 1 warmup)\n", flush=True)

monitor = ResourceMonitor()
summary = []

for model in MODELS:
    print(f"\n{'='*75}", flush=True)
    print(f"  {model}", flush=True)
    print(f"{'='*75}", flush=True)

    print("  [warmup] loading model ...", end=" ", flush=True)
    call(model, 64)
    print("done", flush=True)

    runs = []
    for i in range(1, RUNS + 1):
        print(f"  [run {i}/{RUNS}]", end=" ", flush=True)
        monitor.start()
        d = call(model, 200)
        monitor.stop()

        pt  = d.get("prompt_eval_count", 0)
        pns = d.get("prompt_eval_duration", 1)
        gt  = d.get("eval_count", 0)
        gns = d.get("eval_duration", 1)

        ttft_ms   = pns / 1e6
        cache_tps = pt / (pns / 1e9) if pns > 0 else 0.0
        out_tps   = gt / (gns / 1e9) if gns > 0 else 0.0
        avg_cpu, peak_cpu, peak_mem = monitor.stats()

        runs.append((ttft_ms, cache_tps, out_tps, avg_cpu, peak_cpu, peak_mem))
        print(
            f"TTFT {ttft_ms:6.0f} ms  |  cache {cache_tps:6.1f} t/s  |  out {out_tps:5.1f} t/s"
            f"  |  CPU avg {avg_cpu:5.1f}% peak {peak_cpu:5.1f}%  |  Mem {peak_mem:.1f} GB",
            flush=True,
        )

    n = len(runs)
    avg = tuple(sum(r[k] for r in runs) / n for k in range(6))
    print(
        f"  {'AVG':>8}  TTFT {avg[0]:6.0f} ms  |  cache {avg[1]:6.1f} t/s  |  out {avg[2]:5.1f} t/s"
        f"  |  CPU avg {avg[3]:5.1f}%  |  Mem {avg[5]:.1f} GB",
        flush=True,
    )
    summary.append((model, *avg))

    # Running summary after each model completes
    W = 75
    print(f"\n  -- progress ({len(summary)}/{len(MODELS)} models) --")
    print(f"  {'Model':<32}  {'TTFT':>8}  {'Cache t/s':>9}  {'Out t/s':>7}  {'CPU%':>6}  {'Mem GB':>7}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*7}")
    for row in summary:
        m, t, c, o, ac, _, pm = row
        print(f"  {m:<32}  {t:6.0f} ms  {c:9.1f}  {o:7.1f}  {ac:6.1f}  {pm:7.1f}")
    print(flush=True)

W = 75
print(f"\n\n{'='*W}")
print(f"  BENCHMARK SUMMARY  (avg of {RUNS} runs, post-warmup)")
print(f"{'='*W}")
print(f"  {'Model':<32}  {'TTFT':>8}  {'Cache t/s':>9}  {'Out t/s':>7}  {'CPU%':>6}  {'Mem GB':>7}")
print(f"  {'-'*32}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*7}")
for row in summary:
    model, ttft, cache, out, avg_cpu, _, peak_mem = row
    print(f"  {model:<32}  {ttft:6.0f} ms  {cache:9.1f}  {out:7.1f}  {avg_cpu:6.1f}  {peak_mem:7.1f}")
print()
