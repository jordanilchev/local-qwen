#!/usr/bin/env bash
# Pull Ollama models (if needed) and rerun bench_compare Ollama phase only.
set -u
cd "$(dirname "$0")/.."

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/rerun_compare_ollama_${RUN_ID}.log"
STATUS="/tmp/rerun_compare_ollama_status.json"

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[$(date -u +%FT%TZ)] starting ollama serve"
  ollama serve >/tmp/ollama_serve_${RUN_ID}.log 2>&1 &
  sleep 3
fi

echo "[$(date -u +%FT%TZ)] pulling Ollama models (skip if cached)..."
for m in qwen3.6:35b qwen3.8:27b-q4_K_M qwen3.6:27b; do
  echo "[$(date -u +%FT%TZ)] ollama pull $m"
  ollama pull "$m" || exit 1
done

echo "[$(date -u +%FT%TZ)] starting bench_compare (BENCH_PHASE=ollama) → $LOG"
t0=$(date +%s)
BENCH_PHASE=ollama HF_HOME="${HF_HOME:-$HOME/Models/HuggingFace}" \
  .venv/bin/python -u -m benchmark.bench_compare >"$LOG" 2>&1
rc=$?
t1=$(date +%s)
dt=$((t1 - t0))

if [ "$rc" -eq 0 ]; then status="ok"; else status="failed"; fi
{
  echo "{"
  echo "  \"run_id\":\"$RUN_ID\","
  echo "  \"status\":\"$status\","
  echo "  \"rc\":$rc,"
  echo "  \"duration_s\":$dt,"
  echo "  \"log\":\"$LOG\","
  echo "  \"completed_at\":\"$(date -u +%FT%TZ)\""
  echo "}"
} > "$STATUS"

echo "[$(date -u +%FT%TZ)] done ($status, rc=$rc) in ${dt}s — $LOG"
cat "$STATUS"
