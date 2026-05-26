#!/usr/bin/env bash
# Run bench_llamacpp_mtp after the Ollama compare re-run finishes.
# Usage:
#   ./benchmark/rerun_llamacpp_mtp_after_ollama.sh          # wait for any rerun_compare_ollama.sh
#   ./benchmark/rerun_llamacpp_mtp_after_ollama.sh <pid>    # wait for a specific PID
set -u
cd "$(dirname "$0")/.."

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/rerun_llamacpp_mtp_${RUN_ID}.log"
STATUS="/tmp/rerun_llamacpp_mtp_status.json"
WAIT_PID="${1:-}"
LLAMACPP_SERVER="${LLAMACPP_SERVER:-/opt/homebrew/bin/llama-server}"

if [ -n "$WAIT_PID" ]; then
  echo "[$(date -u +%FT%TZ)] waiting for PID $WAIT_PID to exit..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
else
  echo "[$(date -u +%FT%TZ)] waiting for rerun_compare_ollama.sh to finish..."
  while pgrep -f "benchmark/rerun_compare_ollama.sh" >/dev/null 2>&1; do
    sleep 60
  done
fi

echo "[$(date -u +%FT%TZ)] Ollama re-run done — cooldown 120s before llama.cpp MTP"
sleep 120

pkill -f 'ollama serve' 2>/dev/null || true
sleep 3

if [ ! -x "$LLAMACPP_SERVER" ] && [ ! -f "$LLAMACPP_SERVER" ]; then
  echo "[$(date -u +%FT%TZ)] ERROR: llama-server not found at $LLAMACPP_SERVER" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] starting bench_llamacpp_mtp → $LOG"
t0=$(date +%s)
LLAMACPP_SERVER="$LLAMACPP_SERVER" HF_HOME="${HF_HOME:-$HOME/Models/HuggingFace}" \
  .venv/bin/python -u -m benchmark.bench_llamacpp_mtp >"$LOG" 2>&1
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

echo "[$(date -u +%FT%TZ)] bench_llamacpp_mtp $status (rc=$rc) in ${dt}s — see $LOG"
cat "$STATUS"
