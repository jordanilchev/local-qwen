#!/usr/bin/env bash
# Run Ollama compare (27b + 35b only) after llama.cpp MTP finishes.
# Usage:
#   ./benchmark/rerun_compare_ollama_after_mtp.sh          # wait for any MTP rerun
#   ./benchmark/rerun_compare_ollama_after_mtp.sh <pid>    # wait for a specific PID
set -u
cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "[$(date -u +%FT%TZ)] waiting for PID $WAIT_PID to exit..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
else
  echo "[$(date -u +%FT%TZ)] waiting for rerun_llamacpp_mtp_after_ollama.sh to finish..."
  while pgrep -f "rerun_llamacpp_mtp_after_ollama.sh" >/dev/null 2>&1; do
    sleep 60
  done
  while pgrep -f "benchmark.bench_llamacpp_mtp" >/dev/null 2>&1; do
    sleep 60
  done
fi

echo "[$(date -u +%FT%TZ)] MTP done — cooldown 120s before Ollama compare"
sleep 120

exec ./benchmark/rerun_compare_ollama.sh
