#!/usr/bin/env bash
# Re-run bench_ddtree after the main run_all_benches process exits.
# Usage:
#   ./benchmark/rerun_ddtree_after_main.sh          # wait for any run_all_benches.sh
#   ./benchmark/rerun_ddtree_after_main.sh <pid>    # wait for a specific PID
set -u
cd "$(dirname "$0")/.."

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/rerun_ddtree_${RUN_ID}.log"
STATUS="/tmp/rerun_ddtree_status.json"
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "[$(date -u +%FT%TZ)] waiting for PID $WAIT_PID to exit..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
else
  echo "[$(date -u +%FT%TZ)] waiting for run_all_benches.sh to finish..."
  while pgrep -f "benchmark/run_all_benches.sh" >/dev/null 2>&1; do
    sleep 60
  done
fi

echo "[$(date -u +%FT%TZ)] main suite done — cooldown 120s before DDTree re-run"
sleep 120

pkill -f 'ollama serve' 2>/dev/null || true
sleep 3

echo "[$(date -u +%FT%TZ)] starting bench_ddtree → $LOG"
t0=$(date +%s)
HF_HOME="${HF_HOME:-$HOME/Models/HuggingFace}" \
  .venv/bin/python -u -m benchmark.bench_ddtree >"$LOG" 2>&1
rc=$?
t1=$(date +%s)
dt=$((t1 - t0))

if [ "$rc" -eq 0 ]; then
  status="ok"
else
  status="failed"
fi

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

echo "[$(date -u +%FT%TZ)] bench_ddtree $status (rc=$rc) in ${dt}s — see $LOG"
cat "$STATUS"
