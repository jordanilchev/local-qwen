#!/usr/bin/env bash
# Run every benchmark sequentially against the currently-available models.
# Each bench runs in a fresh Python process (clean RAM/Metal context). Cooldowns
# between benches give the fanless M4 time to recover thermally.
#
# Order is light → heavy to fail fast on environment issues:
#   1. bench_llamacpp_mtp.py     llama.cpp baseline vs MTP
#   2. bench_extras.py           plain-MLX on Qwen3.6-27B + Qwen3.6-27B-OptiQ
#   3. bench_ddtree.py           DDTree on Qwen3.6-35B-A3B-4bit-DWQ
#   4. bench_vllm.py             vllm-mlx on Qwen3.6-27B + Qwen3.6-35B-A3B
#   5. bench_compare.py          Ollama (3 models) + MLX/DDTree on 35B MoE
#
# Per-bench logs: /tmp/run_all_benches_<bench>_<run_id>.log
# Status JSON:    /tmp/run_all_benches_status.json
#
# Usage: ./benchmark/run_all_benches.sh
set -u
cd "$(dirname "$0")/.."

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="/tmp"
STATUS_FILE="$LOG_DIR/run_all_benches_status.json"
INTER_BENCH_COOL_S=120

BENCHES=(
  "llamacpp_mtp:benchmark.bench_llamacpp_mtp"
  "extras:benchmark.bench_extras"
  "ddtree:benchmark.bench_ddtree"
  "vllm:benchmark.bench_vllm"
  "compare:benchmark.bench_compare"
)

# Ensure Ollama is running for bench_compare; harmless if already running.
ensure_ollama() {
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] starting ollama serve in background"
    nohup ollama serve >/tmp/ollama_serve_$RUN_ID.log 2>&1 &
    sleep 5
  fi
}

# Stop Ollama before non-Ollama benches to free RAM for MLX.
stop_ollama() {
  pkill -f 'ollama serve' 2>/dev/null
  sleep 3
}

declare -a STATUSES

echo "[$(date -u +%FT%TZ)] run_all_benches start (run_id=$RUN_ID)"

for i in "${!BENCHES[@]}"; do
  spec="${BENCHES[$i]}"
  name="${spec%%:*}"
  module="${spec#*:}"
  log="$LOG_DIR/run_all_benches_${name}_${RUN_ID}.log"

  if [ "$i" -gt 0 ]; then
    echo "[$(date -u +%FT%TZ)] cooldown ${INTER_BENCH_COOL_S}s before next bench"
    sleep $INTER_BENCH_COOL_S
  fi

  case "$name" in
    compare) ensure_ollama ;;
    *)       stop_ollama ;;
  esac

  echo "[$(date -u +%FT%TZ)] >>> $name → log $log"
  t0=$(date +%s)
  HF_HOME="${HF_HOME:-$HOME/Models/HuggingFace}" \
    .venv/bin/python -u -m "$module" >"$log" 2>&1
  rc=$?
  t1=$(date +%s)
  dt=$((t1 - t0))

  if [ "$rc" -eq 0 ]; then
    STATUSES+=("\"$name\":{\"status\":\"ok\",\"duration_s\":$dt,\"log\":\"$log\"}")
    echo "[$(date -u +%FT%TZ)] <<< $name OK in ${dt}s"
  else
    STATUSES+=("\"$name\":{\"status\":\"failed\",\"rc\":$rc,\"duration_s\":$dt,\"log\":\"$log\"}")
    echo "[$(date -u +%FT%TZ)] <<< $name FAILED (rc=$rc) in ${dt}s — continuing"
  fi
done

# Write final status JSON
{
  echo "{"
  echo "  \"run_id\":\"$RUN_ID\","
  echo "  \"completed_at\":\"$(date -u +%FT%TZ)\","
  echo "  \"results\":{"
  echo "    $(IFS=,; echo "${STATUSES[*]}")"
  echo "  }"
  echo "}"
} > "$STATUS_FILE"

echo "[$(date -u +%FT%TZ)] all done. status -> $STATUS_FILE"
cat "$STATUS_FILE"
