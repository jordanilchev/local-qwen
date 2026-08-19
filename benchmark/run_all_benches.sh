#!/usr/bin/env bash
# Unified fair benchmark: MLX-first sessions per model family, single session_id.
#
# Order per family (bench_session.py):
#   plain-mlx → vllm-mlx → ddtree-mlx → ollama → llama.cpp [→ llama.cpp MTP]
#
# Families (MLX-heavy first): 3.6-35b-moe → 3.8-27b → 3.6-27b
# Then optional plain-MLX extras (OptiQ).
#
# Usage: ./benchmark/run_all_benches.sh
set -u
cd "$(dirname "$0")/.."

RUN_ID="${BENCH_SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="/tmp"
STATUS_FILE="$LOG_DIR/run_all_benches_status.json"
INTER_FAMILY_COOL_S=180
FAMILY_TRIES=3

FAMILIES=(3.6-35b-moe 3.8-27b 3.6-27b)

ensure_ollama() {
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] starting ollama serve in background"
    nohup ollama serve >"$LOG_DIR/ollama_serve_$RUN_ID.log" 2>&1 &
    sleep 5
  fi
}

stop_ollama() {
  pkill -f 'ollama serve' 2>/dev/null || true
  sleep 3
}

declare -a STATUSES

echo "[$(date -u +%FT%TZ)] run_all_benches start (session_id=$RUN_ID)"

# Pull Ollama models up front (fail fast).
ensure_ollama
for tag in qwen3.6:35b qwen3.8:27b-q4_K_M qwen3.6:27b; do
  echo "[$(date -u +%FT%TZ)] ollama pull $tag"
  ollama pull "$tag" || exit 1
done
stop_ollama

for i in "${!FAMILIES[@]}"; do
  family="${FAMILIES[$i]}"
  log="$LOG_DIR/run_all_benches_${family}_${RUN_ID}.log"

  if [ "$i" -gt 0 ]; then
    echo "[$(date -u +%FT%TZ)] cooldown ${INTER_FAMILY_COOL_S}s before $family"
    sleep "$INTER_FAMILY_COOL_S"
  fi

  ensure_ollama
  echo "[$(date -u +%FT%TZ)] >>> session family=$family session_id=$RUN_ID → $log"
  t0=$(date +%s)
  rc=1
  for try in $(seq 1 "$FAMILY_TRIES"); do
    echo "[$(date -u +%FT%TZ)] family $family attempt $try/$FAMILY_TRIES"
    BENCH_FAMILY="$family" BENCH_SESSION_ID="$RUN_ID" \
      HF_HOME="${HF_HOME:-$HOME/Models/HuggingFace}" \
      .venv/bin/python -u -m benchmark.bench_session >"$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      break
    fi
    echo "[$(date -u +%FT%TZ)] family $family attempt $try failed rc=$rc"
    sleep 90
  done
  t1=$(date +%s)
  dt=$((t1 - t0))
  stop_ollama

  if [ "$rc" -eq 0 ]; then
    STATUSES+=("\"$family\":{\"status\":\"ok\",\"duration_s\":$dt,\"log\":\"$log\"}")
    echo "[$(date -u +%FT%TZ)] <<< $family OK in ${dt}s"
  else
    STATUSES+=("\"$family\":{\"status\":\"failed\",\"rc\":$rc,\"duration_s\":$dt,\"log\":\"$log\"}")
    echo "[$(date -u +%FT%TZ)] <<< $family FAILED (rc=$rc) in ${dt}s — continuing"
  fi
done

# Optional OptiQ extra (plain-MLX only, no cross-backend row).
extras_log="$LOG_DIR/run_all_benches_extras_${RUN_ID}.log"
echo "[$(date -u +%FT%TZ)] >>> extras (OptiQ) → $extras_log"
t0=$(date +%s)
BENCH_SESSION_ID="$RUN_ID" HF_HOME="${HF_HOME:-$HOME/Models/HuggingFace}" \
  .venv/bin/python -u -m benchmark.bench_extras >"$extras_log" 2>&1
rc=$?
t1=$(date +%s)
if [ "$rc" -eq 0 ]; then
  STATUSES+=("\"extras\":{\"status\":\"ok\",\"duration_s\":$((t1 - t0)),\"log\":\"$extras_log\"}")
else
  STATUSES+=("\"extras\":{\"status\":\"failed\",\"rc\":$rc,\"duration_s\":$((t1 - t0)),\"log\":\"$extras_log\"}")
fi

summary_log="$LOG_DIR/run_all_benches_summary_${RUN_ID}.log"
BENCH_SESSION_ID="$RUN_ID" .venv/bin/python -m benchmark.summarize --session-id "$RUN_ID" \
  >"$summary_log" 2>&1 || true

{
  echo "{"
  echo "  \"session_id\":\"$RUN_ID\","
  echo "  \"completed_at\":\"$(date -u +%FT%TZ)\","
  echo "  \"summary_log\":\"$summary_log\","
  echo "  \"results\":{"
  echo "    $(IFS=,; echo "${STATUSES[*]}")"
  echo "  }"
  echo "}"
} > "$STATUS_FILE"

echo "[$(date -u +%FT%TZ)] all done. status → $STATUS_FILE"
cat "$STATUS_FILE"
echo ""
echo "Summary table → $summary_log"
