#!/usr/bin/env bash
# Run the DDTree tree_budget sweep one budget at a time, each in a fresh Python process.
# Rationale: a single process holding 22 GB of model + drafter for 90+ minutes on a fanless
# M4 Air OOM'd in our prior run. Fresh process per budget = clean RAM + clean Metal context,
# at the cost of one extra ~30 s model load per budget.
#
# Each budget run writes its own JSON to benchmark/results/. Logs go to /tmp/.
# Final state file: /tmp/sweep_overnight_status.json
#
# Usage: ./benchmark/run_sweep_overnight.sh
set -u
cd "$(dirname "$0")/.."

BUDGETS=(2 3 4 5 6)
INTER_BUDGET_COOL_S=90
RETRIES_PER_BUDGET=1
LOG_DIR="/tmp"
STATUS_FILE="/tmp/sweep_overnight_status.json"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)

declare -a STATUSES

echo "[$(date -u +%FT%TZ)] sweep_overnight start (run_id=$RUN_ID)"
echo "[$(date -u +%FT%TZ)] BUDGETS=${BUDGETS[*]}  cool=${INTER_BUDGET_COOL_S}s  retries=${RETRIES_PER_BUDGET}"

# Stop any Ollama daemon competing for RAM
pkill -f 'ollama serve' 2>/dev/null
sleep 3

for i in "${!BUDGETS[@]}"; do
  b="${BUDGETS[$i]}"
  log="${LOG_DIR}/sweep_b${b}_${RUN_ID}.log"
  status="UNKNOWN"

  for attempt in $(seq 0 ${RETRIES_PER_BUDGET}); do
    echo "[$(date -u +%FT%TZ)] budget=$b attempt=$((attempt+1)) -> $log"
    BUDGETS_TO_RUN=$b HF_HOME="$HOME/Models/HuggingFace" PYTHONPATH=. \
      .venv/bin/python benchmark/bench_tree_budget_sweep.py >>"$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
      # Verify a JSON for this budget actually landed
      if ls -1 benchmark/results/ddtree-budget-sweep_*tree_budget=${b}* 2>/dev/null | head -1 | grep -q . \
         || ls -1 benchmark/results/ddtree-budget-sweep_*.json 2>/dev/null | xargs -I{} grep -l "\"tree_budget\": ${b}" {} 2>/dev/null | head -1 | grep -q .; then
        status="OK"
        echo "[$(date -u +%FT%TZ)] budget=$b OK"
        break
      else
        status="OK_NO_JSON"
        echo "[$(date -u +%FT%TZ)] budget=$b script exited 0 but no JSON written, retrying..."
      fi
    else
      status="EXIT_${rc}"
      echo "[$(date -u +%FT%TZ)] budget=$b rc=$rc, retrying..."
    fi
    sleep 30
  done

  STATUSES+=("\"$b\":\"$status\"")

  if [[ $i -lt $((${#BUDGETS[@]} - 1)) ]]; then
    echo "[$(date -u +%FT%TZ)] cool-down ${INTER_BUDGET_COOL_S}s before next budget"
    sleep "$INTER_BUDGET_COOL_S"
  fi
done

# Write status file
{
  echo "{"
  echo "  \"run_id\": \"$RUN_ID\","
  echo "  \"finished_at\": \"$(date -u +%FT%TZ)\","
  echo "  \"budgets\": {$(IFS=,; echo "${STATUSES[*]}")}"
  echo "}"
} > "$STATUS_FILE"

echo "[$(date -u +%FT%TZ)] sweep_overnight done -> $STATUS_FILE"
cat "$STATUS_FILE"
