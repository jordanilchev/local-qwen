#!/usr/bin/env bash
# Unattended: sequential HF downloads (resume + login + retry + stale watchdog), then benches.
# Log: /tmp/complete_unattended.log
set -u

# Keep the Mac awake for the whole pipeline (idle sleep stalled Hub transfers overnight).
if [ "${CAFFEINATED:-0}" != 1 ]; then
  export CAFFEINATED=1
  exec /usr/bin/caffeinate -dims "$0" "$@"
fi

cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/Models/HuggingFace}"
export LLAMA_CACHE="${LLAMA_CACHE:-$HOME/Models/llamacpp}"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

LOG=/tmp/complete_unattended.log
STATUS=/tmp/complete_unattended_status.json
RUN_ID="${BENCH_SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export BENCH_SESSION_ID="$RUN_ID"
PY=.venv/bin/python
MAX_TRIES=6
STALE_SECS="${STALE_SECS:-300}"
STALE_POLL_S=20

ts() { date -u +%FT%TZ; }

retry() {
  local name="$1"
  shift
  local n=0
  local delay=20
  while true; do
    n=$((n + 1))
    echo "[$(ts)] try $n/$MAX_TRIES $name"
    if "$@"; then
      echo "[$(ts)] ok $name"
      return 0
    fi
    if [ "$n" -ge "$MAX_TRIES" ]; then
      echo "[$(ts)] FAIL $name after $MAX_TRIES tries"
      return 1
    fi
    echo "[$(ts)] retry $name in ${delay}s"
    sleep "$delay"
    delay=$((delay * 2))
    if [ "$delay" -gt 300 ]; then delay=300; fi
  done
}

repo_dir() {
  local repo="$1"
  local cache="${2:-$HF_HOME}"
  local slug="models--${repo//\//--}"
  if [ -d "$cache/hub/$slug" ]; then
    echo "$cache/hub/$slug"
  elif [ -d "$cache/$slug" ]; then
    echo "$cache/$slug"
  elif [ "$cache" = "$HF_HOME" ]; then
    echo "$HF_HOME/hub/$slug"
  else
    echo "$cache/hub/$slug"
  fi
}

dir_kb() {
  local d="$1"
  if [ -d "$d" ]; then
    du -sk "$d" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

# Kill pid if on-disk size does not increase for STALE_SECS (last-night stall).
watch_growth() {
  local pid="$1"
  local dir="$2"
  local label="$3"
  local last kb stale=0
  last=$(dir_kb "$dir")
  echo "[$(ts)] watch $label dir=$dir start_kb=$last stale_after=${STALE_SECS}s"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$STALE_POLL_S"
    kill -0 "$pid" 2>/dev/null || break
    kb=$(dir_kb "$dir")
    if [ "$kb" -gt "$last" ]; then
      echo "[$(ts)] growth $label ${last}k → ${kb}k"
      last=$kb
      stale=0
    else
      stale=$((stale + STALE_POLL_S))
      echo "[$(ts)] no-growth $label kb=$kb stagnant=${stale}s/${STALE_SECS}s"
      if [ "$stale" -ge "$STALE_SECS" ]; then
        echo "[$(ts)] STALE $label — killing pid $pid (resume on retry)"
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
        return 1
      fi
    fi
  done
  wait "$pid"
  return $?
}

snap() {
  local repo="$1"
  local patterns="${2:-}"
  local cache="${3:-}"
  local dir
  dir=$(repo_dir "$repo" "${cache:-$HF_HOME}")
  echo "[$(ts)] snapshot $repo patterns=${patterns:-all} cache=${cache:-$HF_HOME} dir=$dir"
  "$PY" -u -c "
from huggingface_hub import snapshot_download, whoami
import sys
print('hf user:', whoami().get('name'), flush=True)
kwargs = dict(repo_id=sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2]:
    kwargs['allow_patterns'] = sys.argv[2].split(',')
if len(sys.argv) > 3 and sys.argv[3]:
    kwargs['cache_dir'] = sys.argv[3]
path = snapshot_download(**kwargs)
print('DONE', path, flush=True)
" "$repo" "$patterns" "${cache:-}" &
  watch_growth $! "$dir" "$repo"
}

ensure_ollama() {
  if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null; then
    echo "[$(ts)] starting ollama serve"
    nohup ollama serve >/tmp/ollama_serve_unattended.log 2>&1 &
    sleep 5
  fi
  curl -sf http://127.0.0.1:11434/api/version >/dev/null
}

ollama_pull_watched() {
  local tag="$1"
  local dir="${HOME}/.ollama/models"
  echo "[$(ts)] ollama pull $tag"
  ollama pull "$tag" &
  watch_growth $! "$dir" "ollama:$tag"
}

{
  echo "[$(ts)] complete_unattended start run_id=$RUN_ID stale_secs=$STALE_SECS"
  pkill -f snapshot_download 2>/dev/null || true
  sleep 2

  retry "mlx-community/Qwen3.8-27B-4bit" snap mlx-community/Qwen3.8-27B-4bit || exit 1
  retry "unsloth/Qwen3.8-27B-GGUF UD-Q4_K_XL" snap unsloth/Qwen3.8-27B-GGUF '*UD-Q4_K_XL*' "$LLAMA_CACHE" || exit 1
  retry "z-lab/Qwen3.8-27B-DFlash2" snap z-lab/Qwen3.8-27B-DFlash2 || exit 1

  retry "ollama serve" ensure_ollama || exit 1
  retry "ollama pull qwen3.8:27b-q4_K_M" ollama_pull_watched qwen3.8:27b-q4_K_M || exit 1

  echo "[$(ts)] downloads phase done; thermal floor 180s before benches"
  sleep 180

  echo "[$(ts)] starting run_all_benches.sh session_id=$RUN_ID"
  retry "run_all_benches.sh" ./benchmark/run_all_benches.sh || echo "[$(ts)] WARN benches ended non-zero after retries"

  "$PY" -m benchmark.summarize --session-id "$RUN_ID" || true
  echo "[$(ts)] complete_unattended finished"
} >>"$LOG" 2>&1

echo "{\"run_id\":\"$RUN_ID\",\"log\":\"$LOG\",\"finished_at\":\"$(ts)\"}" >"$STATUS"
