#!/usr/bin/env bash
# Tunnel any local OpenAI-compatible server and print Cursor Settings → Models values.
set -euo pipefail

PORT=""
MODEL=""
HEALTH="/health"
API_PREFIX="/v1"
START_CMD=""
NAME=""
API_KEY="local"
USE_CAFFEINATE=0
WAIT_SECS="${WAIT_SECS:-240}"

usage() {
  cat <<'EOF'
Usage: up.sh --port PORT [--model MODEL_ID] [options]

  --port PORT           Local listen port (required)
  --model ID            Model id for Cursor (default: first from /v1/models)
  --health PATH         Local health path (default: /health; falls back to /v1/models)
  --api-prefix PATH     Cursor Base URL suffix (default: /v1)
  --start 'CMD'         Shell command to start the server if not healthy
  --name TAG            Log/url file tag (default: port)
  --api-key KEY         Printed OpenAI API key placeholder (default: local)
  --caffeinate          Wrap --start with: caffeinate -dims …
  -h, --help            Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --health) HEALTH="${2:?}"; shift 2 ;;
    --api-prefix) API_PREFIX="${2:?}"; shift 2 ;;
    --start) START_CMD="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --api-key) API_KEY="${2:?}"; shift 2 ;;
    --caffeinate) USE_CAFFEINATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PORT" ]]; then
  echo "--port is required" >&2
  usage >&2
  exit 2
fi

NAME="${NAME:-$PORT}"
LOG_SERVER="${LOG_SERVER:-/tmp/wire-cursor-${NAME}-server.log}"
LOG_TUNNEL="${LOG_TUNNEL:-/tmp/wire-cursor-${NAME}-tunnel.log}"
URL_FILE="${URL_FILE:-/tmp/wire-cursor-${NAME}-tunnel.url}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found; install with: brew install cloudflared" >&2
  exit 1
fi

local_get() {
  local path="$1"
  curl -sf -m 3 "http://127.0.0.1:${PORT}${path}" 2>/dev/null
}

health_ok() {
  if local_get "$HEALTH" >/dev/null; then
    return 0
  fi
  # Common OpenAI-compatible fallback
  if [[ "$HEALTH" != "/v1/models" ]] && local_get "/v1/models" >/dev/null; then
    return 0
  fi
  return 1
}

discover_model() {
  local body
  body="$(local_get "/v1/models" || true)"
  if [[ -z "$body" ]]; then
    return 1
  fi
  python3 -c '
import json,sys
d=json.load(sys.stdin)
data=d.get("data") or []
if not data:
    sys.exit(1)
print(data[0].get("id") or "")
' <<<"$body" 2>/dev/null
}

start_server() {
  if health_ok; then
    echo "server already up on :${PORT}"
    return
  fi
  if [[ -z "$START_CMD" ]]; then
    echo "nothing listening on :${PORT}${HEALTH} (or /v1/models)." >&2
    echo "Start your server, or re-run with --start '…'" >&2
    exit 1
  fi
  : >"$LOG_SERVER"
  local launch=()
  if [[ "$USE_CAFFEINATE" -eq 1 ]] && command -v caffeinate >/dev/null 2>&1; then
    launch=(caffeinate -dims bash -lc "$START_CMD")
  else
    launch=(bash -lc "$START_CMD")
  fi
  nohup "${launch[@]}" >"$LOG_SERVER" 2>&1 &
  echo "starting server pid=$! (log $LOG_SERVER)"
  local n=$((WAIT_SECS / 2))
  local i
  for i in $(seq 1 "$n"); do
    if health_ok; then
      echo "server ready"
      return
    fi
    sleep 2
  done
  echo "server startup timeout; tail $LOG_SERVER:" >&2
  tail -40 "$LOG_SERVER" >&2 || true
  exit 1
}

extract_tunnel_url() {
  rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_TUNNEL" 2>/dev/null | head -1 || true
}

start_tunnel() {
  pkill -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" 2>/dev/null || true
  sleep 1
  : >"$LOG_TUNNEL"
  nohup cloudflared tunnel --url "http://127.0.0.1:${PORT}" >"$LOG_TUNNEL" 2>&1 &
  echo "starting cloudflared pid=$! (log $LOG_TUNNEL)"
  local url=""
  local _
  for _ in $(seq 1 45); do
    url="$(extract_tunnel_url)"
    if [[ -n "$url" ]]; then
      printf '%s\n' "$url" >"$URL_FILE"
      echo "tunnel=$url"
      return 0
    fi
    sleep 1
  done
  echo "tunnel URL timeout; see $LOG_TUNNEL" >&2
  tail -40 "$LOG_TUNNEL" >&2 || true
  exit 1
}

probe_tunnel() {
  local base="$1"
  local path="${2:-$HEALTH}"
  local host="${base#https://}"
  local ip=""
  local _
  for _ in $(seq 1 20); do
    ip="$(dig +short "$host" @1.1.1.1 | head -1 || true)"
    [[ -n "$ip" ]] && break
    sleep 1
  done
  if [[ -z "$ip" ]]; then
    echo "warn: public DNS has no A for $host yet; Cursor may still resolve it" >&2
    return 0
  fi
  if curl -sf -m 20 --resolve "${host}:443:${ip}" "${base}${path}" >/dev/null \
    || curl -sf -m 20 --resolve "${host}:443:${ip}" "${base}/v1/models" >/dev/null; then
    echo "tunnel health OK (via 1.1.1.1 → $ip)"
  else
    echo "warn: tunnel health probe failed; check $LOG_TUNNEL" >&2
  fi
}

start_server

if [[ -z "$MODEL" ]]; then
  MODEL="$(discover_model || true)"
  if [[ -z "$MODEL" ]]; then
    echo "could not discover model id from /v1/models; pass --model" >&2
    exit 1
  fi
  echo "discovered model=$MODEL"
fi

# Normalize api prefix (leading slash, no trailing slash)
API_PREFIX="/${API_PREFIX#/}"
API_PREFIX="${API_PREFIX%/}"
[[ "$API_PREFIX" == "/" ]] && API_PREFIX=""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VALIDATE_PY="${SCRIPT_DIR}/validate.py"
VALIDATE_OUT="${VALIDATE_OUT:-/tmp/wire-cursor-${NAME}-validate.json}"

echo
echo "=== validate (local) ==="
python3 "$VALIDATE_PY" \
  --base "http://127.0.0.1:${PORT}${API_PREFIX}" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --out "$VALIDATE_OUT"

start_tunnel
TUNNEL_URL="$(cat "$URL_FILE")"
probe_tunnel "$TUNNEL_URL" "$HEALTH"

# Tunnel smoke via public DNS + curl --resolve (LAN DNS often NXDOMAIN trycloudflare).
host="${TUNNEL_URL#https://}"
ip="$(dig +short "$host" @1.1.1.1 | head -1 || true)"
if [[ -n "$ip" ]]; then
  echo
  echo "=== validate (tunnel smoke) ==="
  if curl -sf -m 30 --resolve "${host}:443:${ip}" \
      -H "Authorization: Bearer ${API_KEY}" \
      "${TUNNEL_URL}${API_PREFIX}/models" >/dev/null \
    && curl -sf -m 120 --resolve "${host}:443:${ip}" \
      -H "Authorization: Bearer ${API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK only.\"}],\"max_tokens\":8,\"temperature\":0}" \
      "${TUNNEL_URL}${API_PREFIX}/chat/completions" >/dev/null; then
    echo "tunnel smoke PASS"
  else
    echo "tunnel smoke FAIL" >&2
    exit 1
  fi
else
  echo "warn: no public DNS A for $host; skipped tunnel smoke (local validate PASSED)" >&2
fi

echo
echo "=== Cursor Settings → Models ==="
echo "OpenAI API Key:              ${API_KEY}"
echo "Override OpenAI Base URL:    ${TUNNEL_URL}${API_PREFIX}"
echo "Add custom model:            ${MODEL}"
echo
echo "Validate: PASS → ${VALIDATE_OUT}"
echo "Select model ${MODEL} in Chat/Agent."
echo "Turn Override off when switching back to hosted Cursor models."
echo "Quick-tunnel URL changes if cloudflared restarts — re-run this script."
