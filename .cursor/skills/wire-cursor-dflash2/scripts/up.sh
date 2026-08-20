#!/usr/bin/env bash
# local-qwen DFlash2 profile → wire-cursor-local (repo copy, else personal)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
REPO_UP="$ROOT/.cursor/skills/wire-cursor-local/scripts/up.sh"
HOME_UP="${HOME}/.cursor/skills/wire-cursor-local/scripts/up.sh"
if [[ -x "$REPO_UP" ]]; then
  UP="$REPO_UP"
elif [[ -x "$HOME_UP" ]]; then
  UP="$HOME_UP"
else
  echo "wire-cursor-local not found (repo or ~/.cursor/skills)" >&2
  exit 1
fi
exec "$UP" \
  --port 8007 \
  --model qwen3.8-27b-dflash2 \
  --name dflash2 \
  --caffeinate \
  --start "HF_HOME=\$HOME/Models/HuggingFace \"$ROOT/.venv/bin/python\" scripts/dflash2_server.py --port 8007 --max-context 32768" \
  "$@"
