---
name: wire-cursor-local
description: >-
  Wires any local OpenAI-compatible inference server to Cursor Chat/Agent via a
  public HTTPS Cloudflare quick tunnel, always validates (models + smoke + coding
  + agentic), then prints Settings → Models values. Use when the user says wire
  Cursor, connect Cursor to a local model/server, tunnel localhost for Override
  OpenAI Base URL, or expose ollama/vLLM/llama.cpp/MLX/DFlash/DDTree/
  OpenAI-compatible ports to Cursor.
---

# Wire Cursor → local OpenAI-compatible server

Cursor Chat/Agent **cannot** use `http://localhost:…`. Requests go through Cursor’s backend, so the Base URL must be **public HTTPS**. There is **no** supported file config for Override Base URL — **Cursor Settings → Models** only.

Works with **any** server that speaks OpenAI Chat Completions (`/v1/chat/completions`), regardless of model or engine.

## Do this

1. Ensure the local server is up **or** pass `--start '…'`.

2. Run:

```bash
# From this repo:
./.cursor/skills/wire-cursor-local/scripts/up.sh --port PORT --model MODEL_ID

# Or personal install:
~/.cursor/skills/wire-cursor-local/scripts/up.sh --port PORT --model MODEL_ID
```

`up.sh` **always validates** before printing Cursor values (fails closed):

| Step | What |
|------|------|
| local `validate.py` | `/v1/models` · smoke · coding · agentic · **Cursor agent token test** |
| tunnel smoke | `/v1/models` + short chat via `curl --resolve` (public DNS) |

**Cursor agent token test** (`cursor_agent_token_test.py`): multi-turn plan → tool/file result → subagent follow-up with synthetic open-editor context; asserts prompt tokens grow and completions are non-empty.

Results: `/tmp/wire-cursor-<name>-validate.json` (+ `*-cursor-agent.json`)

Omit `--model` to take the first id from `GET /v1/models`.

Optional flags: `--health`, `--api-prefix`, `--start`, `--name`, `--api-key`, `--caffeinate`.

3. Paste printed values into **Cursor Settings → Models**, then select the model.

## Agent checklist

- [ ] Local validate **PASS** (models + smoke + coding + agentic + cursor-agent-token)
- [ ] Tunnel smoke **PASS** when public DNS resolves (else noted warn)
- [ ] User got: API key, Base URL with `/v1`, model id
- [ ] Reminded: Tab stays Cursor-hosted; re-run after cloudflared restart
- [ ] Did **not** use localhost Base URL; did **not** append `/chat/completions`
- [ ] Did **not** dig through `state.vscdb` for secrets

## Standalone validate / Cursor token test

```bash
python3 ~/.cursor/skills/wire-cursor-local/scripts/validate.py \
  --base http://127.0.0.1:PORT/v1 \
  --model MODEL_ID \
  --out /tmp/wire-validate.json

# Cursor-shaped multi-turn token series only:
python3 ~/.cursor/skills/wire-cursor-local/scripts/cursor_agent_token_test.py \
  --base http://127.0.0.1:PORT/v1 \
  --model MODEL_ID \
  --context-chars 6000 \
  --out /tmp/wire-cursor-agent-token.json
```

Prefer running the token test via a **subagent** (keeps the parent session light); return prompt/completion series + PASS/FAIL only.
## Examples

```bash
~/.cursor/skills/wire-cursor-local/scripts/up.sh --port 8007 --model qwen3.8-27b-dflash2

~/.cursor/skills/wire-cursor-local/scripts/up.sh \
  --port 8007 \
  --model qwen3.8-27b-dflash2 \
  --caffeinate \
  --start 'HF_HOME=$HOME/Models/HuggingFace .venv/bin/python scripts/dflash2_server.py --port 8007 --max-context 32768'

~/.cursor/skills/wire-cursor-local/scripts/up.sh --port 8080 --model my-gguf
```

## DNS note

Some LAN resolvers NXDOMAIN `*.trycloudflare.com`. Tunnel smoke uses `dig @1.1.1.1` + `curl --resolve`.
