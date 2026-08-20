#!/usr/bin/env python3
"""Validate an OpenAI-compatible chat server (local or tunneled).

Always used by wire-cursor-local up.sh. Exit 0 only if all checks pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CODING_PROMPT = (
    "Implement a thread-safe LRU cache in Python using collections.OrderedDict. "
    "Support get(key, default=None), put(key, value), and delete(key). "
    "Include type hints. Keep the answer under 80 lines."
)

AGENTIC_SYSTEM = (
    "You are a coding agent. Prefer short plans, then concrete code. No fluff."
)
AGENTIC_PLAN = (
    "Bug: parse_args crashes when argv is empty. Plan the fix in 3 bullets, then wait."
)
AGENTIC_EDIT = (
    "Here is the file:\n\n"
    "```python\n"
    "import sys\n\n"
    "def parse_args(argv=None):\n"
    "    argv = argv if argv is not None else sys.argv\n"
    "    return argv[1]\n"
    "```\n\n"
    "Apply the fix. Return only the corrected function."
)


def _http_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            payload = json.loads(raw) if raw else {"error": str(exc)}
        except json.JSONDecodeError:
            payload = {"error": raw or str(exc)}
        return exc.code, payload


def chat(
    base: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str,
    max_tokens: int,
    label: str,
) -> dict[str, Any]:
    url = f"{base.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()
    status, payload = _http_json(
        "POST",
        url,
        body={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    elapsed = time.perf_counter() - t0
    if status != 200 or not isinstance(payload, dict):
        return {
            "label": label,
            "ok": False,
            "error": f"HTTP {status}: {payload}",
            "elapsed_s": round(elapsed, 2),
        }
    try:
        content = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        return {
            "label": label,
            "ok": False,
            "error": f"bad payload: {exc}; {payload!r}"[:400],
            "elapsed_s": round(elapsed, 2),
        }
    usage = payload.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    ok = bool(str(content).strip()) and completion > 0
    tps = completion / elapsed if elapsed > 0 and completion else 0.0
    return {
        "label": label,
        "ok": ok,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion,
        "tok_s": round(tps, 2),
        "preview": str(content)[:300],
    }


def check_models(base: str, model: str, api_key: str) -> dict[str, Any]:
    status, payload = _http_json(
        "GET",
        f"{base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    if status != 200 or not isinstance(payload, dict):
        return {"label": "models", "ok": False, "error": f"HTTP {status}: {payload}"}
    ids = [m.get("id") for m in (payload.get("data") or []) if isinstance(m, dict)]
    ok = model in ids
    return {
        "label": "models",
        "ok": ok,
        "ids": ids,
        "error": None if ok else f"model {model!r} not in {ids}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8007/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--out", default="")
    parser.add_argument("--max-tokens-coding", type=int, default=120)
    parser.add_argument("--max-tokens-agentic", type=int, default=120)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    results.append(check_models(args.base, args.model, args.api_key))
    print(
        f"[models] ok={results[-1]['ok']} {results[-1].get('error') or results[-1].get('ids')}",
        flush=True,
    )

    smoke = chat(
        args.base,
        args.model,
        [{"role": "user", "content": "Reply with OK only."}],
        api_key=args.api_key,
        max_tokens=8,
        label="smoke",
    )
    results.append(smoke)
    print(
        f"[smoke] ok={smoke['ok']} {smoke.get('tok_s', 0)} tok/s "
        f"gen={smoke.get('completion_tokens', 0)} {smoke.get('error', '')}",
        flush=True,
    )

    coding = chat(
        args.base,
        args.model,
        [{"role": "user", "content": CODING_PROMPT}],
        api_key=args.api_key,
        max_tokens=args.max_tokens_coding,
        label="coding",
    )
    results.append(coding)
    print(
        f"[coding] ok={coding['ok']} {coding.get('tok_s', 0)} tok/s "
        f"gen={coding.get('completion_tokens', 0)} {coding.get('error', '')}",
        flush=True,
    )

    plan = chat(
        args.base,
        args.model,
        [
            {"role": "system", "content": AGENTIC_SYSTEM},
            {"role": "user", "content": AGENTIC_PLAN},
        ],
        api_key=args.api_key,
        max_tokens=args.max_tokens_agentic,
        label="agentic:plan",
    )
    results.append(plan)
    print(
        f"[agentic:plan] ok={plan['ok']} {plan.get('tok_s', 0)} tok/s "
        f"gen={plan.get('completion_tokens', 0)} {plan.get('error', '')}",
        flush=True,
    )

    edit = chat(
        args.base,
        args.model,
        [
            {"role": "system", "content": AGENTIC_SYSTEM},
            {"role": "user", "content": AGENTIC_PLAN},
            {"role": "assistant", "content": plan.get("preview") or "(empty)"},
            {"role": "user", "content": AGENTIC_EDIT},
        ],
        api_key=args.api_key,
        max_tokens=args.max_tokens_agentic,
        label="agentic:edit",
    )
    results.append(edit)
    print(
        f"[agentic:edit] ok={edit['ok']} {edit.get('tok_s', 0)} tok/s "
        f"gen={edit.get('completion_tokens', 0)} {edit.get('error', '')}",
        flush=True,
    )

    # Real Cursor-shaped multi-turn token test (always)
    token_script = Path(__file__).with_name("cursor_agent_token_test.py")
    token_out = (
        str(Path(args.out).with_name(Path(args.out).stem + "-cursor-agent.json"))
        if args.out
        else "/tmp/wire-cursor-agent-token.json"
    )
    token_cmd = [
        sys.executable,
        str(token_script),
        "--base",
        args.base,
        "--model",
        args.model,
        "--api-key",
        args.api_key,
        "--out",
        token_out,
    ]
    print("[cursor-agent-token] running…", flush=True)
    token_rc = subprocess.call(token_cmd)
    results.append(
        {
            "label": "cursor-agent-token",
            "ok": token_rc == 0,
            "out": token_out,
            "error": None if token_rc == 0 else f"exit {token_rc}",
        }
    )

    all_ok = all(bool(r.get("ok")) for r in results)
    summary = {
        "base": args.base,
        "model": args.model,
        "all_ok": all_ok,
        "results": results,
    }
    out = Path(args.out) if args.out else None
    if out:
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}", flush=True)

    print(f"validate {'PASS' if all_ok else 'FAIL'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
