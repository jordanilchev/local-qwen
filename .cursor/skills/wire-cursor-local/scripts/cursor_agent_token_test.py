#!/usr/bin/env python3
"""Cursor-shaped agentic token test against an OpenAI-compatible server.

Simulates a real Chat/Agent turn sequence: system instructions, user task,
tool/file context, follow-up edit — reports per-turn and cumulative tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM = """You are Auto, a coding agent in Cursor.
Use tools conceptually: read files, then propose a minimal patch.
Be concise. Prefer unified diffs or full corrected functions.
Do not invent files that were not provided."""


def _post_chat(
    base: str,
    *,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    url = f"{base.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw[:500]}") from exc
    elapsed = time.perf_counter() - t0
    content = payload["choices"][0]["message"]["content"] or ""
    usage = payload.get("usage") or {}
    return {
        "content": content,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(
            (int(usage.get("completion_tokens") or 0) / elapsed) if elapsed > 0 else 0.0,
            2,
        ),
    }


def build_repo_context(filler_chars: int) -> str:
    """Synthetic open-file context like Cursor attaches (deterministic)."""
    unit = (
        "# wire_cursor_token_fixture.py\n"
        "def helper(n: int) -> int:\n"
        "    return n * 2\n\n"
    )
    out = []
    while sum(len(x) for x in out) < filler_chars:
        out.append(unit)
    text = "".join(out)
    return text[:filler_chars]


def run(base: str, model: str, api_key: str, context_chars: int) -> dict[str, Any]:
    bug_file = (
        "```python\n"
        "# scripts/parse_cli.py\n"
        "import sys\n\n"
        "def parse_args(argv=None):\n"
        "    argv = argv if argv is not None else sys.argv\n"
        "    return argv[1]  # IndexError when argv == [] or len==1\n"
        "```\n"
    )
    repo_blob = build_repo_context(context_chars)

    turns: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Real Cursor-style task: `parse_args` crashes on empty argv.\n"
                "Open files in the editor (context):\n\n"
                f"{repo_blob}\n\n"
                f"Focus file:\n{bug_file}\n"
                "1) Restate the bug in one sentence.\n"
                "2) Say which tool you'd call next (read/search).\n"
                "Keep under 120 words."
            ),
        },
    ]

    r1 = _post_chat(base, model=model, api_key=api_key, messages=messages, max_tokens=160)
    turns.append({"turn": "plan", **{k: v for k, v in r1.items() if k != "content"}, "preview": r1["content"][:240]})
    messages.append({"role": "assistant", "content": r1["content"]})

    # Tool result turn (what Cursor injects after a Read)
    messages.append(
        {
            "role": "user",
            "content": (
                "<tool_result name=\"Read\" path=\"scripts/parse_cli.py\">\n"
                f"{bug_file}\n"
                "</tool_result>\n"
                "Apply the fix. Return only the corrected `parse_args` function."
            ),
        }
    )
    r2 = _post_chat(base, model=model, api_key=api_key, messages=messages, max_tokens=200)
    turns.append({"turn": "edit", **{k: v for k, v in r2.items() if k != "content"}, "preview": r2["content"][:240]})
    messages.append({"role": "assistant", "content": r2["content"]})

    # Follow-up like a subagent continuation
    messages.append(
        {
            "role": "user",
            "content": (
                "Subagent follow-up: add a one-line docstring and a tiny "
                "`if __name__ == '__main__'` guard that prints the first arg or '(none)'. "
                "Return the full short module."
            ),
        }
    )
    r3 = _post_chat(base, model=model, api_key=api_key, messages=messages, max_tokens=280)
    turns.append({"turn": "subagent_followup", **{k: v for k, v in r3.items() if k != "content"}, "preview": r3["content"][:240]})

    ok = all(
        t["prompt_tokens"] > 0 and t["completion_tokens"] > 0 for t in turns
    ) and ("argv" in r2["content"].lower() or "len(" in r2["content"])

    # Prompt tokens should grow across turns (Cursor context accumulation)
    growth_ok = turns[1]["prompt_tokens"] > turns[0]["prompt_tokens"] and turns[2]["prompt_tokens"] > turns[1]["prompt_tokens"]

    return {
        "ok": bool(ok and growth_ok),
        "growth_ok": growth_ok,
        "context_chars": context_chars,
        "turns": turns,
        "prompt_tokens_series": [t["prompt_tokens"] for t in turns],
        "completion_tokens_series": [t["completion_tokens"] for t in turns],
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in turns),
        "total_completion_tokens": sum(t["completion_tokens"] for t in turns),
        "edit_preview": r2["content"][:400],
        "followup_preview": r3["content"][:400],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="local")
    p.add_argument(
        "--context-chars",
        type=int,
        default=6000,
        help="Synthetic open-editor context size (Cursor-like attachment)",
    )
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(
        f"cursor-agent token test base={args.base} model={args.model} "
        f"context_chars={args.context_chars}",
        flush=True,
    )
    try:
        summary = run(args.base, args.model, args.api_key, args.context_chars)
    except Exception as exc:  # noqa: BLE001 — surface to CLI
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for t in summary["turns"]:
        print(
            f"[{t['turn']}] prompt={t['prompt_tokens']} "
            f"completion={t['completion_tokens']} "
            f"{t['tok_s']} tok/s {t['elapsed_s']}s",
            flush=True,
        )
    print(
        f"series prompt={summary['prompt_tokens_series']} "
        f"growth_ok={summary['growth_ok']} ok={summary['ok']}",
        flush=True,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}", flush=True)

    print(f"cursor-agent-token {'PASS' if summary['ok'] else 'FAIL'}", flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
