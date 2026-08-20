#!/usr/bin/env python3
"""
OpenAI-compatible chat server: Qwen3.8-27B-4bit + official DFlash2.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/dflash2_server.py
    HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/dflash2_server.py --port 8007

Cursor: Settings → Models → Override OpenAI Base URL → http(s)://…/v1
        Add custom model id: qwen3.8-27b-dflash2

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (stream and non-stream)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from benchmark._lib import CHAT_TEMPLATE_KWARGS, set_benchmark_seed
from benchmark.dflash2 import (
    DEFAULT_DFLASH2_DRAFT_BITS,
    DEFAULT_MAX_CONTEXT,
    DEFAULT_MAX_NEW_TOKENS,
    load_dflash2_mlx_runtime,
    require_context_fits,
)

DEFAULT_MODEL_ID = "qwen3.8-27b-dflash2"
DEFAULT_TARGET = "mlx-community/Qwen3.8-27B-4bit"
DEFAULT_DRAFT = "z-lab/Qwen3.8-27B-DFlash2"

app = FastAPI(title="DFlash2 MLX Server")
STATE: dict[str, Any] = {}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content) if content is not None else ""


def _messages_to_prompt(tokenizer, messages: list[dict[str, Any]]) -> str:
    normalized = [
        {"role": m.get("role", "user"), "content": _message_text(m.get("content", ""))}
        for m in messages
    ]
    return tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )


def _chat_completion_payload(
    *,
    response_id: str,
    created: int,
    model_id: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if extra:
        body["dflash2_stats"] = extra
    return body


def _stream_chunk(
    *,
    response_id: str,
    created: int,
    model_id: str,
    delta_content: str | None = None,
    finish_reason: str | None = None,
) -> str:
    delta: dict[str, Any] = {}
    if delta_content:
        delta["content"] = delta_content
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "dflash2-mlx",
        "model": STATE.get("model_id"),
        "target": STATE.get("target_ref"),
        "draft": STATE.get("draft_ref"),
        "block_size": STATE.get("block_size"),
        "max_context": STATE.get("max_context"),
        "default_max_tokens": STATE.get("default_max_tokens"),
    }


@app.get("/v1/models")
async def models():
    model_id = STATE["model_id"]
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "dflash2-mlx",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    from dflash.model_mlx import stream_generate

    payload = await request.json()
    messages = payload.get("messages", [])
    max_context = int(STATE["max_context"])
    default_max_tokens = int(STATE["default_max_tokens"])
    max_tokens = int(payload.get("max_tokens", default_max_tokens))
    stream = bool(payload.get("stream", False))
    temperature = float(payload.get("temperature", 0.0))

    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    draft = STATE["draft"]
    block_size = STATE["block_size"]
    model_id = STATE["model_id"]

    prompt_str = _messages_to_prompt(tokenizer, messages)
    prompt_tokens = len(tokenizer.encode(prompt_str))
    try:
        require_context_fits(prompt_tokens)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if prompt_tokens > max_context:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt is {prompt_tokens:,} tokens; max_context is {max_context:,}. "
                "Shorten the conversation or raise MAX_CONTEXT."
            ),
        )
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    set_benchmark_seed()

    if not stream:
        text_parts: list[str] = []
        completion_tokens = 0
        accepted_chunks: list[float] = []
        for response in stream_generate(
            model,
            draft,
            tokenizer,
            prompt_str,
            block_size=block_size,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if response.text:
                text_parts.append(response.text)
            if response.tokens:
                completion_tokens += len(response.tokens)
                if response.accepted is not None:
                    accepted_chunks.append(float(response.accepted))
        content = "".join(text_parts)
        extra = {
            "block_size": block_size,
            "avg_acceptance": (
                sum(accepted_chunks) / len(accepted_chunks)
                if accepted_chunks
                else None
            ),
        }
        return _chat_completion_payload(
            response_id=response_id,
            created=created,
            model_id=model_id,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            extra=extra,
        )

    def event_stream():
        completion_tokens = 0
        accepted_chunks: list[float] = []
        for response in stream_generate(
            model,
            draft,
            tokenizer,
            prompt_str,
            block_size=block_size,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if response.text:
                yield _stream_chunk(
                    response_id=response_id,
                    created=created,
                    model_id=model_id,
                    delta_content=response.text,
                )
            if response.tokens:
                completion_tokens += len(response.tokens)
                if response.accepted is not None:
                    accepted_chunks.append(float(response.accepted))
        yield _stream_chunk(
            response_id=response_id,
            created=created,
            model_id=model_id,
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash2 OpenAI-compatible server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--draft-bits", type=int, default=DEFAULT_DFLASH2_DRAFT_BITS)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--max-context",
        type=int,
        default=int(os.environ.get("MAX_CONTEXT", DEFAULT_MAX_CONTEXT)),
    )
    parser.add_argument(
        "--default-max-tokens",
        type=int,
        default=int(os.environ.get("DEFAULT_MAX_TOKENS", DEFAULT_MAX_NEW_TOKENS)),
    )
    args = parser.parse_args()

    print(f"Loading {args.target} + DFlash2 draft {args.draft}...", flush=True)
    model, tokenizer, draft, block_size = load_dflash2_mlx_runtime(
        args.target, args.draft, args.draft_bits
    )
    if args.block_size is not None:
        block_size = max(1, int(args.block_size))

    STATE.update(
        {
            "model_id": args.model_id,
            "target_ref": args.target,
            "draft_ref": args.draft,
            "model": model,
            "tokenizer": tokenizer,
            "draft": draft,
            "block_size": block_size,
            "max_context": max(
                1, min(int(args.max_context), DEFAULT_MAX_CONTEXT)
            ),
            "default_max_tokens": max(1, int(args.default_max_tokens)),
        }
    )

    print(f"DFlash2 server ready: {args.model_id}", flush=True)
    print(f"  Target: {args.target}", flush=True)
    print(f"  Draft:  {args.draft} ({args.draft_bits}-bit)", flush=True)
    print(f"  block_size: {block_size}", flush=True)
    print(f"  max_context: {STATE['max_context']:,}", flush=True)
    print(f"  default_max_tokens: {STATE['default_max_tokens']:,}", flush=True)
    print(f"  Models:  http://{args.host}:{args.port}/v1/models", flush=True)
    print(f"  Chat:    http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
