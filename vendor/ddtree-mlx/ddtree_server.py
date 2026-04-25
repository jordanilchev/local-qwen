"""
FastAPI server for DDTree-MLX — OpenAI-compatible endpoint.

Usage:
    python3.12 ddtree_server.py [--port 8006] [--model mlx-community/Qwen3.5-27B-4bit] [--tree-budget 4]
"""

import argparse
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request

from dflash_mlx.generate import (
    DRAFT_REGISTRY,
    decode_token,
    get_stop_token_ids,
    load_runtime_components,
)
from ddtree_mlx.runtime import generate_ddtree_once

app = FastAPI(title="DDTree MLX Server")

STATE = {}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": STATE.get("model_ref"),
        "engine": "ddtree-mlx",
        "tree_budget": STATE.get("tree_budget"),
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": STATE["model_ref"],
                "object": "model",
                "owned_by": "ddtree-mlx",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    tokenizer = STATE["tokenizer"]
    target_model = STATE["target_model"]
    draft_model = STATE["draft_model"]
    model_ref = STATE["model_ref"]

    messages = payload.get("messages", [])
    chat_template_kwargs = payload.get(
        "chat_template_args", {"enable_thinking": False}
    )

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_tokens = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        )
    else:
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            parts.append(f"{m.get('role', 'user')}: {content}")
        prompt = "\n\n".join(parts)
        prompt_tokens = list(tokenizer.encode(prompt))

    max_tokens = int(payload.get("max_tokens", 2048))
    tree_budget = int(payload.get("tree_budget", STATE.get("tree_budget", 4)))
    stop_token_ids = get_stop_token_ids(tokenizer)
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    summary = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_tokens,
        tree_budget=tree_budget,
        stop_token_ids=stop_token_ids,
    )

    token_ids = summary.get("generated_token_ids", [])
    text = tokenizer.decode(token_ids) if token_ids else ""
    gen_tok = int(summary.get("generation_tokens", 0))

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_ref,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": gen_tok,
            "total_tokens": len(prompt_tokens) + gen_tok,
        },
        "ddtree_stats": {
            "tree_budget": tree_budget,
            "cycles": summary.get("cycles_completed", 0),
            "avg_acceptance": summary.get("avg_acceptance", 0),
            "fast_path_ratio": summary.get("fast_path_ratio", 0),
            "tok_per_sec": summary.get("tokens_per_second", 0),
            "tree_aware_linear": summary.get("tree_aware_linear", False),
            "tree_aware_commit_count": summary.get("tree_aware_commit_count", 0),
            "ddtree_cycles": summary.get("ddtree_cycles_completed", 0),
            "dflash_cycles": summary.get("dflash_cycles_completed", 0),
            "dflash_controller_enabled": summary.get("dflash_controller_enabled", False),
            "dflash_controller_mode": summary.get("dflash_controller_mode", "ddtree"),
            "dflash_controller_switch_count": summary.get(
                "dflash_controller_switch_count", 0
            ),
            "dflash_controller_min_probes": summary.get(
                "dflash_controller_min_probes", 0
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--draft", default=None)
    parser.add_argument("--tree-budget", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    target_model, tokenizer, draft_model, draft_ref = load_runtime_components(
        model_ref=args.model, draft_ref=args.draft,
    )
    STATE.update({
        "model_ref": args.model,
        "target_model": target_model,
        "tokenizer": tokenizer,
        "draft_model": draft_model,
        "tree_budget": args.tree_budget,
    })
    print(
        f"DDTree server ready: {args.model} + {draft_ref or DRAFT_REGISTRY.get(args.model, 'no draft')}"
    )
    print(f"  Tree budget: {args.tree_budget}")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port)
