#!/usr/bin/env python3
"""Single-turn interactive inference with Qwen3.5-27B-4bit + DDTree."""
import os, sys
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once

from benchmark._lib import apply_chat_prompt_tokens

TARGET = "mlx-community/Qwen3.5-27B-4bit"
MAX_NEW_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
TREE_BUDGET = int(os.environ.get("TREE_BUDGET", "3"))


def run(prompt: str) -> None:
    target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET, draft_ref=None)

    prompt_tokens = apply_chat_prompt_tokens(tokenizer, prompt)

    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        tree_budget=TREE_BUDGET,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )

    print(tokenizer.decode(result["generated_token_ids"]))
    print(f"\n→ {result['tokens_per_second']:.1f} tok/s", file=sys.stderr)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ")
    run(prompt)
