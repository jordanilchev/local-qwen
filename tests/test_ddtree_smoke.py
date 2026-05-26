import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

import pytest

from benchmark._lib import apply_chat_prompt_tokens

TARGET = "mlx-community/Qwen3.5-27B-4bit"
DRAFTER = "z-lab/Qwen3.5-27B-DFlash"


def test_imports():
    from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
    from ddtree_mlx.runtime import generate_ddtree_once


def test_short_inference():
    from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
    from ddtree_mlx.runtime import generate_ddtree_once

    target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET, draft_ref=None)

    prompt_tokens = apply_chat_prompt_tokens(tokenizer, "Say hello in one word.")

    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=32,
        tree_budget=4,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )

    assert len(result["generated_token_ids"]) > 0
    assert result["tokens_per_second"] > 0.0
    text = tokenizer.decode(result["generated_token_ids"])
    assert len(text.strip()) > 0


def test_result_keys():
    from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
    from ddtree_mlx.runtime import generate_ddtree_once

    target_model, tokenizer, draft_model, _ = load_runtime_components(model_ref=TARGET, draft_ref=None)
    prompt_tokens = apply_chat_prompt_tokens(tokenizer, "1+1=")
    result = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=8,
        tree_budget=4,
        stop_token_ids=get_stop_token_ids(tokenizer),
    )
    assert "generated_token_ids" in result
    assert "tokens_per_second" in result
