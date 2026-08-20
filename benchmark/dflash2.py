"""Qwen 3.8 DFlash2 loader.

`dflash-mlx` 0.1.0 only implements DFlash 1 (`DFlashDraftModelArgs` requires
top-level `rope_theta` and `block_size`). DFlash2 nests those fields and adds
dynamic conv + a candidate selector. Official MLX support is
`dflash.model_mlx.load_draft` (z-lab/dflash).

4-bit MLX targets (and 4-bit drafts) cannot verify blocks larger than 5
because of int4 matmul limits. Cap `block_size` and quantize the draft so
target (~16 GB) + draft (~4 GB) fits 32 GB UMA.

100k context **will jetsam a 32 GB Mac**: Qwen3.8 GQA KV is ~256 KiB/token
(fp16), so 100k ≈ 24 GB KV on top of ~20 GB weights. Default context is 8k.
"""
from __future__ import annotations

from typing import Any, Optional

MLX_INT4_MAX_BLOCK_SIZE = 5
DEFAULT_DFLASH2_DRAFT_BITS = 4
DEFAULT_MAX_CONTEXT = 8192
DEFAULT_MAX_NEW_TOKENS = 2048

# Qwen3.8-27B text: 64 layers, GQA 4 KV heads, head_dim 256, fp16 KV.
QWEN38_LAYERS = 64
QWEN38_KV_HEADS = 4
QWEN38_HEAD_DIM = 256
QWEN38_KV_BYTES_PER_ELEM = 2
DFLASH2_WEIGHTS_BYTES = 20 * 1024**3
OS_RESERVE_BYTES = 6 * 1024**3
UMA_32GB_BYTES = 32 * 1024**3


def kv_cache_bytes(
    seq_len: int,
    *,
    layers: int = QWEN38_LAYERS,
    kv_heads: int = QWEN38_KV_HEADS,
    head_dim: int = QWEN38_HEAD_DIM,
    bytes_per_elem: int = QWEN38_KV_BYTES_PER_ELEM,
) -> int:
    """Target KV size: 2 (K+V) × layers × kv_heads × head_dim × seq × dtype."""
    return 2 * layers * kv_heads * head_dim * max(0, int(seq_len)) * bytes_per_elem


def max_safe_context_tokens(
    *,
    uma_bytes: int = UMA_32GB_BYTES,
    weights_bytes: int = DFLASH2_WEIGHTS_BYTES,
    os_reserve_bytes: int = OS_RESERVE_BYTES,
) -> int:
    budget = uma_bytes - weights_bytes - os_reserve_bytes
    per_token = kv_cache_bytes(1)
    if budget <= 0 or per_token <= 0:
        return 0
    return budget // per_token


def require_context_fits(seq_len: int) -> None:
    """Raise if seq_len would exceed DEFAULT_MAX_CONTEXT or 32 GB KV budget."""
    n = int(seq_len)
    hardware_cap = max_safe_context_tokens()
    cap = min(DEFAULT_MAX_CONTEXT, hardware_cap)
    if n > cap:
        kv_gb = kv_cache_bytes(n) / (1024**3)
        raise ValueError(
            f"{n:,} tokens need ~{kv_gb:.1f} GB KV plus ~20 GB weights; "
            f"cap is {cap:,} on 32 GB UMA (100k context jetsams this machine)."
        )


def mlx_dflash2_block_size(config_block_size: int, *, quantized: bool = True) -> int:
    size = max(1, int(config_block_size))
    if quantized:
        return min(size, MLX_INT4_MAX_BLOCK_SIZE)
    return size


def is_dflash2_config(config: dict[str, Any]) -> bool:
    arches = config.get("architectures") or []
    return any("DFlash2" in str(a) for a in arches)


def is_dflash2_ref(draft_ref: Optional[str]) -> bool:
    if not draft_ref:
        return False
    if "DFlash2" in draft_ref:
        return True
    try:
        from huggingface_hub import snapshot_download
        from pathlib import Path
        import json

        path = snapshot_download(draft_ref, allow_patterns=["config.json"])
        cfg = json.loads((Path(path) / "config.json").read_text())
        return is_dflash2_config(cfg)
    except Exception:
        return False


def load_dflash2_draft(draft_ref: str) -> Any:
    """Load a DFlash2 draft checkpoint with the official MLX implementation."""
    from dflash.model_mlx import load_draft

    return load_draft(draft_ref)


def load_dflash2_mlx_runtime(
    model_ref: str,
    draft_ref: str,
    draft_bits: int | None = DEFAULT_DFLASH2_DRAFT_BITS,
):
    """Load target + DFlash2 draft for official `dflash.model_mlx.stream_generate`.

    `draft_bits=4` matches `dflash generate mlx --draft-bits 4`. Block size is
    capped for int4 matmul even when the checkpoint's `dflash_config.block_size`
    is larger (Qwen3.8 DFlash2 ships 8).
    """
    from dflash.benchmark import load_mlx_models

    model, draft, tokenizer = load_mlx_models(model_ref, draft_ref, draft_bits)
    block_size = mlx_dflash2_block_size(
        int(draft.config.block_size),
        quantized=True,
    )
    return model, tokenizer, draft, block_size


def build_filler_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Build a chat-formatted prompt with at least ``target_tokens`` tokens."""
    from benchmark._lib import apply_chat_prompt

    unit = "fill "
    lo, hi = 1, max(target_tokens * 4, 1024)
    best_prompt, best_count = "", 0
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = apply_chat_prompt(tokenizer, unit * mid)
        count = len(tokenizer.encode(prompt))
        if count >= target_tokens:
            best_prompt, best_count = prompt, count
            hi = mid - 1
        else:
            lo = mid + 1
    if not best_prompt:
        raise RuntimeError(
            f"Could not build prompt with >={target_tokens} tokens (max tried {best_count})"
        )
    return best_prompt, best_count
