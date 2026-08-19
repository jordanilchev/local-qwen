"""Qwen 3.8 DFlash2 loader.

`dflash-mlx` 0.1.0 only implements DFlash 1 (`DFlashDraftModelArgs` requires
top-level `rope_theta` and `block_size`). DFlash2 nests those fields and adds
dynamic conv + a candidate selector. Official MLX support is
`dflash.model_mlx.load_draft` (z-lab/dflash).

4-bit MLX targets (and 4-bit drafts) cannot verify blocks larger than 5
because of int4 matmul limits. Cap `block_size` and quantize the draft so
target (~16 GB) + draft (~4 GB) fits 32 GB UMA.
"""
from __future__ import annotations

from typing import Any, Optional

MLX_INT4_MAX_BLOCK_SIZE = 5
DEFAULT_DFLASH2_DRAFT_BITS = 4


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
