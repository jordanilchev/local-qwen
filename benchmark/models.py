#!/usr/bin/env python3
"""Model registry for fair cross-backend benchmarks on Apple Silicon."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelFamily:
    """One logical model compared across inference stacks."""

    id: str
    label: str
    # MLX target (mlx-lm / vllm-mlx / DDTree)
    mlx_ref: Optional[str]
    mlx_quant: str
    draft_ref: Optional[str] = None
    # Ollama tag — prefer *-mlx on Apple Silicon when available
    ollama_ref: Optional[str] = None
    ollama_quant: str = "GGUF-Q4_K_M"
    # llama.cpp via llama-server -hf (same publisher/quant family when possible)
    llamacpp_hf: Optional[str] = None
    llamacpp_mtp_hf: Optional[str] = None
    gguf_quant: str = "GGUF-Q4_K_XL"
    notes: str = ""


# Order: MLX-heavy families first (MoE best case), then dense 3.8, then 3.6 dense.
MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        id="3.6-35b-moe",
        label="Qwen3.6-35B-MoE",
        mlx_ref="mlx-community/Qwen3.6-35B-A3B-4bit-DWQ",
        mlx_quant="MLX-int4-DWQ",
        draft_ref="z-lab/Qwen3.6-35B-A3B-DFlash",
        ollama_ref="qwen3.6:35b",
        ollama_quant="GGUF-Q4_K_M",
        notes="MoE; Ollama quant differs from MLX DWQ — decode comparison is approximate.",
    ),
    ModelFamily(
        id="3.8-27b",
        label="Qwen3.8-27B-dense",
        mlx_ref="mlx-community/Qwen3.8-27B-4bit",
        mlx_quant="MLX-int4",
        draft_ref="z-lab/Qwen3.8-27B-DFlash2",
        ollama_ref="qwen3.8:27b-q4_K_M",
        ollama_quant="GGUF-Q4_K_M",
        llamacpp_hf="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
        gguf_quant="GGUF-Q4_K_XL",
        notes="DFlash2 is MLX-only (dflash2-mlx / DDTree). Ollama uses full 27B Q4_K_M (not 27b-mlx, which is NVFP4).",
    ),
    ModelFamily(
        id="3.6-27b",
        label="Qwen3.6-27B-dense",
        mlx_ref="mlx-community/Qwen3.6-27B-4bit",
        mlx_quant="MLX-int4",
        draft_ref=None,
        ollama_ref="qwen3.6:27b",
        ollama_quant="GGUF-Q4_K_M",
        llamacpp_hf="unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
        llamacpp_mtp_hf="unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL",
        gguf_quant="GGUF-Q4_K_XL",
        notes="27B dense; llama.cpp uses Unsloth XL vs Ollama K_M.",
    ),
)

# Extra plain-MLX-only targets (no cross-backend row in session tables).
EXTRA_MLX_TARGETS: tuple[tuple[str, str], ...] = (
    ("Qwen3.6-27B-dense  [MLX-OptiQ-4bit]", "mlx-community/Qwen3.6-27B-OptiQ-4bit"),
)

_FAMILY_BY_ID = {f.id: f for f in MODEL_FAMILIES}


def get_family(family_id: str) -> ModelFamily:
    try:
        return _FAMILY_BY_ID[family_id]
    except KeyError as exc:
        known = ", ".join(sorted(_FAMILY_BY_ID))
        raise ValueError(f"Unknown BENCH_FAMILY {family_id!r}; known: {known}") from exc


def family_label(family: ModelFamily) -> str:
    return f"{family.label} [{family.mlx_quant}]"
