"""DFlash2 (Qwen 3.8) loader — dflash-mlx DFlash1 cannot parse this checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

from benchmark.dflash2 import (
    DEFAULT_DFLASH2_DRAFT_BITS,
    DEFAULT_MAX_CONTEXT,
    MLX_INT4_MAX_BLOCK_SIZE,
    is_dflash2_config,
    kv_cache_bytes,
    load_dflash2_draft,
    mlx_dflash2_block_size,
    require_context_fits,
)
from ddtree_mlx.runtime import (
    _draft_block_size,
    _draft_layer_ids,
    _draft_mask_id,
    _is_dflash2_draft,
)

DFLASH2_REF = "z-lab/Qwen3.8-27B-DFlash2"
_HUB = Path.home() / "Models/HuggingFace/hub/models--z-lab--Qwen3.8-27B-DFlash2"
_CONFIGS = sorted(_HUB.glob("snapshots/*/config.json"))
CONFIG_PATH = _CONFIGS[-1] if _CONFIGS else Path("/nonexistent")


def test_qwen38_100k_kv_exceeds_32gb_uma():
    """Fanless 32GB crash: 100k tokens of GQA KV plus ~20GB weights do not fit."""
    kv = kv_cache_bytes(100_000)
    weights = 20 * 1024**3
    assert kv > 20 * 1024**3
    assert kv + weights > 32 * 1024**3


def test_default_max_context_fits_32gb_with_dflash2_weights():
    assert DEFAULT_MAX_CONTEXT == 32768
    kv = kv_cache_bytes(DEFAULT_MAX_CONTEXT)
    assert kv + 20 * 1024**3 < 32 * 1024**3
    # ~4 GB left for OS / Cursor / browser under the UMA budget.
    assert 32 * 1024**3 - kv - 20 * 1024**3 >= 4 * 1024**3


def test_require_context_fits_rejects_100k():
    import pytest

    with pytest.raises(ValueError, match="100"):
        require_context_fits(100_000)
    require_context_fits(DEFAULT_MAX_CONTEXT)


def test_mlx_int4_caps_dflash2_block_size_at_5():
    """Quantized MLX int4 matmul cannot verify blocks larger than 5."""
    assert MLX_INT4_MAX_BLOCK_SIZE == 5
    assert DEFAULT_DFLASH2_DRAFT_BITS == 4
    assert mlx_dflash2_block_size(8, quantized=True) == 5
    assert mlx_dflash2_block_size(4, quantized=True) == 4
    assert mlx_dflash2_block_size(8, quantized=False) == 8


def test_qwen38_dflash2_config_is_detected():
    cfg = json.loads(CONFIG_PATH.read_text())
    assert is_dflash2_config(cfg) is True
    assert "block_size" not in cfg
    assert cfg["dflash_config"]["block_size"] == 8
    assert "rope_theta" not in cfg
    assert cfg["rope_parameters"]["rope_theta"] == 10000000


def test_dflash1_config_is_not_dflash2():
    assert is_dflash2_config({"architectures": ["DFlashDraftModel"]}) is False
    assert is_dflash2_config({}) is False


def test_dflash2_runtime_helpers():
    class _Cfg:
        target_layer_ids = (5, 19, 33)
        block_size = 8
        mask_token_id = 248070

    class _Draft:
        config = _Cfg()

        def hidden_states(self, *args, **kwargs):
            return None

        def make_cache(self):
            return []

    draft = _Draft()
    assert _is_dflash2_draft(draft) is True
    assert _draft_layer_ids(draft) == [5, 19, 33]
    assert _draft_block_size(draft) == 8
    assert _draft_mask_id(draft) == 248070


def test_dflash1_runtime_helpers_use_model_attrs():
    class _Draft:
        target_layer_ids = [1, 16, 31]
        block_size = 16
        mask_token_id = 7

    draft = _Draft()
    assert _is_dflash2_draft(draft) is False
    assert _draft_layer_ids(draft) == [1, 16, 31]
    assert _draft_block_size(draft) == 16
    assert _draft_mask_id(draft) == 7


def test_load_dflash2_draft_weights():
    draft = load_dflash2_draft(DFLASH2_REF)
    assert type(draft).__name__ == "DFlash2DraftModel"
    assert int(draft.config.block_size) == 8
    assert int(draft.config.mask_token_id) == 248070
    assert tuple(draft.config.target_layer_ids) == (5, 19, 33, 47, 61)
    assert hasattr(draft, "hidden_states")
    assert hasattr(draft, "make_cache")
    assert hasattr(draft, "candidate_selector")
    caches = draft.make_cache()
    assert len(caches) == len(draft.layers) == 5
