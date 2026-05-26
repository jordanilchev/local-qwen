"""Unit tests for benchmark/_lib.py timing and schema helpers."""
import pytest

from benchmark._lib import (
    CODING_PROMPTS,
    DEFAULT_SAMPLING,
    PROMPT_SET,
    RunMetrics,
    decode_tps_from_wallclock,
    make_results_payload,
    runs_to_result_dict,
    validate_results_payload,
)


def test_decode_tps_from_wallclock_basic():
    # 101 tokens, 100 ms TTFT, 1100 ms total -> 1000 ms decode for 100 tokens
    tps = decode_tps_from_wallclock(101, ttft_ms=100.0, total_ms=1100.0)
    assert tps == pytest.approx(100.0)


def test_decode_tps_single_token():
    assert decode_tps_from_wallclock(1, ttft_ms=50.0, total_ms=200.0) == 0.0


def test_runs_to_result_dict_medians():
    runs = [
        RunMetrics(100.0, 10.0, 50, 600.0),
        RunMetrics(200.0, 20.0, 60, 800.0),
        RunMetrics(150.0, 15.0, 55, 700.0),
    ]
    out = runs_to_result_dict(runs)
    assert out["ttft_ms_median"] == 150.0
    assert out["decode_tps_median"] == 15.0
    assert out["completion_tokens_median"] == 55
    assert len(out["ttft_ms_runs"]) == 3


def test_resolve_draft_ref_qwen36_moe():
    from benchmark._lib import resolve_draft_ref

    ref = resolve_draft_ref("mlx-community/Qwen3.6-35B-A3B-4bit-DWQ")
    assert ref == "z-lab/Qwen3.6-35B-A3B-DFlash"


def test_load_ddtree_runtime_requires_drafter():
    from benchmark._lib import load_ddtree_runtime

    target, tok, draft, loaded_ref, stop = load_ddtree_runtime(
        "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ"
    )
    assert draft is not None
    assert loaded_ref == "z-lab/Qwen3.6-35B-A3B-DFlash"
    assert len(stop) > 0


def test_validate_results_payload_accepts_minimal():
    payload = make_results_payload(
        ts="20260524T120000Z",
        method="plain-mlx",
        model_label="test",
        model_ref="org/model",
        results=[
            {
                "prompt": CODING_PROMPTS[0][0],
                **runs_to_result_dict([
                    RunMetrics(10.0, 5.0, 20, 100.0),
                ]),
                "avg_acceptance": None,
            }
        ],
        prompt_set=PROMPT_SET,
        sampling=dict(DEFAULT_SAMPLING),
        warmups=1,
        runs_per_prompt=1,
    )
    validate_results_payload(payload)


def test_validate_results_payload_rejects_incomplete_prompt_entry():
    payload = make_results_payload(
        ts="20260524T120000Z",
        method="plain-mlx",
        model_label="test",
        model_ref="org/model",
        results=[
            {
                "prompt": "code-algo",
                **runs_to_result_dict([RunMetrics(10.0, 5.0, 20, 100.0)]),
            }
        ],
    )
    with pytest.raises(ValueError, match="completion_tokens_runs"):
        validate_results_payload({
            **payload,
            "results": [{"prompt": "code-algo", "ttft_ms_runs": [1], "decode_tps_runs": [2]}],
        })
