"""
Benchmark DDTree-MLX vs DFlash baseline.

Usage:
    python3.12 benchmark.py [--max-tokens 4096] [--budgets 4]
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("DFLASH_DRAFT_SINK", "64")
os.environ.setdefault("DFLASH_DRAFT_WINDOW", "1024")

import mlx.core as mx

from dflash_mlx.generate import (
    get_stop_token_ids,
    load_runtime_components,
)
from dflash_mlx.runtime import generate_dflash_once
from ddtree_mlx.runtime import generate_ddtree_once


PROMPTS = [
    "Explain the key differences between TCP and UDP protocols, including when you would use each one.",
    "Write a Python function that implements binary search on a sorted array. Include edge cases.",
    "What are the main causes of the French Revolution? Provide a detailed analysis.",
    "Describe the process of photosynthesis in detail, including the light and dark reactions.",
    "Write a comprehensive comparison of REST and GraphQL APIs, with pros and cons of each.",
]


def run_dflash(target_model, draft_model, tokenizer, prompt_tokens, max_tokens, stop_ids):
    return generate_dflash_once(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt="",
        max_new_tokens=max_tokens,
        use_chat_template=False,
        stop_token_ids=stop_ids,
        prompt_tokens_override=prompt_tokens,
    )


def run_ddtree(target_model, draft_model, tokenizer, prompt_tokens, max_tokens, stop_ids, budget):
    return generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_tokens,
        tree_budget=budget,
        stop_token_ids=stop_ids,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--draft", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--budgets", default="4")
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts to test")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    max_tokens = args.max_tokens
    n_prompts = min(args.prompts, len(PROMPTS))

    print(f"Loading {args.model}...")
    target_model, tokenizer, draft_model, draft_ref = load_runtime_components(
        model_ref=args.model, draft_ref=args.draft,
    )
    stop_ids = get_stop_token_ids(tokenizer)
    print(f"Model loaded: {args.model} + {draft_ref}")
    print(f"Max tokens: {max_tokens}, Budgets: {budgets}, Prompts: {n_prompts}")
    print()

    results = []

    for prompt_idx in range(n_prompts):
        prompt_text = PROMPTS[prompt_idx]
        prompt_tokens = list(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ))

        print(f"=== Prompt {prompt_idx + 1}/{n_prompts}: {prompt_text[:60]}... ===")
        print(f"  Prompt tokens: {len(prompt_tokens)}")

        # --- DFlash baseline ---
        print(f"  Running DFlash...")
        dflash_result = run_dflash(
            target_model, draft_model, tokenizer,
            prompt_tokens, max_tokens, stop_ids,
        )
        dflash_tps = dflash_result.get("generation_tokens", 0) / (dflash_result.get("elapsed_us", 1) / 1e6)
        dflash_tokens = dflash_result.get("generation_tokens", 0)
        print(f"    DFlash: {dflash_tps:.1f} tok/s, {dflash_tokens} tokens, "
              f"acceptance={dflash_result.get('acceptance_ratio', 0):.2f}")

        results.append({
            "prompt_idx": prompt_idx,
            "method": "dflash",
            "budget": None,
            "tokens": dflash_tokens,
            "tok_per_sec": round(dflash_tps, 1),
            "elapsed_ms": round(dflash_result.get("elapsed_us", 0) / 1000, 1),
            "cycles": dflash_result.get("cycles_completed", 0),
            "acceptance_ratio": dflash_result.get("acceptance_ratio", 0),
        })

        # --- DDTree at each budget ---
        for budget in budgets:
            print(f"  Running DDTree (budget={budget})...")
            try:
                ddtree_result = run_ddtree(
                    target_model, draft_model, tokenizer,
                    prompt_tokens, max_tokens, stop_ids, budget,
                )
                ddtree_tps = ddtree_result.get("tokens_per_second", 0)
                ddtree_tokens = ddtree_result.get("generation_tokens", 0)
                fast_ratio = ddtree_result.get("fast_path_ratio", 0)
                avg_accept = ddtree_result.get("avg_acceptance", 0)
                speedup = ddtree_tps / dflash_tps if dflash_tps > 0 else 0

                print(f"    DDTree-{budget}: {ddtree_tps:.1f} tok/s ({speedup:.2f}x vs DFlash), "
                      f"{ddtree_tokens} tokens, avg_accept={avg_accept:.1f}, "
                      f"fast_path={fast_ratio:.0%}")

                results.append({
                    "prompt_idx": prompt_idx,
                    "method": f"ddtree-{budget}",
                    "budget": budget,
                    "tokens": ddtree_tokens,
                    "tok_per_sec": round(ddtree_tps, 1),
                    "elapsed_ms": round(ddtree_result.get("elapsed_us", 0) / 1000, 1),
                    "cycles": ddtree_result.get("cycles_completed", 0),
                    "avg_acceptance": round(avg_accept, 2),
                    "fast_path_ratio": round(fast_ratio, 3),
                    "speedup_vs_dflash": round(speedup, 3),
                    "tree_aware_linear": ddtree_result.get("tree_aware_linear", False),
                    "tree_aware_commit_count": ddtree_result.get(
                        "tree_aware_commit_count", 0
                    ),
                    "ddtree_cycles_completed": ddtree_result.get(
                        "ddtree_cycles_completed", 0
                    ),
                    "dflash_cycles_completed": ddtree_result.get(
                        "dflash_cycles_completed", 0
                    ),
                    "dflash_controller_enabled": ddtree_result.get(
                        "dflash_controller_enabled", False
                    ),
                    "dflash_controller_mode": ddtree_result.get(
                        "dflash_controller_mode", "ddtree"
                    ),
                    "dflash_controller_probe_count": ddtree_result.get(
                        "dflash_controller_probe_count", 0
                    ),
                    "dflash_controller_switch_count": ddtree_result.get(
                        "dflash_controller_switch_count", 0
                    ),
                    "dflash_controller_min_probes": ddtree_result.get(
                        "dflash_controller_min_probes", 0
                    ),
                    "ddtree_cycle_tps_avg": ddtree_result.get(
                        "ddtree_cycle_tps_avg", 0
                    ),
                    "dflash_cycle_tps_avg": ddtree_result.get(
                        "dflash_cycle_tps_avg", 0
                    ),
                    "phase_timings_us": ddtree_result.get("phase_timings_us", {}),
                })
            except Exception as e:
                print(f"    DDTree-{budget}: FAILED — {e}")
                results.append({
                    "prompt_idx": prompt_idx,
                    "method": f"ddtree-{budget}",
                    "budget": budget,
                    "error": str(e),
                })

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    dflash_avg = sum(r["tok_per_sec"] for r in results if r["method"] == "dflash") / n_prompts
    print(f"  DFlash avg: {dflash_avg:.1f} tok/s")
    for budget in budgets:
        method = f"ddtree-{budget}"
        ddtree_runs = [r for r in results if r["method"] == method and "tok_per_sec" in r]
        if ddtree_runs:
            avg_tps = sum(r["tok_per_sec"] for r in ddtree_runs) / len(ddtree_runs)
            avg_speedup = avg_tps / dflash_avg if dflash_avg > 0 else 0
            avg_fast = sum(r.get("fast_path_ratio", 0) for r in ddtree_runs) / len(ddtree_runs)
            print(f"  DDTree-{budget} avg: {avg_tps:.1f} tok/s ({avg_speedup:.2f}x), fast_path={avg_fast:.0%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
