# DDTree-MLX Benchmark Results & Findings

## Setup

- **Hardware**: Mac Studio M3 Ultra 256GB
- **Target model**: `mlx-community/Qwen3.5-27B-4bit` (hybrid: 48 GatedDeltaNet + 16 full attention)
- **Draft model**: `z-lab/Qwen3.5-27B-DFlash` (block diffusion drafter)
- **DDTree budget**: 4 (optimal for this model)
- **Max tokens**: 2048 unless noted

## Measured Performance: AR vs DFlash vs DDTree

Code generation prompt (binary search), 8K max tokens, end-to-end with visualization tool:

| Method | tok/s | vs Autoregressive | Acceptance |
|--------|------:|------------------:|-----------:|
| Autoregressive | 27.9 | 1.0x | — |
| DFlash | 38.6 | **1.38x** | 85% |
| **DFlash + DDTree** | **42.3** | **1.52x** | 4.2/cycle |

DDTree adds **~10-15% on top of DFlash** for code and structured content.

### Content-Type Sensitivity

| Content Type | Draft Acceptance | DDTree vs DFlash |
|-------------|------------------:|:-----------------|
| Code generation | 85%+ | **+10-15%** — tree catches occasional misses |
| Structured/factual | 70-80% | **+10-15%** — moderate room for tree alternatives |
| Creative prose | 5-10% | **~0%** — low acceptance means tree branches are just as wrong |

For creative writing and open-ended prose, DDTree roughly equals autoregressive speed because
the draft model's acceptance drops to 5-10%. When the draft model can't predict well, the
tree's backup branches are equally wrong, and the tree overhead eats any potential gain.

### Correction: Previous Estimates

Earlier versions of this file included estimated DDTree tok/s (73-95 tok/s, "2.6x over AR")
computed by multiplying DFlash long-context benchmark speeds by DDTree's speedup ratio. Those
estimates were never directly measured end-to-end and significantly overstated actual performance.
The numbers above are from real runs with all three methods on the same prompts.

## DDTree vs DFlash (Benchmark Script)

| Method | Avg tok/s | vs DFlash | Notes |
|--------|----------:|----------:|-------|
| DFlash (baseline) | 25.1 | 1.00x | Block diffusion speculative decoding |
| DDTree-4 (pre-conv-kernel) | 31.1 | 1.24x | Tree-aware commit + eval reduction |
| **DDTree-4 (with conv kernel)** | **~35** | **~1.39x** | + parent-aware conv Metal kernel (short probe) |

### Per-Prompt Breakdown

| Prompt | DFlash Accept | DFlash tok/s | DDTree-4 tok/s | Speedup |
|--------|-------------:|-------------:|---------------:|--------:|
| TCP/UDP explanation | 70% | 18.9 | 28.8 | **1.52x** |
| Binary search code | 86% | 39.3 | 37.5 | 0.95x |
| French Revolution | 68% | 17.6 | 27.8 | **1.58x** |

**Key insight**: DDTree excels when DFlash has moderate acceptance (68-70%), achieving 1.5-1.6x speedup. When DFlash already has high acceptance (86%), the tree overhead slightly exceeds the marginal acceptance gain.

### Long-Context Scaling (DDTree-8)

| Tokens | DFlash tok/s | DDTree-8 tok/s | Speedup |
|-------:|-------------:|---------------:|--------:|
| 1,024 | 13.3 | 18.4 | 1.38x |
| 2,048 | 14.0 | 18.1 | 1.29x |
| 4,096 | 13.9 | 18.6 | 1.34x |
| 8,192 | 14.7 | 19.2 | 1.31x |
| 16,384 | 13.0 | 12.7 | 0.98x |

DDTree maintains 1.29-1.38x through 8K tokens. At 16K, attention cost over the long prefix causes DDTree to break even.

## Phase Timing Breakdown

DDTree-4, prompt 1 (1117 tokens generated):

| Phase | Time | % Total |
|-------|-----:|--------:|
| Prefill | 223ms | 0.4% |
| Draft | 8,156ms | 13.2% |
| Tree Build | 887ms | 1.4% |
| **Tree Verify** | **51,920ms** | **84.3%** |
| Commit | 275ms | 0.4% |

### Within Tree Verify (profiled)

| Layer Type | Time | % of Verify |
|------------|-----:|-----------:|
| Linear (48 GatedDeltaNet) | 24,325ms | **73.2%** |
| Attention (16 full attention) | 7,269ms | 21.9% |

## Optimization History

### What Worked

1. **Custom Metal kernel for tree-aware GatedDelta** (PR #2)
   - Parent-indexed recurrence instead of sequential depth-group processing
   - Enables correct logits for ALL tree paths, not just DFS prefix
   - Combined with tree_aware_path_commit: installs per-node states directly
   - **Commit cost dropped 97%** (8,511ms -> 275ms)

2. **Removing unnecessary mx.eval() sync points**
   - Removed `mx.eval(draft_logits)` — `_build_tree_from_mlx_logits` evals only top-k
   - Deferred `mx.eval(committed_hidden)` — next cycle evals implicitly
   - Avoids full vocab-size tensor materialization and reduces GPU sync overhead

3. **Budget=4 as default**
   - 5 tree nodes is the sweet spot for this hybrid model
   - 86% fast-path rate, 3.2 tokens accepted per cycle
   - Higher budgets (8, 16, 32) increase verify cost faster than acceptance gains

4. **Parent-aware conv Metal kernel** (verify-fusion-controller branch)
   - Replaces the Python depth-group conv loop before the GatedDelta recurrence kernel
   - Short TCP/UDP probe, 128 generated tokens: DDTree-4 improved from 31.8 tok/s to 35.0 tok/s
   - Tree verify time dropped from 3,193ms to 2,833ms on that probe

### What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| **Attention-only tree verify** | 0.44x (worse) | LM head needs all 64 layers; 16-layer hidden states produce garbage logits. avg_accept drops to 1.0. |
| **Attention-only + keep MLP** | 0.34x (worse) | Even keeping feed-forward networks doesn't compensate for missing recurrent contributions. |
| **Chain tree shape** | 1.14x (worse than heap) | Linear chain has 100% fast path but lower acceptance diversity. |
| **Hybrid tree shape** | 0.98x (break-even) | Half chain + half root alternatives underperforms the heap algorithm. |
| **Root-wide tree shape** | 0.75x (worse) | All siblings at depth 1 gives only 2.0 acceptance — not enough depth. |
| **Split prefix/tree attention at 2K** | 1.24x (neutral) | Manual matmul + LSE combination is slower than MLX's optimized SDPA at 2K context. |
| **Exact prefix/tree attention synthetic 8K/16K** | 3-4x slower than SDPA | Correct within bf16 tolerance, but manual matmul/LSE loses to MLX SDPA for Qwen-like small-query shapes. Kept as opt-in only. |
| **Adaptive budget controller** | 1.25x (neutral) | Adds complexity, no measurable gain when eval reduction is already applied. |
| **Aggressive DFlash controller** | Worse on short prompts | A single lucky DFlash probe can switch too early. The implemented controller is opt-in and now requires sustained probe wins. |
| **Breadth bias (depth_penalty)** | ~1.24x (neutral) | Tree is too small at budget=4 for depth redistribution to matter. |
| **DFS-order mode** | 1.04x (worse than tree-aware) | DFS contamination of non-prefix paths reduces acceptance. |

## Architecture Constraints

The fundamental limitation is Qwen 3.5 27B's hybrid architecture:

- **48/64 layers are recurrent** (GatedDeltaNet) — must process each tree node sequentially
- **Per-node verify cost (~20ms) equals DFlash per-token cost (~21ms)** — no parallelism benefit for 75% of the model
- DDTree's advantage comes from **better acceptance density**: the tree concentrates budget on the most probable tokens, achieving higher acceptance per verified node
- On a **pure-attention model** (Llama, standard Qwen), DDTree would benefit much more since all layers could process tree nodes in parallel via the tree attention mask

## Key Metrics

- **Acceptance**: DDTree-4 accepts 3.2 tokens/cycle vs DFlash's 3.3 tokens/cycle — similar, but DDTree verifies only 5 nodes vs DFlash's 8 tokens
- **Fast-path rate**: 86% of cycles use tape rollback (cheap commit); 14% require suffix re-forward
- **Tree verify**: 84% of cycle time — the sole bottleneck
- **Commit**: Essentially free (0.4%) thanks to tree_aware_path_commit

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DDTREE_BUDGET` | `4` | Tree node budget (excluding root) |
| `DDTREE_EXACT_COMMIT` | `1` | Re-forward accepted tokens sequentially for lossless output |
| `DDTREE_TREE_AWARE_LINEAR` | `1` | Use parent-state forking for GatedDeltaNet |
| `DDTREE_TREE_KERNEL` | `1` | Use Metal kernel for tree-aware recurrence |
| `DDTREE_TREE_CONV_KERNEL` | `1` | Use Metal kernel for parent-aware causal conv |
| `DDTREE_EXACT_TREE_ATTENTION` | `0` | Opt-in exact prefix/tree attention; set to `auto` for long-context testing |
| `DDTREE_EXACT_TREE_ATTENTION_MIN_PREFIX` | `8192` | Prefix length for exact split attention in auto mode |
| `DDTREE_DFLASH_CONTROLLER` | `0` | Opt-in in-place DDTree/DFlash cycle controller |
| `DDTREE_CONTROLLER_MARGIN` | `1.20` | Required DFlash probe advantage before switching |
| `DDTREE_PROFILE_VERIFY` | `0` | Profile linear vs attention layer timing; use `detail` for per-op timings |
| `DDTREE_PROFILE_DETAIL` | `0` | Enable detailed synchronized verify timings |

## Files

| File | Description |
|------|-------------|
| `benchmarks_first_run.json` | Initial DDTree benchmarks (bonus double-count bug) |
| `benchmarks_pr1_merged.json` | After PR#1 fix, budgets 16/32/64 |
| `benchmarks_profile_small_budgets.json` | Profiled run, budgets 4/8/16 (linear vs attention breakdown) |
| `bench_pr2_final.json` | PR#2 Metal kernel, budget 4, cooled GPU |
| `bench_pr2_clean.json` | PR#2, budgets 4/8/16/24 |
