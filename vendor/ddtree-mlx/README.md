# DDTree-MLX

**Tree-based speculative decoding for Apple Silicon.** ~10-15% faster than DFlash, ~1.5x faster than autoregressive on Qwen 3.5 27B.

DDTree extends [DFlash](https://github.com/bstnxbt/dflash-mlx) speculative decoding by building a **draft tree** from per-position logits and verifying the entire tree in one forward pass. Instead of betting on a single draft sequence, DDTree explores multiple likely continuations simultaneously, accepting more tokens per verification cycle.

Based on the paper [*Accelerating Speculative Decoding with Block Diffusion Draft Trees*](https://liranringel.github.io/ddtree/DDTree.pdf) by Liran Ringel & Yaniv Romano. This is the first MLX port for Apple Silicon, with custom Metal kernels for hybrid model support.

## Performance

Measured on Mac Studio M3 Ultra 256GB, Qwen 3.5 27B 4-bit, code generation prompt at 8K max tokens:

| Method | tok/s | vs Autoregressive | Acceptance |
|--------|------:|------------------:|-----------:|
| Autoregressive | 27.9 | 1.0x | — |
| DFlash | 38.6 | **1.38x** | 85% |
| **DFlash + DDTree** | **42.3** | **1.52x** | 4.2/cycle |

DDTree adds **~10-15% on top of DFlash** for code and structured content where draft acceptance is high. Output is lossless -- every token is verified against the target model.

### When DDTree Helps (and When It Doesn't)

| Content Type | DFlash Acceptance | DDTree Benefit |
|-------------|------------------:|:---------------|
| Code generation | 85%+ | **+10-15%** over DFlash — tree catches rejected tokens with backup branches |
| Structured/factual | 70-80% | **+10-15%** — moderate acceptance leaves room for tree alternatives |
| Creative prose | 5-10% | **~0%** — low acceptance means most tree branches are wrong too; DDTree roughly equals autoregressive |

DDTree's advantage depends entirely on draft model acceptance. When the draft model predicts well (code, structured output), the tree's backup branches catch occasional misses. When the draft model struggles (creative writing, open-ended prose), tree branches are just as wrong as the primary guess, and the tree overhead eats any gain.

## How It Works

1. **Draft**: The DFlash block diffusion model generates per-position token probabilities in parallel
2. **Tree Build**: A heap-based algorithm constructs an optimal draft tree from the top-K tokens at each position, maximizing coverage under a node budget
3. **Tree Verify**: All tree nodes are verified through the target model in one forward pass using tree attention masks (ancestor-only visibility) and per-token RoPE positions
4. **Tree Walk**: Greedy walk through the verified tree to find the longest accepted path
5. **Commit**: The accepted path's cache state is installed directly via per-node state capture (zero-cost commit)

### Hybrid Model Support

Qwen 3.5 27B is a hybrid architecture with 48 GatedDeltaNet (recurrent) layers and 16 full attention layers. DDTree handles this with:

- **Attention layers**: Process all tree nodes in parallel via custom tree attention masks
- **Recurrent layers**: A custom Metal kernel performs parent-indexed GatedDelta recurrence, forking state at each branch point so every tree path gets exact logits
- **Tree-aware commit**: Accepted path's recurrent state is installed directly from captured per-node states, eliminating the need for re-forward passes

## Installation

Requires Python 3.11+ and Apple Silicon (M1/M2/M3/M4).

```bash
# Install dflash-mlx (required dependency)
pip install dflash-mlx

# Clone and install ddtree-mlx
git clone https://github.com/humanrouter/ddtree-mlx.git
cd ddtree-mlx
pip install -e .
```

The target model and DFlash drafter will be downloaded automatically on first run from Hugging Face:
- Target: `mlx-community/Qwen3.5-27B-4bit` (~16GB)
- Drafter: `z-lab/Qwen3.5-27B-DFlash` (~3GB)
- Total memory: ~19GB

## Usage

### OpenAI-Compatible Server

```bash
python ddtree_server.py --port 8006
```

Then use any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8006/v1", api_key="unused")
response = client.chat.completions.create(
    model="ddtree",
    messages=[{"role": "user", "content": "Explain TCP vs UDP"}],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

### Python API

```python
from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once

# Load models (downloads from HF on first run)
target_model, tokenizer, draft_model, _ = load_runtime_components(
    model_ref="mlx-community/Qwen3.5-27B-4bit"
)

# Tokenize
prompt_tokens = list(tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Python quicksort"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
))

# Generate
result = generate_ddtree_once(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    prompt_tokens=prompt_tokens,
    max_new_tokens=2048,
    tree_budget=4,
    stop_token_ids=get_stop_token_ids(tokenizer),
)

print(tokenizer.decode(result["generated_token_ids"]))
print(f"{result['tokens_per_second']:.1f} tok/s, "
      f"{result['avg_acceptance']:.1f} tokens/cycle, "
      f"{result['fast_path_ratio']:.0%} fast path")
```

### Benchmarking

```bash
python benchmark.py --max-tokens 2048 --budgets 4 --prompts 3
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DDTREE_BUDGET` | `4` | Tree node budget (excluding root). Budget 4 is optimal for hybrid models. |
| `DDTREE_EXACT_COMMIT` | `0` | Re-forward accepted tokens sequentially (slow but guaranteed lossless). Usually unnecessary — the kernel precision fix makes tree commit match sequential. |
| `DDTREE_TREE_AWARE_LINEAR` | `1` | Enable parent-state forking for recurrent layers (recommended). |
| `DDTREE_TREE_KERNEL` | `1` | Use custom Metal kernel for tree-aware GatedDelta recurrence. |
| `DDTREE_TREE_CONV_KERNEL` | `1` | Use Metal kernel for parent-aware causal conv inside GatedDelta verification. |
| `DDTREE_EXACT_TREE_ATTENTION` | `0` | Opt-in exact prefix/tree attention without a prefix-width mask. Set to `auto` for long-context testing. |
| `DDTREE_EXACT_TREE_ATTENTION_MIN_PREFIX` | `8192` | Prefix length where exact split attention turns on in `auto` mode. |
| `DDTREE_DFLASH_CONTROLLER` | `0` | Opt-in in-place controller that can switch future cycles to DFlash after sustained probe wins. |
| `DDTREE_PROFILE_VERIFY` | `0` | Profile linear vs attention layer timing within tree verify. Use `detail` for per-operation timings. |
| `DDTREE_PROFILE_DETAIL` | `0` | Enable detailed synchronized verify timings when `DDTREE_PROFILE_VERIFY` is set. |

## Architecture

```
ddtree_mlx/
  tree.py       # Heap-based tree construction (Algorithm 1 from the paper)
  compile.py    # Converts tree structure to MLX tensors (masks, positions, DFS order)
  verify.py     # Custom forward pass: tree attention + parent-indexed recurrence
  kernels.py    # Metal kernels for tree-aware conv and GatedDelta state update
  cache.py      # Cache management: snapshot, rollback, tree-aware path commit
  runtime.py    # Main generate loop: draft -> build -> verify -> walk -> commit
ddtree_server.py  # OpenAI-compatible FastAPI server
benchmark.py      # Benchmark script (DDTree vs DFlash comparison)
```

## Quantization & Model Compatibility

DDTree works at the architecture level (tree attention masks, per-token RoPE, parent-indexed recurrence), so it applies across **any quantization** of the same model family. For Qwen 3.5 27B on M3 Ultra:

| Quantization | AR tok/s | Memory | DDTree estimated | vs AR |
|-------------|----------:|-------:|-----------------:|------:|
| 4-bit | 55 | ~16GB | ~73-95 tok/s | ~1.9-2.6x |
| 6-bit | 56 | ~22GB | ~74-97 tok/s | ~1.9-2.6x |
| mxfp8 | 57 | ~30GB | ~75-99 tok/s | ~1.9-2.6x |
| bf16 | 57 | ~54GB | ~75-99 tok/s | ~1.9-2.6x |

AR speeds are nearly identical across quantizations on M3 Ultra (memory bandwidth bottlenecked). Since DDTree's speedup is a multiplier on top of the base speed, the same ~2-2.6x ratio applies to all of them. The practical sweet spot is **4-bit** — same speed as bf16 but 3.4x less memory.

### The Draft Model is Key

DDTree's performance depends entirely on having a good **DFlash draft model** for the target model. The draft model (`z-lab/Qwen3.5-27B-DFlash`) is a small block diffusion model (~3GB) specifically trained to predict what Qwen 3.5 27B will say next. The better the draft model predicts, the higher the acceptance rate, and the bigger DDTree's speedup.

Currently, DFlash drafters exist for the **Qwen 3.5 family**. As more DFlash drafters get trained for other model families (Llama, Mistral, etc.), DDTree automatically extends to them — no code changes needed. The tree construction, verification, and commit logic are model-agnostic; only the draft model needs to match the target.

If no DFlash drafter is available for a model, DDTree cannot be used. This is the main adoption constraint — the acceleration is only as good as the draft model that powers it.

## Findings & Insights

See [BENCHMARKS.md](BENCHMARKS.md) for detailed results, including:

- **What worked**: Metal kernels for tree-aware conv/recurrent verification, zero-cost commit via per-node state capture, eval sync point reduction
- **What didn't work**: Attention-only tree verify (LM head needs all 64 layers), alternative tree shapes (chain, hybrid, root-wide), split prefix/tree attention, adaptive budget controller
- **The fundamental constraint**: On hybrid models (75% recurrent layers), tree verification has limited parallelism. DDTree's advantage comes from better acceptance density -- the tree concentrates budget on the most probable tokens. Pure-attention models (Llama, standard Qwen) would benefit more.

## Citation

```bibtex
@article{ringel2025ddtree,
  title={Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author={Ringel, Liran and Romano, Yaniv},
  year={2025},
  url={https://liranringel.github.io/ddtree/}
}
```

## License

MIT
