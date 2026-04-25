"""Tests for DDTree tree building and walking."""

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModel, TextModelArgs
from dflash_mlx.runtime import make_target_cache, target_forward_with_hidden_states
import ddtree_mlx.verify as verify_module
from ddtree_mlx.cache import tree_aware_path_commit
from ddtree_mlx.compile import compile_tree
from ddtree_mlx.runtime import _build_tree_from_mlx_logits, _walk_dfs_exact_prefix
from ddtree_mlx.verify import (
    _linear_forward_tree_aware,
    _split_prefix_tree_attention_exact,
    _tree_depth_groups,
    tree_verify_forward,
)
from ddtree_mlx.kernels import tree_conv1d_kernel
from ddtree_mlx.tree import build_ddtree_tree, follow_verified_tree, compute_dfs_order


def test_empty_budget():
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=0)
    assert tree.node_count == 0
    assert tree.visibility.shape == (1, 1)
    assert tree.visibility[0, 0] == True


def test_empty_logits():
    logits = np.empty((0, 100), dtype=np.float32)
    tree = build_ddtree_tree(logits, budget=16)
    assert tree.node_count == 0


def test_basic_tree_structure():
    """Build a tree with budget=4 from 3-position logits."""
    np.random.seed(42)
    logits = np.random.randn(3, 50).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=4)

    assert tree.node_count == 4
    assert len(tree.node_token_ids) == 4
    assert len(tree.node_depths) == 4
    assert len(tree.parents) == 5  # root + 4 nodes
    assert len(tree.child_maps) == 5
    assert tree.visibility.shape == (5, 5)

    # Root sees only itself
    assert tree.visibility[0, 0] == True
    assert not np.any(tree.visibility[0, 1:])

    # Every node sees itself
    for i in range(5):
        assert tree.visibility[i, i] == True

    # Every non-root node sees root
    for i in range(1, 5):
        assert tree.visibility[i, 0] == True

    # Parent is always -1 for root
    assert tree.parents[0] == -1

    # All depths are >= 1
    assert np.all(tree.node_depths >= 1)
    assert np.all(tree.node_depths <= 3)


def test_visibility_is_ancestor_only():
    """Verify that visibility matrix only allows ancestor attention."""
    np.random.seed(0)
    logits = np.random.randn(4, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=8)

    # For each node, check it can only see its ancestors and itself
    for i in range(1, 1 + tree.node_count):
        # Collect ancestors
        ancestors = set()
        node = i
        while node >= 0:
            ancestors.add(node)
            node = tree.parents[node]

        for j in range(1 + tree.node_count):
            if j in ancestors:
                assert tree.visibility[i, j], f"Node {i} should see ancestor {j}"
            else:
                assert not tree.visibility[i, j], f"Node {i} should NOT see non-ancestor {j}"


def test_follow_verified_tree_full_accept():
    """Test tree walk when all tokens on primary path match."""
    np.random.seed(42)
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=8)

    # Build posterior that exactly matches the primary path (depth-first, rank-0 tokens)
    # Find the primary path: root -> highest prob child -> ...
    primary_path = [0]
    current = 0
    posterior = [0] * (1 + tree.node_count)
    for depth_idx in range(tree.node_depths.max()):
        children = tree.child_maps[current]
        if not children:
            break
        # Pick the first child (should be highest prob from heap)
        token_id = next(iter(children))
        child_idx = children[token_id]
        posterior[current] = token_id
        primary_path.append(child_idx)
        current = child_idx

    # Set a dummy posterior for the last node
    posterior[current] = 99999

    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert accepted == primary_path
    assert bonus == 99999


def test_follow_verified_tree_immediate_reject():
    """Test tree walk when first token doesn't match any child."""
    np.random.seed(42)
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=4)

    # Posterior with a token that matches no child
    posterior = [-1] * (1 + tree.node_count)
    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert accepted == [0]  # only root accepted
    assert bonus == -1


def test_dfs_order():
    """Test DFS ordering produces valid traversal."""
    np.random.seed(42)
    logits = np.random.randn(4, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=16)

    dfs_order, inv_dfs_order = compute_dfs_order(tree)

    # DFS order should contain all nodes exactly once
    assert sorted(dfs_order) == list(range(1 + tree.node_count))

    # Root should be first
    assert dfs_order[0] == 0

    # Inverse should round-trip
    for pos, idx in enumerate(dfs_order):
        assert inv_dfs_order[idx] == pos

    # Parent should appear before child in DFS order
    for idx in range(1, 1 + tree.node_count):
        parent = tree.parents[idx]
        assert inv_dfs_order[parent] < inv_dfs_order[idx], \
            f"Parent {parent} (pos {inv_dfs_order[parent]}) should precede child {idx} (pos {inv_dfs_order[idx]})"


def test_budget_respected():
    """Tree should never have more nodes than budget."""
    for budget in [1, 2, 4, 8, 16, 32]:
        logits = np.random.randn(8, 200).astype(np.float32)
        tree = build_ddtree_tree(logits, budget=budget)
        assert tree.node_count <= budget


def test_budget_equal_vocab():
    """Tree build should support top-k equal to vocab size."""
    logits = np.random.randn(3, 16).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=16)
    assert tree.node_count == 16


def test_compile_tree_mask_is_tree_only():
    logits = np.random.randn(3, 50).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=4)
    compiled = compile_tree(tree, root_token_id=1, prefix_len=128)
    assert compiled.attention_mask.shape == (1, 1, 5, 5)


def test_mlx_tree_build_matches_numpy_tree_build():
    np.random.seed(7)
    logits = np.random.randn(4, 64).astype(np.float32)
    numpy_tree = build_ddtree_tree(logits, budget=8)
    mlx_tree = _build_tree_from_mlx_logits(mx.array(logits), budget=8)
    assert mlx_tree.node_token_ids.tolist() == numpy_tree.node_token_ids.tolist()
    assert mlx_tree.node_depths.tolist() == numpy_tree.node_depths.tolist()
    assert mlx_tree.parents == numpy_tree.parents


def test_walk_dfs_exact_prefix_fast_path():
    np.random.seed(42)
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=8)
    dfs_order, _ = compute_dfs_order(tree)

    posterior = [-1] * (1 + tree.node_count)
    first_child = dfs_order[1]
    posterior[0] = int(tree.node_token_ids[first_child - 1])
    posterior[first_child] = 99999

    accepted, bonus, exact_prefix_len = _walk_dfs_exact_prefix(
        tree.child_maps, posterior, dfs_order
    )
    assert accepted == [0, first_child]
    assert bonus == 99999
    assert exact_prefix_len == len(accepted)


def test_walk_dfs_exact_prefix_divergence():
    logits = np.array(
        [
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
        ],
        dtype=np.float32,
    )
    tree = build_ddtree_tree(logits, budget=4)
    dfs_order, _ = compute_dfs_order(tree)

    root_children = list(tree.child_maps[0].values())
    assert len(root_children) >= 2
    divergent_child = root_children[1]
    posterior = [-1] * (1 + tree.node_count)
    posterior[0] = int(tree.node_token_ids[divergent_child - 1])

    accepted, bonus, exact_prefix_len = _walk_dfs_exact_prefix(
        tree.child_maps, posterior, dfs_order
    )
    assert accepted == [0, divergent_child]
    assert bonus is None
    assert exact_prefix_len == 1


def test_tree_aware_gated_delta_matches_chain_forward():
    args = TextModelArgs(
        hidden_size=8,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
    )
    linear = GatedDeltaNet(args)
    inputs = mx.random.normal((1, 4, 8))

    seq_cache = ArraysCache(size=2)
    seq_out = linear(inputs, cache=seq_cache)

    tree_cache = ArraysCache(size=2)
    parents = [-1, 0, 1, 2]
    depths = [0, 1, 2, 3]
    tree_out, node_states, node_conv_states = _linear_forward_tree_aware(
        linear,
        inputs,
        tree_cache,
        parents=parents,
        depth_groups=_tree_depth_groups(depths),
    )
    mx.eval(seq_out, tree_out, node_states, node_conv_states)

    assert np.max(np.abs(np.array(seq_out - tree_out))) < 1e-4
    assert np.max(np.abs(np.array(seq_cache[0] - node_conv_states[-1:]))) < 1e-4
    assert np.max(np.abs(np.array(seq_cache[1] - node_states[-1:]))) < 1e-4


def test_tree_aware_gated_delta_forks_branch_state():
    args = TextModelArgs(
        hidden_size=8,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
    )
    linear = GatedDeltaNet(args)
    inputs = mx.random.normal((1, 3, 8))

    tree_out, _, _ = _linear_forward_tree_aware(
        linear,
        inputs,
        ArraysCache(size=2),
        parents=[-1, 0, 0],
        depth_groups=_tree_depth_groups([0, 1, 1]),
    )

    path_01 = mx.take(inputs, mx.array([0, 1], dtype=mx.int32), axis=1)
    path_02 = mx.take(inputs, mx.array([0, 2], dtype=mx.int32), axis=1)
    out_01 = linear(path_01, cache=ArraysCache(size=2))
    out_02 = linear(path_02, cache=ArraysCache(size=2))
    mx.eval(tree_out, out_01, out_02)

    assert np.max(np.abs(np.array(tree_out[:, 0, :] - out_01[:, 0, :]))) < 1e-4
    assert np.max(np.abs(np.array(tree_out[:, 1, :] - out_01[:, 1, :]))) < 1e-4
    assert np.max(np.abs(np.array(tree_out[:, 2, :] - out_02[:, 1, :]))) < 1e-4


def test_tree_aware_gated_delta_kernel_matches_fallback():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return

    args = TextModelArgs(
        hidden_size=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
    )
    linear = GatedDeltaNet(args)
    inputs = mx.random.normal((1, 6, 64))
    parents = [-1, 0, 0, 1, 1, 2]
    depths = [0, 1, 1, 2, 2, 2]
    depth_groups = _tree_depth_groups(depths)

    old_kernel_flag = verify_module._TREE_KERNEL_ENABLED
    try:
        verify_module._TREE_KERNEL_ENABLED = False
        fallback_out, fallback_states, fallback_conv_states = _linear_forward_tree_aware(
            linear,
            inputs,
            ArraysCache(size=2),
            parents=parents,
            depth_groups=depth_groups,
        )

        verify_module._TREE_KERNEL_ENABLED = True
        kernel_out, kernel_states, kernel_conv_states = _linear_forward_tree_aware(
            linear,
            inputs,
            ArraysCache(size=2),
            parents=parents,
            depth_groups=depth_groups,
        )
    finally:
        verify_module._TREE_KERNEL_ENABLED = old_kernel_flag

    mx.eval(
        fallback_out,
        fallback_states,
        fallback_conv_states,
        kernel_out,
        kernel_states,
        kernel_conv_states,
    )
    assert np.max(np.abs(np.array(fallback_out - kernel_out))) < 1e-4
    assert np.max(np.abs(np.array(fallback_states - kernel_states))) < 1e-4
    assert np.max(np.abs(np.array(fallback_conv_states - kernel_conv_states))) < 1e-4


def test_tree_conv1d_kernel_matches_python_reference():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return

    rng = np.random.default_rng(123)
    qkv_np = rng.normal(size=(1, 6, 11)).astype(np.float32)
    base_np = rng.normal(size=(1, 3, 11)).astype(np.float32)
    weight_np = rng.normal(size=(4, 11)).astype(np.float32)
    parents = [-1, 0, 0, 1, 2, 3]

    conv_out, conv_states = tree_conv1d_kernel(
        mx.array(qkv_np),
        mx.array(base_np),
        mx.array(weight_np),
        mx.array(parents, dtype=mx.int32),
    )
    mx.eval(conv_out, conv_states)

    expected_out = np.empty_like(qkv_np)
    expected_states = np.empty((1, 6, 3, 11), dtype=np.float32)
    for t, parent in enumerate(parents):
        parent_state = base_np[0] if parent < 0 else expected_states[0, parent]
        conv_input = np.concatenate([parent_state, qkv_np[0, t : t + 1]], axis=0)
        raw = (conv_input * weight_np).sum(axis=0)
        expected_out[0, t] = raw / (1.0 + np.exp(-raw))
        expected_states[0, t] = conv_input[-3:]

    assert np.max(np.abs(np.array(conv_out) - expected_out)) < 1e-5
    assert np.max(np.abs(np.array(conv_states) - expected_states)) < 1e-5


def test_exact_prefix_tree_attention_matches_masked_sdpa():
    prefix_len = 5
    tree_len = 3
    heads = 2
    head_dim = 4

    queries = mx.random.normal((1, heads, tree_len, head_dim))
    keys = mx.random.normal((1, 1, prefix_len + tree_len, head_dim))
    values = mx.random.normal((1, 1, prefix_len + tree_len, head_dim))
    tree_visible = np.array(
        [
            [True, False, False],
            [True, True, False],
            [True, False, True],
        ],
        dtype=np.bool_,
    )
    tree_mask = mx.array(np.where(tree_visible, 0.0, -np.inf).astype(np.float32))[
        None,
        None,
        :,
        :,
    ]
    prefix_mask = mx.zeros((1, 1, tree_len, prefix_len), dtype=mx.float32)
    full_mask = mx.concatenate([prefix_mask, tree_mask], axis=-1)
    repeated_keys = mx.repeat(keys, heads, axis=1)
    repeated_values = mx.repeat(values, heads, axis=1)

    baseline = mx.fast.scaled_dot_product_attention(
        queries,
        repeated_keys,
        repeated_values,
        scale=head_dim ** -0.5,
        mask=full_mask,
    )
    exact = _split_prefix_tree_attention_exact(
        queries=queries,
        keys=keys,
        values=values,
        scale=head_dim ** -0.5,
        tree_mask=tree_mask,
        cached_prefix_len=prefix_len,
    )
    mx.eval(baseline, exact)
    assert np.max(np.abs(np.array(baseline - exact))) < 1e-4


class _FakeKVCache:
    def __init__(self):
        self.keys = mx.arange(6, dtype=mx.float32).reshape(1, 1, 6, 1)
        self.values = (mx.arange(6, dtype=mx.float32) + 10).reshape(1, 1, 6, 1)
        self.offset = 6


class _FakeLinearCache:
    def __init__(self):
        self.state = [None, None]


def test_tree_aware_path_commit_packs_arbitrary_path():
    linear_cache = _FakeLinearCache()
    kv_cache = _FakeKVCache()
    prefix_len = 2
    accepted = [0, 3, 1]
    original_keys = np.array(kv_cache.keys)
    original_values = np.array(kv_cache.values)
    tree_cache_state = {
        "linear_layers": {
            0: {
                "conv_states": mx.arange(4, dtype=mx.float32).reshape(4, 1, 1),
                "states": mx.arange(4, dtype=mx.float32).reshape(4, 1, 1, 1),
            }
        }
    }

    tree_aware_path_commit(
        [linear_cache, kv_cache],
        prefix_len=prefix_len,
        accepted_indices=accepted,
        tree_cache_state=tree_cache_state,
    )
    mx.eval(kv_cache.keys, kv_cache.values, *linear_cache.state)

    source_positions = [prefix_len + idx for idx in accepted]
    assert kv_cache.offset == prefix_len + len(accepted)
    assert np.array_equal(
        np.array(kv_cache.keys)[0, 0, prefix_len : kv_cache.offset, 0],
        original_keys[0, 0, source_positions, 0],
    )
    assert np.array_equal(
        np.array(kv_cache.values)[0, 0, prefix_len : kv_cache.offset, 0],
        original_values[0, 0, source_positions, 0],
    )
    assert np.array_equal(np.array(linear_cache.state[0]), np.array([[[1.0]]]))
    assert np.array_equal(np.array(linear_cache.state[1]), np.array([[[[1.0]]]]))


def test_tree_aware_verify_matches_sequential_path_logits():
    args = TextModelArgs(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        full_attention_interval=4,
    )
    model = TextModel(args)
    root_token = 1
    draft_logits = np.full((1, args.vocab_size), -10.0, dtype=np.float32)
    draft_logits[0, 2] = 10.0
    draft_logits[0, 3] = 9.0
    tree = build_ddtree_tree(draft_logits, budget=2)
    compiled = compile_tree(tree, root_token_id=root_token, prefix_len=0)

    tree_cache = make_target_cache(model, enable_speculative_linear_cache=False)
    tree_logits, _ = tree_verify_forward(
        model,
        compiled_tree=compiled,
        cache=tree_cache,
        capture_layer_ids=set(),
        tree_aware_linear=True,
        tree_cache_state={},
    )

    path_token = int(tree.node_token_ids[0])
    sequential_cache = make_target_cache(model, enable_speculative_linear_cache=False)
    sequential_logits, _ = target_forward_with_hidden_states(
        model,
        input_ids=mx.array([[root_token, path_token]], dtype=mx.uint32),
        cache=sequential_cache,
        capture_layer_ids=set(),
    )
    mx.eval(tree_logits, sequential_logits)

    tree_path_logits = mx.take(
        tree_logits,
        mx.array([0, 1], dtype=mx.int32),
        axis=1,
    )
    assert np.max(np.abs(np.array(tree_path_logits - sequential_logits))) < 1e-3


if __name__ == "__main__":
    test_empty_budget()
    test_empty_logits()
    test_basic_tree_structure()
    test_visibility_is_ancestor_only()
    test_follow_verified_tree_full_accept()
    test_follow_verified_tree_immediate_reject()
    test_dfs_order()
    test_budget_respected()
    test_budget_equal_vocab()
    test_compile_tree_mask_is_tree_only()
    test_mlx_tree_build_matches_numpy_tree_build()
    test_walk_dfs_exact_prefix_fast_path()
    test_walk_dfs_exact_prefix_divergence()
    test_tree_aware_gated_delta_matches_chain_forward()
    test_tree_aware_gated_delta_forks_branch_state()
    test_tree_aware_gated_delta_kernel_matches_fallback()
    test_tree_conv1d_kernel_matches_python_reference()
    test_exact_prefix_tree_attention_matches_masked_sdpa()
    test_tree_aware_path_commit_packs_arbitrary_path()
    test_tree_aware_verify_matches_sequential_path_logits()
    print("All tree tests passed!")
