#!/usr/bin/env python3
"""
Symbolic Shape Inference Analysis for PagedAttention in LLM Models.

This script creates a synthetic LLM model (similar to a small TinyLlama-like architecture),
applies the SDPAToPagedAttention transformation, runs SymbolicPropagation (via VisualizeTree),
and prints the symbolic shape propagation results for all operations.
"""

import os
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as ops
from openvino import Model, PartialShape, Type, Shape, Dimension
from openvino._pyopenvino.op.util import Variable, VariableInfo
from openvino._pyopenvino.op import assign, read_value
from openvino._offline_transformations import paged_attention_transformation
from openvino.passes import VisualizeTree, Manager

# Model configuration (small TinyLlama-like model with 1 layer)
BATCH = -1       # dynamic batch
SEQ_LEN = -1     # dynamic sequence length
NUM_HEADS = 4
HEAD_DIM = 8
HIDDEN_DIM = NUM_HEADS * HEAD_DIM  # 32
VOCAB_SIZE = 64


def make_variable(var_id, shape, dtype=Type.f32):
    """Create an OV Variable with the given id and shape."""
    vi = VariableInfo()
    vi.data_shape = PartialShape(shape)
    vi.data_type = dtype
    vi.variable_id = var_id
    return Variable(vi)


def create_llm_sdpa_model():
    """
    Create a synthetic LLM model with:
    - input_ids [batch, seq_len]
    - beam_idx [batch]
    - attention_mask [batch, full_seq_len]
    - position_ids [batch, seq_len]
    - Embedding layer
    - Single transformer layer with QKV projection, RoPE, KV cache, SDPA, output projection
    - LM head
    """
    # --- Parameters ---
    input_ids = ops.parameter(PartialShape([BATCH, SEQ_LEN]), Type.i64, name="input_ids")
    input_ids.output(0).get_tensor().set_names({"input_ids"})

    beam_idx = ops.parameter(PartialShape([BATCH]), Type.i64, name="beam_idx")
    beam_idx.output(0).get_tensor().set_names({"beam_idx"})

    attention_mask = ops.parameter(PartialShape([BATCH, SEQ_LEN]), Type.i64, name="attention_mask")
    attention_mask.output(0).get_tensor().set_names({"attention_mask"})

    position_ids = ops.parameter(PartialShape([BATCH, SEQ_LEN]), Type.i64, name="position_ids")
    position_ids.output(0).get_tensor().set_names({"position_ids"})

    params = [input_ids, beam_idx, attention_mask, position_ids]

    # --- Embedding ---
    embedding_table = ops.constant(np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32))
    embeddings = ops.gather(embedding_table, input_ids, ops.constant(np.int64(0)))
    # embeddings: [batch, seq_len, hidden_dim]

    # --- QKV Projection ---
    # Combined QKV weight: [hidden_dim, 3 * hidden_dim]
    qkv_weight = ops.constant(np.random.randn(HIDDEN_DIM, 3 * HIDDEN_DIM).astype(np.float32))
    qkv = ops.matmul(embeddings, qkv_weight, False, False)
    # qkv: [batch, seq_len, 3*hidden_dim]

    # Split into Q, K, V
    split_lengths = ops.constant(np.array([HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM], dtype=np.int64))
    qkv_split = ops.variadic_split(qkv, ops.constant(np.int64(-1)), split_lengths)
    q_proj = qkv_split.output(0)  # [batch, seq_len, hidden_dim]
    k_proj = qkv_split.output(1)  # [batch, seq_len, hidden_dim]
    v_proj = qkv_split.output(2)  # [batch, seq_len, hidden_dim]

    # Reshape to [batch, seq_len, num_heads, head_dim]
    reshape_pattern = ops.constant(np.array([0, 0, NUM_HEADS, HEAD_DIM], dtype=np.int64))
    q = ops.reshape(q_proj, reshape_pattern, True)  # special_zero=True
    k = ops.reshape(k_proj, reshape_pattern, True)
    v = ops.reshape(v_proj, reshape_pattern, True)

    # Transpose to [batch, num_heads, seq_len, head_dim]
    perm = ops.constant(np.array([0, 2, 1, 3], dtype=np.int64))
    q_t = ops.transpose(q, perm)
    k_t = ops.transpose(k, perm)
    v_t = ops.transpose(v, perm)

    # --- KV Cache ---
    # Key cache variable
    k_cache_var = make_variable("key_cache.0", [BATCH, -1, NUM_HEADS, HEAD_DIM])
    k_cache_init = ops.constant(np.zeros([1, 0, NUM_HEADS, HEAD_DIM], dtype=np.float32))
    k_read = read_value(k_cache_init, k_cache_var)

    # Gather past cache by beam_idx  [batch, past_seq, num_heads, head_dim]
    k_past = ops.gather(k_read, beam_idx, ops.constant(np.int64(0)))
    # Transpose past to [batch, num_heads, past_seq, head_dim]
    k_past_t = ops.transpose(k_past, perm)
    # Concatenate past and current key along seq dimension
    k_full = ops.concat([k_past_t, k_t], axis=2)
    # [batch, num_heads, total_seq, head_dim]

    # Value cache variable
    v_cache_var = make_variable("value_cache.0", [BATCH, -1, NUM_HEADS, HEAD_DIM])
    v_cache_init = ops.constant(np.zeros([1, 0, NUM_HEADS, HEAD_DIM], dtype=np.float32))
    v_read = read_value(v_cache_init, v_cache_var)
    v_past = ops.gather(v_read, beam_idx, ops.constant(np.int64(0)))
    v_past_t = ops.transpose(v_past, perm)
    v_full = ops.concat([v_past_t, v_t], axis=2)

    # --- SDPA ---
    # Build attention mask: [batch, 1, 1, total_seq]
    attn_mask_f32 = ops.convert(attention_mask, Type.f32)
    attn_mask_unsqueeze1 = ops.unsqueeze(attn_mask_f32, ops.constant(np.int64(1)))
    attn_mask_unsqueeze2 = ops.unsqueeze(attn_mask_unsqueeze1, ops.constant(np.int64(1)))

    sdpa = ops.scaled_dot_product_attention(q_t, k_full, v_full, attn_mask_unsqueeze2, causal=False)
    # sdpa: [batch, num_heads, seq_len, head_dim]

    # Transpose back to [batch, seq_len, num_heads, head_dim]
    sdpa_t = ops.transpose(sdpa, perm)
    # Reshape to [batch, seq_len, hidden_dim]
    reshape_out = ops.constant(np.array([0, 0, HIDDEN_DIM], dtype=np.int64))
    sdpa_flat = ops.reshape(sdpa_t, reshape_out, True)

    # --- Output projection ---
    out_weight = ops.constant(np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32))
    hidden = ops.matmul(sdpa_flat, out_weight, False, False)

    # --- LM Head ---
    lm_head_weight = ops.constant(np.random.randn(HIDDEN_DIM, VOCAB_SIZE).astype(np.float32))
    logits = ops.matmul(hidden, lm_head_weight, False, False)
    # logits: [batch, seq_len, vocab_size]

    result = ops.result(logits)

    # --- KV cache assignments (save updated cache) ---
    # Transpose k_full back to [batch, total_seq, num_heads, head_dim]
    k_new = ops.transpose(k_full, perm)
    v_new = ops.transpose(v_full, perm)

    k_assign = assign(k_new, k_cache_var)
    v_assign = assign(v_new, v_cache_var)

    model = Model(
        results=[result],
        sinks=[k_assign, v_assign],
        parameters=params,
        name="SyntheticLLM"
    )

    model.validate_nodes_and_infer_types()
    return model


def print_model_shapes(model, title="Model Shapes"):
    """Print all operation shapes with symbol information."""
    print(f"\n{'='*100}")
    print(f" {title}")
    print(f"{'='*100}")

    # Print parameters
    print(f"\n--- Parameters ---")
    for param in model.get_parameters():
        names = list(param.output(0).get_names())
        print(f"  {param.get_friendly_name():30s}  type={param.get_element_type()}  shape={param.get_partial_shape()}  names={names}")

    # Print all ops
    print(f"\n--- Operations (in topological order) ---")
    for op in model.get_ordered_ops():
        op_name = f"{op.get_type_name()} ({op.get_friendly_name()})"
        if len(op_name) > 60:
            op_name = op_name[:57] + "..."

        for i in range(op.get_output_size()):
            out = op.output(i)
            pshape = out.get_partial_shape()

            # Check for symbols
            sym_info = ""
            if pshape.rank.is_static:
                sym_dims = []
                for dim_idx in range(pshape.rank.get_length()):
                    dim = pshape[dim_idx]
                    symbol = dim.get_symbol()
                    if symbol is not None:
                        sym_dims.append(f"d{dim_idx}=S{id(symbol) % 10000}")
                    elif dim.is_dynamic:
                        sym_dims.append(f"d{dim_idx}=?")
                if sym_dims:
                    sym_info = f"  symbols=[{', '.join(sym_dims)}]"

            tensor_names = list(out.get_names()) if out.get_names() else []
            names_str = f"  names={tensor_names}" if tensor_names else ""

            if op.get_output_size() > 1:
                print(f"  {op_name:60s}  out[{i}]: {str(out.get_element_type()):8s} {str(pshape):30s}{sym_info}{names_str}")
            else:
                print(f"  {op_name:60s}  {str(out.get_element_type()):8s} {str(pshape):30s}{sym_info}{names_str}")

    # Print results
    print(f"\n--- Results ---")
    for result in model.get_results():
        pshape = result.input(0).get_partial_shape()
        print(f"  {result.get_friendly_name():30s}  shape={pshape}")

    print()


def print_paged_attention_details(model):
    """Print detailed information about PagedAttention operations and their inputs/outputs."""
    print(f"\n{'='*100}")
    print(f" PagedAttention Operation Details")
    print(f"{'='*100}")

    pa_count = 0
    for op in model.get_ordered_ops():
        if op.get_type_name() == "PagedAttentionExtension":
            pa_count += 1
            print(f"\n  PagedAttentionExtension #{pa_count} ({op.get_friendly_name()})")
            print(f"  {'─'*80}")

            pa_input_names = [
                "query", "key", "value", "key_cache", "value_cache",
                "past_lens", "subsequence_begins", "block_indices",
                "block_indices_begins", "scale", "sliding_window",
                "alibi_slopes", "max_context_len",
            ]

            print(f"  Inputs:")
            for i in range(op.get_input_size()):
                inp = op.input(i)
                src_output = inp.get_source_output()
                src_node = src_output.get_node()
                src_name = src_node.get_friendly_name()
                pshape = inp.get_partial_shape()

                name_label = pa_input_names[i] if i < len(pa_input_names) else f"input_{i}"

                sym_info = ""
                if pshape.rank.is_static:
                    for dim_idx in range(pshape.rank.get_length()):
                        dim = pshape[dim_idx]
                        symbol = dim.get_symbol()
                        if symbol is not None:
                            sym_info += f" d{dim_idx}=S{id(symbol) % 10000}"
                        elif dim.is_dynamic:
                            sym_info += f" d{dim_idx}=?"

                print(f"    [{i:2d}] {name_label:40s}  {str(inp.get_element_type()):8s} {str(pshape):25s}  <- {src_name}{(' symbols:' + sym_info) if sym_info else ''}")

            print(f"  Outputs:")
            for i in range(op.get_output_size()):
                out = op.output(i)
                pshape = out.get_partial_shape()

                sym_info = ""
                if pshape.rank.is_static:
                    for dim_idx in range(pshape.rank.get_length()):
                        dim = pshape[dim_idx]
                        symbol = dim.get_symbol()
                        if symbol is not None:
                            sym_info += f" d{dim_idx}=S{id(symbol) % 10000}"
                        elif dim.is_dynamic:
                            sym_info += f" d{dim_idx}=?"

                print(f"    [{i}] {str(out.get_element_type()):8s} {str(pshape):25s}{(' symbols:' + sym_info) if sym_info else ''}")

    if pa_count == 0:
        print("  No PagedAttentionExtension operations found in the model.")

    print()


def analyze_symbol_coverage(model):
    """Analyze what percentage of dynamic dimensions have symbols."""
    total_dynamic = 0
    symbolized = 0
    unsymbolized_ops = []

    for op in model.get_ordered_ops():
        for i in range(op.get_output_size()):
            out = op.output(i)
            pshape = out.get_partial_shape()
            if pshape.rank.is_static:
                for dim_idx in range(pshape.rank.get_length()):
                    dim = pshape[dim_idx]
                    if dim.is_dynamic:
                        total_dynamic += 1
                        if dim.get_symbol() is not None:
                            symbolized += 1
                        else:
                            unsymbolized_ops.append(
                                f"  {op.get_type_name()} ({op.get_friendly_name()}) output[{i}] dim[{dim_idx}]"
                            )

    print(f"\n{'='*100}")
    print(f" Symbol Coverage Analysis")
    print(f"{'='*100}")
    print(f"  Total dynamic dimensions: {total_dynamic}")
    print(f"  Symbolized:               {symbolized}")
    print(f"  Unsymbolized:             {total_dynamic - symbolized}")
    if total_dynamic > 0:
        print(f"  Coverage:                 {100.0 * symbolized / total_dynamic:.1f}%")

    if unsymbolized_ops:
        print(f"\n  Unsymbolized dynamic dimensions (these would cause fallback to slow shape inference path):")
        # Show unique ops only
        seen = set()
        for entry in unsymbolized_ops:
            op_key = entry.split("(")[0].strip()
            if op_key not in seen:
                print(entry)
                seen.add(op_key)
            if len(seen) > 20:
                print(f"  ... and {len(unsymbolized_ops) - 20} more")
                break
    print()


def count_unique_symbols(model):
    """Count and map all unique symbols in the model."""
    symbols = {}  # id -> list of (op_name, output_idx, dim_idx)

    for op in model.get_ordered_ops():
        for i in range(op.get_output_size()):
            out = op.output(i)
            pshape = out.get_partial_shape()
            if pshape.rank.is_static:
                for dim_idx in range(pshape.rank.get_length()):
                    dim = pshape[dim_idx]
                    symbol = dim.get_symbol()
                    if symbol is not None:
                        sym_id = id(symbol) % 10000
                        if sym_id not in symbols:
                            symbols[sym_id] = []
                        symbols[sym_id].append((op.get_type_name(), op.get_friendly_name(), i, dim_idx))

    print(f"\n{'='*100}")
    print(f" Unique Symbols Map")
    print(f"{'='*100}")
    print(f"  Total unique symbols: {len(symbols)}")
    for sym_id, usages in sorted(symbols.items()):
        print(f"\n  Symbol S{sym_id} (used in {len(usages)} places):")
        for op_type, op_name, out_idx, dim_idx in usages[:5]:
            print(f"    {op_type} ({op_name}) output[{out_idx}] dim[{dim_idx}]")
        if len(usages) > 5:
            print(f"    ... and {len(usages) - 5} more")
    print()


def main():
    print("Creating synthetic LLM model with SDPA...")
    model = create_llm_sdpa_model()
    print(f"Model created: {model.get_friendly_name()}")
    print(f"  Parameters: {len(model.get_parameters())}")
    print(f"  Results:    {len(model.get_results())}")
    print(f"  Ops:        {len(model.get_ordered_ops())}")

    # Count SDPA ops
    sdpa_count = sum(1 for op in model.get_ordered_ops() if op.get_type_name() == "ScaledDotProductAttention")
    print(f"  SDPA ops:   {sdpa_count}")

    # Save the SDPA model before transformation
    ov.save_model(model, "/tmp/llm_sdpa_before.xml")
    print("\nSaved SDPA model to /tmp/llm_sdpa_before.xml")

    print_model_shapes(model, "BEFORE PagedAttention Transformation (SDPA model)")

    # --- Apply SDPAToPagedAttention transformation ---
    print("\n" + "="*100)
    print(" Applying paged_attention_transformation...")
    print("="*100)

    paged_attention_transformation(
        model,
        False,  # use_block_indices_inputs
        False,  # use_score_outputs
        False,  # allow_score_aggregation
        False,  # allow_cache_rotation
        False,  # allow_xattention
        False,  # allow_adaptive_rkv
    )

    model.validate_nodes_and_infer_types()

    # Count PA ops
    pa_count = sum(1 for op in model.get_ordered_ops() if op.get_type_name() == "PagedAttentionExtension")
    sdpa_count_after = sum(1 for op in model.get_ordered_ops() if op.get_type_name() == "ScaledDotProductAttention")
    print(f"  SDPA ops after:  {sdpa_count_after}")
    print(f"  PA ops after:    {pa_count}")
    print(f"  Parameters:      {len(model.get_parameters())}")

    # Save the PA model
    ov.save_model(model, "/tmp/llm_pa_model.xml")
    print("Saved PA model to /tmp/llm_pa_model.xml")

    print_model_shapes(model, "AFTER PagedAttention Transformation (before SymbolicPropagation)")
    print_paged_attention_details(model)

    # --- Apply SymbolicPropagation via VisualizeTree ---
    print("\n" + "="*100)
    print(" Running SymbolicPropagation (via OV_VISUALIZE_APPLY_SYMBOLIC_PROPAGATION)...")
    print("="*100)

    # Set the env var that triggers symbolic propagation in VisualizeTree
    os.environ["OV_VISUALIZE_APPLY_SYMBOLIC_PROPAGATION"] = "1"
    os.environ["OV_VISUALIZE_TREE_OUTPUT_SHAPES"] = "1"
    os.environ["OV_VISUALIZE_TREE_OUTPUT_TYPES"] = "1"
    os.environ["OV_VISUALIZE_TREE_IO"] = "1"
    os.environ["OV_VISUALIZE_PARTIAL_VALUES_AND_LABELS"] = "1"

    # Run VisualizeTree which triggers SymbolicPropagation + generates DOT file
    vis = VisualizeTree("/tmp/llm_pa_symbolic.svg", dot_only=True)
    manager = Manager()
    manager.register_pass(vis)
    manager.run_passes(model)

    print("Generated symbolic visualization: /tmp/llm_pa_symbolic.svg")

    # Now analyze the model with symbols
    print_model_shapes(model, "AFTER SymbolicPropagation (symbols should be visible)")
    print_paged_attention_details(model)
    analyze_symbol_coverage(model)
    count_unique_symbols(model)

    # Save the symbolized model
    ov.save_model(model, "/tmp/llm_pa_symbolized.xml")
    print("Saved symbolized PA model to /tmp/llm_pa_symbolized.xml")

    # Print summary
    print(f"\n{'='*100}")
    print(f" SUMMARY")
    print(f"{'='*100}")
    print(f"  1. Created synthetic LLM with {NUM_HEADS} heads, {HEAD_DIM} head dim, {HIDDEN_DIM} hidden dim")
    print(f"  2. Applied SDPAToPagedAttention transformation")
    print(f"  3. Ran SymbolicPropagation pass")
    print(f"  4. Models saved to /tmp/llm_sdpa_before.xml, /tmp/llm_pa_model.xml, /tmp/llm_pa_symbolized.xml")
    print(f"  5. Visualization saved to /tmp/llm_pa_symbolic.svg (DOT format)")
    print()


if __name__ == "__main__":
    main()
