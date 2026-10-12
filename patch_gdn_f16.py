#!/usr/bin/env python3
"""Patch gated_delta_net.cu to support F16 recurrent state.
Strategy:
1. Add T_state template param to all 3 kernels
2. Add T_state to all launch functions  
3. Add gdn_launch_all helper that encapsulates KDA/keep_rs/emit_ingr dispatch
4. Rewrite ggml_cuda_op_gated_delta_net_impl with clean 4-way dispatch
"""
import re, sys

path = "ggml/src/ggml-cuda/gated_delta_net.cu"
with open(path, "r") as f:
    src = f.read()

original_len = len(src)

# ============================================================
# PHASE 1: Kernel template params and state pointer types
# ============================================================

# --- Kernel 1: gated_delta_net_cuda ---
src = src.replace(
    "template <typename T_in = float, int S_v = 128, bool KDA = false, bool keep_rs_t = false, bool emit_ingredients_t = false>",
    "template <typename T_in = float, typename T_state = float, int S_v = 128, bool KDA = false, bool keep_rs_t = false, bool emit_ingredients_t = false>"
)

src = src.replace(
    "                                     const float * curr_state,\n                                     T_in *       dst,\n                                     float *       state,",
    "                                     const T_state * curr_state,\n                                     T_in *       dst,\n                                     T_state *       state,"
)

src = src.replace(
    "    float * state_out;\n    if constexpr (emit_ingredients_t) {",
    "    T_state * state_out;\n    if constexpr (emit_ingredients_t) {"
)

# State read in kernel 1
src = src.replace(
    "        s_shard[r]  = curr_state[i];\n    }\n\n    for (int t = 0; t < n_tokens; t++) {\n        const T_in * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;",
    "        s_shard[r]  = gdn_to_float(curr_state[i]);\n    }\n\n    for (int t = 0; t < n_tokens; t++) {\n        const T_in * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;"
)

# Ingredients slot
src = src.replace(
    "                    float * ingr_slot = state_out + (int64_t) target_slot * state_slot_stride;",
    "                    T_state * ingr_slot = state_out + (int64_t) target_slot * state_slot_stride;"
)

# Ingredients writes
src = src.replace(
    "                            ingr_slot[i]   = k_reg[r];\n                            if constexpr (KDA) {\n                                ingr_slot[2 * S_v + i] = gdn_to_float(g_t[i]);\n                            } else {\n                                ingr_slot[2 * S_v + i] = gdn_to_float(*g_t);\n                            }\n                            ingr_slot[3 * S_v + i] = beta_val;",
    "                            ingr_slot[i]   = gdn_from_float<T_state>(k_reg[r]);\n                            if constexpr (KDA) {\n                                ingr_slot[2 * S_v + i] = gdn_from_float<T_state>(gdn_to_float(g_t[i]));\n                            } else {\n                                ingr_slot[2 * S_v + i] = gdn_from_float<T_state>(gdn_to_float(*g_t));\n                            }\n                            ingr_slot[3 * S_v + i] = gdn_from_float<T_state>(beta_val);"
)

src = src.replace(
    "                        ingr_slot[S_v + col] = gdn_to_float(v_t[col]);",
    "                        ingr_slot[S_v + col] = gdn_from_float<T_state>(gdn_to_float(v_t[col]));"
)

# keep_rs_t snapshot in kernel 1
src = src.replace(
    "                    float * curr_state = state_out + (int64_t) target_slot * state_slot_stride;\n#pragma unroll\n                    for (int r = 0; r < rows_per_lane; r++) {\n                        const int i = r * warp_size + lane;\n                        curr_state[col * S_v + i] = s_shard[r];\n                    }",
    "                    T_state * curr_state = state_out + (int64_t) target_slot * state_slot_stride;\n#pragma unroll\n                    for (int r = 0; r < rows_per_lane; r++) {\n                        const int i = r * warp_size + lane;\n                        curr_state[col * S_v + i] = gdn_from_float<T_state>(s_shard[r]);\n                    }"
)

# Checkpoint slot
src = src.replace(
    "                    float * ckpt_slot = state + (int64_t) K * state_slot_stride +\n                                         S_v * S_v * H * n_seqs + (sequence * H + h_idx) * S_v * S_v;\n#pragma unroll\n                    for (int r = 0; r < rows_per_lane; r++) {\n                        const int i = r * warp_size + lane;\n                        ckpt_slot[col * S_v + i] = s_shard[r];\n                    }",
    "                    T_state * ckpt_slot = state + (int64_t) K * state_slot_stride +\n                                         S_v * S_v * H * n_seqs + (sequence * H + h_idx) * S_v * S_v;\n#pragma unroll\n                    for (int r = 0; r < rows_per_lane; r++) {\n                        const int i = r * warp_size + lane;\n                        ckpt_slot[col * S_v + i] = gdn_from_float<T_state>(s_shard[r]);\n                    }"
)

# Final slot
src = src.replace(
    "        float * final_slot = state + (int64_t) K * state_slot_stride + (sequence * H + h_idx) * S_v * S_v;\n#pragma unroll\n        for (int r = 0; r < rows_per_lane; r++) {\n            const int i = r * warp_size + lane;\n            final_slot[col * S_v + i] = s_shard[r];\n        }",
    "        T_state * final_slot = state + (int64_t) K * state_slot_stride + (sequence * H + h_idx) * S_v * S_v;\n#pragma unroll\n        for (int r = 0; r < rows_per_lane; r++) {\n            const int i = r * warp_size + lane;\n            final_slot[col * S_v + i] = gdn_from_float<T_state>(s_shard[r]);\n        }"
)

# Final state write (!keep_rs_t) in kernel 1
src = src.replace(
    "    if constexpr (!keep_rs_t) {\n#pragma unroll\n        for (int r = 0; r < rows_per_lane; r++) {\n            const int i              = r * warp_size + lane;\n            state_out[col * S_v + i] = s_shard[r];\n        }\n    }\n}",
    "    if constexpr (!keep_rs_t) {\n#pragma unroll\n        for (int r = 0; r < rows_per_lane; r++) {\n            const int i              = r * warp_size + lane;\n            state_out[col * S_v + i] = gdn_from_float<T_state>(s_shard[r]);\n        }\n    }\n}"
)

# --- Kernel 2: gated_delta_net_cuda_ilp ---
src = src.replace(
    "template <typename T_in = float, int S_v, bool keep_rs_t, int NC, bool PREFETCH>",
    "template <typename T_in = float, typename T_state = float, int S_v, bool keep_rs_t, int NC, bool PREFETCH>"
)

src = src.replace(
    "                         const float * curr_state,\n                         T_in *       dst,\n                         float *       state,",
    "                         const T_state * curr_state,\n                         T_in *       dst,\n                         T_state *       state,"
)

# State read in kernel 2
src = src.replace(
    "            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];\n        }\n    }\n\n    const T_in * q_base",
    "            s_shard[c][r] = gdn_to_float(curr_state[c * S_v + r * warp_size + lane]);\n        }\n    }\n\n    const T_in * q_base"
)

# Snapshot in kernel 2
src = src.replace(
    "                float * snap = state + target_slot * state_slot_stride;\n#pragma unroll\n                for (int c = 0; c < NC; c++) {\n#pragma unroll\n                    for (int r = 0; r < rows_per_lane; r++) {\n                        snap[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];\n                    }\n                }",
    "                T_state * snap = state + target_slot * state_slot_stride;\n#pragma unroll\n                for (int c = 0; c < NC; c++) {\n#pragma unroll\n                    for (int r = 0; r < rows_per_lane; r++) {\n                        snap[(col0 + c) * S_v + r * warp_size + lane] = gdn_from_float<T_state>(s_shard[c][r]);\n                    }\n                }"
)

# Final state write in kernel 2
src = src.replace(
    "    if constexpr (!keep_rs_t) {\n#pragma unroll\n        for (int c = 0; c < NC; c++) {\n#pragma unroll\n            for (int r = 0; r < rows_per_lane; r++) {\n                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];\n            }\n        }\n    }\n}",
    "    if constexpr (!keep_rs_t) {\n#pragma unroll\n        for (int c = 0; c < NC; c++) {\n#pragma unroll\n            for (int r = 0; r < rows_per_lane; r++) {\n                state[(col0 + c) * S_v + r * warp_size + lane] = gdn_from_float<T_state>(s_shard[c][r]);\n            }\n        }\n    }\n}"
)

# --- Kernel 3: gated_delta_net_tiled_cuda ---
src = src.replace(
    "template <typename T_in, int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool keep_rs_t>",
    "template <typename T_in, typename T_state, int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool keep_rs_t>"
)

src = src.replace(
    "                           const float * curr_state,\n                           T_in *       dst,\n                           float *       state,",
    "                           const T_state * curr_state,\n                           T_in *       dst,\n                           T_state *       state,"
)

# State read in kernel 3
src = src.replace(
    "            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];\n        }\n    }\n\n    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {",
    "            s_shard[c][r] = gdn_to_float(curr_state[c * S_v + r * warp_size + lane]);\n        }\n    }\n\n    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {"
)

# Snapshot in kernel 3 - need to find the exact pattern
# The tiled kernel snapshot uses different variable names
# Let me search for it
m = re.search(r'(float \* snapshot = state \+ target_slot \* state_slot_stride;\n#pragma unroll\n                        for \(int c = 0; c < COLS; c\+\+\) \{\n#pragma unroll\n                            for \(int r = 0; r < rows_per_lane; r\+\+\) \{\n                                snapshot\[c \* S_v \+ r \* warp_size \+ lane\] = s_shard\[c\]\[r\];\n                            \}\n                        \})', src)
if m:
    old = m.group(1)
    new = old.replace("float * snapshot", "T_state * snapshot").replace("= s_shard[c][r];", "= gdn_from_float<T_state>(s_shard[c][r]);")
    src = src.replace(old, new)
else:
    # Try alternate pattern
    m = re.search(r'(float \* snapshot = state \+ target_slot \* state_slot_stride;)', src)
    if m:
        src = src.replace("float * snapshot = state + target_slot * state_slot_stride;", "T_state * snapshot = state + target_slot * state_slot_stride;")
        # Find and replace the write in the snapshot block
        src = re.sub(
            r'(snapshot\[c \* S_v \+ r \* warp_size \+ lane\] = )s_shard\[c\]\[r\](;)',
            r'\1gdn_from_float<T_state>(s_shard[c][r])\2',
            src
        )
    else:
        print("WARNING: Could not find tiled kernel snapshot pattern")

# Final state write in kernel 3
src = src.replace(
    "    if constexpr (!keep_rs_t) {\n#pragma unroll\n        for (int c = 0; c < COLS; c++) {\n#pragma unroll\n            for (int r = 0; r < rows_per_lane; r++) {\n                state[(col0 + colw + c) * S_v + r * warp_size + lane] = s_shard[c][r];\n            }\n        }\n    }\n}",
    "    if constexpr (!keep_rs_t) {\n#pragma unroll\n        for (int c = 0; c < COLS; c++) {\n#pragma unroll\n            for (int r = 0; r < rows_per_lane; r++) {\n                state[(col0 + colw + c) * S_v + r * warp_size + lane] = gdn_from_float<T_state>(s_shard[c][r]);\n            }\n        }\n    }\n}"
)

print("Phase 1 done: kernel template params + state pointer types")

# ============================================================
# PHASE 2: Launch function template params
# ============================================================

# launch_gated_delta_net_ilp_inst
src = src.replace(
    "template <typename T_in = float, bool keep_rs_t, int NC, bool PREFETCH>\nstatic void launch_gated_delta_net_ilp_inst(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const float * s_d,\n        T_in * dst_d, float * state_d,",
    "template <typename T_in = float, typename T_state = float, bool keep_rs_t, int NC, bool PREFETCH>\nstatic void launch_gated_delta_net_ilp_inst(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const T_state * s_d,\n        T_in * dst_d, T_state * state_d,"
)

src = src.replace(
    "    ggml_cuda_kernel_launch(gated_delta_net_cuda_ilp<T_in, S_v, keep_rs_t, NC, PREFETCH>, launch_params,",
    "    ggml_cuda_kernel_launch(gated_delta_net_cuda_ilp<T_in, T_state, S_v, keep_rs_t, NC, PREFETCH>, launch_params,"
)

# launch_gated_delta_net_ilp
src = src.replace(
    "template <typename T_in = float, bool keep_rs_t>\nstatic bool launch_gated_delta_net_ilp(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const float * s_d,\n        T_in * dst_d, float * state_d,",
    "template <typename T_in = float, typename T_state = float, bool keep_rs_t>\nstatic bool launch_gated_delta_net_ilp(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const T_state * s_d,\n        T_in * dst_d, T_state * state_d,"
)

src = src.replace(
    "    launch_gated_delta_net_ilp_inst<T_in, keep_rs_t, NC_, PF_>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \\",
    "    launch_gated_delta_net_ilp_inst<T_in, T_state, keep_rs_t, NC_, PF_>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \\"
)

# launch_gated_delta_net_tiled
src = src.replace(
    "template <typename T_in = float, bool keep_rs_t>\nstatic bool launch_gated_delta_net_tiled(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const float * s_d,\n        T_in * dst_d, float * state_d,",
    "template <typename T_in = float, typename T_state = float, bool keep_rs_t>\nstatic bool launch_gated_delta_net_tiled(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const T_state * s_d,\n        T_in * dst_d, T_state * state_d,"
)

src = src.replace(
    "    ggml_cuda_kernel_launch(gated_delta_net_tiled_cuda<T_in, S_v_, NUM_WARPS, COLS, TOKEN_TILE, keep_rs_t>, launch_params,",
    "    ggml_cuda_kernel_launch(gated_delta_net_tiled_cuda<T_in, T_state, S_v_, NUM_WARPS, COLS, TOKEN_TILE, keep_rs_t>, launch_params,"
)

# launch_gated_delta_net (original)
src = src.replace(
    "template <typename T_in = float, bool KDA, bool keep_rs_t, bool emit_ingredients_t>\nstatic void launch_gated_delta_net(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const float * s_d,\n        T_in * dst_d, float * state_d,",
    "template <typename T_in = float, typename T_state = float, bool KDA, bool keep_rs_t, bool emit_ingredients_t>\nstatic void launch_gated_delta_net(\n        const T_in * q_d, const T_in * k_d, const T_in * v_d,\n        const T_in * g_d, const T_in * b_d, const T_state * s_d,\n        T_in * dst_d, T_state * state_d,"
)

for sv in [16, 32, 64, 128]:
    src = src.replace(
        f"ggml_cuda_kernel_launch(gated_delta_net_cuda<T_in, {sv}, KDA, keep_rs_t, emit_ingredients_t>, launch_params,",
        f"ggml_cuda_kernel_launch(gated_delta_net_cuda<T_in, T_state, {sv}, KDA, keep_rs_t, emit_ingredients_t>, launch_params,"
    )

print("Phase 2 done: launch function template params")

# ============================================================
# PHASE 3: Rewrite ggml_cuda_op_gated_delta_net_impl
# ============================================================

# Find the start and end of the function
func_start_marker = "static void ggml_cuda_op_gated_delta_net_impl("
func_start = src.find(func_start_marker)
if func_start == -1:
    print("ERROR: Cannot find ggml_cuda_op_gated_delta_net_impl")
    sys.exit(1)

# Find the end of the function (next function or end of file)
# The function ends before "void ggml_cuda_op_gated_delta_net("
next_func = src.find("void ggml_cuda_op_gated_delta_net(", func_start)
if next_func == -1:
    print("ERROR: Cannot find end of ggml_cuda_op_gated_delta_net_impl")
    sys.exit(1)

# Extract the parameters/setup part (from function start to the F16 dispatch)
# I need to keep everything from the function start up to and including the state_src setup
# and the kda/keep_rs/emit_ingr detection, then replace the launch logic.

# Find where the actual launch logic starts (after all the setup)
# The setup ends with the state_src loop and the K/emit_mode detection
# Let me find the marker where the F16 dispatch starts
f16_dispatch_start = src.find("    // F16 path: detect type and launch appropriate kernel", func_start)
if f16_dispatch_start == -1:
    print("ERROR: Cannot find F16 dispatch start")
    sys.exit(1)

# Keep everything from function start to f16_dispatch_start (the setup code)
setup_code = src[func_start:f16_dispatch_start]

# Now I need to add the state type detection and the new dispatch logic
new_dispatch = """
    // Detect state type for T_state dispatch
    const bool state_f16 = (src_state->type == GGML_TYPE_F16);

    // F16 input path
    if (src_q->type == GGML_TYPE_F16) {
        const __half * q_d_f16 = (const __half *) src_q->data;
        const __half * k_d_f16 = (const __half *) src_k->data;
        const __half * v_d_f16 = (const __half *) src_v->data;
        const __half * g_d_f16 = (const __half *) src_g->data;
        const __half * b_d_f16 = (const __half *) src_beta->data;
        __half *       dst_d_f16 = (__half *) dst->data;

        // strides in F16 element units
        const int64_t sq1_f16 = nbq1 / sizeof(__half);
        const int64_t sq2_f16 = nbq2 / sizeof(__half);
        const int64_t sq3_f16 = nbq3 / sizeof(__half);
        const int64_t sv1_f16 = nbv1 / sizeof(__half);
        const int64_t sv2_f16 = nbv2 / sizeof(__half);
        const int64_t sv3_f16 = nbv3 / sizeof(__half);
        const int64_t sb1_f16 = nbb1 / sizeof(__half);
        const int64_t sb2_f16 = nbb2 / sizeof(__half);
        const int64_t sb3_f16 = nbb3 / sizeof(__half);

        const float scale_f16 = 1.0f / sqrtf((float) S_v);
        cudaStream_t stream_f16 = ctx.stream();

        const int  K_f16         = ggml_get_op_params_i32(dst, 0);
        const int  emit_mode_f16 = ggml_get_op_params_i32(dst, 1);
        const bool keep_rs_f16   = (K_f16 > 1) || (emit_mode_f16 != 0);
        const bool emit_ingr_f16 = (emit_mode_f16 == 1);

        // state pointers (type depends on state_f16)
        const void * s_base_f16 = state_src_base ? (const void*)state_src_base : (const void*)src_state->data;
        void *       state_out_f16 = (void*)(dst_d_f16 + S_v * H * n_tokens * n_seqs);
        int64_t state_slot_stride_f16 = emit_ingr_f16 ? (4 * S_v * H * n_seqs) : (S_v * S_v * H * n_seqs);
        if (cache != nullptr) {
            state_out_f16           = cache->data;
            state_slot_stride_f16 = cache->slot_stride;
        }

        if (state_f16) {
            const __half * s_d = (const __half *) s_base_f16;
            __half * state_d   = (__half *) state_out_f16;
            if (kda) {
                if (emit_ingr_f16) {
                    launch_gated_delta_net<__half, __half, true, true, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else if (keep_rs_f16) {
                    launch_gated_delta_net<__half, __half, true, true, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else {
                    launch_gated_delta_net<__half, __half, true, false, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                }
            } else {
                if (emit_ingr_f16) {
                    launch_gated_delta_net<__half, __half, false, true, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else if (keep_rs_f16) {
                    if (launch_gated_delta_net_tiled<__half, __half, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    if (launch_gated_delta_net_ilp<__half, __half, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    launch_gated_delta_net<__half, __half, false, true, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else {
                    if (launch_gated_delta_net_tiled<__half, __half, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    if (launch_gated_delta_net_ilp<__half, __half, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    launch_gated_delta_net<__half, __half, false, false, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                }
            }
        } else {
            const float * s_d = (const float *) s_base_f16;
            float * state_d   = (float *) state_out_f16;
            if (kda) {
                if (emit_ingr_f16) {
                    launch_gated_delta_net<__half, float, true, true, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else if (keep_rs_f16) {
                    launch_gated_delta_net<__half, float, true, true, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else {
                    launch_gated_delta_net<__half, float, true, false, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                }
            } else {
                if (emit_ingr_f16) {
                    launch_gated_delta_net<__half, float, false, true, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else if (keep_rs_f16) {
                    if (launch_gated_delta_net_tiled<__half, float, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    if (launch_gated_delta_net_ilp<__half, float, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    launch_gated_delta_net<__half, float, false, true, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                } else {
                    if (launch_gated_delta_net_tiled<__half, float, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    if (launch_gated_delta_net_ilp<__half, float, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                            S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                            sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src)) {
                        return;
                    }
                    launch_gated_delta_net<__half, float, false, false, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d, dst_d_f16, state_d,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16, state_src);
                }
            }
        }
        return;
    }

    // F32 input path
    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);
    cudaStream_t stream = ctx.stream();

    const int  K         = ggml_get_op_params_i32(dst, 0);
    const int  emit_mode = ggml_get_op_params_i32(dst, 1);
    const bool keep_rs   = (K > 1) || (emit_mode != 0);
    const bool emit_ingr = (emit_mode == 1);

    // state pointers
    const void * s_base = state_src_base ? (const void*)state_src_base : (const void*)src_state->data;
    void *       state_out = (void*)(dst_d + S_v * H * n_tokens * n_seqs);
    int64_t state_slot_stride = emit_ingr ? (4 * S_v * H * n_seqs) : (S_v * S_v * H * n_seqs);
    if (cache != nullptr) {
        state_out           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (state_f16) {
        const __half * s_d = (const __half *) s_base;
        __half * state_d   = (__half *) state_out;
        if (kda) {
            if (emit_ingr) {
                launch_gated_delta_net<float, __half, true, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else if (keep_rs) {
                launch_gated_delta_net<float, __half, true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else {
                launch_gated_delta_net<float, __half, true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            }
        } else {
            if (emit_ingr) {
                launch_gated_delta_net<float, __half, false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else if (keep_rs) {
                if (launch_gated_delta_net_tiled<float, __half, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                if (launch_gated_delta_net_ilp<float, __half, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                launch_gated_delta_net<float, __half, false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else {
                if (launch_gated_delta_net_tiled<float, __half, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                if (launch_gated_delta_net_ilp<float, __half, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                launch_gated_delta_net<float, __half, false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            }
        }
    } else {
        const float * s_d = (const float *) s_base;
        float * state_d   = (float *) state_out;
        if (kda) {
            if (emit_ingr) {
                launch_gated_delta_net<float, float, true, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else if (keep_rs) {
                launch_gated_delta_net<float, float, true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else {
                launch_gated_delta_net<float, float, true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            }
        } else {
            if (emit_ingr) {
                launch_gated_delta_net<float, float, false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else if (keep_rs) {
                if (launch_gated_delta_net_tiled<float, float, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                if (launch_gated_delta_net_ilp<float, float, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                launch_gated_delta_net<float, float, false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            } else {
                if (launch_gated_delta_net_tiled<float, float, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                if (launch_gated_delta_net_ilp<float, float, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src)) {
                    return;
                }
                launch_gated_delta_net<float, float, false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_src);
            }
        }
    }
}
"""

# Replace the old dispatch with the new one
src = src[:f16_dispatch_start] + new_dispatch + src[next_func:]

print("Phase 3 done: dispatch function rewritten")

# ============================================================
# Write the file
# ============================================================
with open(path, "w") as f:
    f.write(src)

print(f"\nDone! File: {path}")
print(f"Original size: {original_len} bytes")
print(f"New size: {len(src)} bytes")
print(f"Diff: +{len(src)-original_len} bytes")
