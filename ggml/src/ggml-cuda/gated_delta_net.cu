#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdlib>


// Type conversion helpers for F16 I/O
template <typename T>
inline __device__ float gdn_to_float(T x) { return x; }
inline __device__ float gdn_to_float(__half x) { return __half2float(x); }

template <typename T>
inline __device__ T gdn_from_float(float x) { return x; }
inline __device__ __half gdn_from_float(float x) { return __float2half(x); }

// emit_ingredients_t (only meaningful when keep_rs_t): write per-token (k,v,g,beta) ingredients
// instead of a full [S_v,S_v] state snapshot per retained slot, plus one fixed-cost trailing
// final-state block -- see ggml_gated_delta_net's emit_mode==1 contract (ggml.h) and the CPU
// reference (ggml-cpu/ops.cpp). Never instantiated with keep_rs_t == false.
template <typename T_in = float, int S_v = 128, bool KDA = false, bool keep_rs_t = false, bool emit_ingredients_t = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const T_in * q,
                                     const T_in * k,
                                     const T_in * v,
                                     const T_in * g,
                                     const T_in * beta,
                                     const float * curr_state,
                                     T_in *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    T_in *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output layout (per-slot stride passed in as state_slot_stride) — same per-(seq,head)
    // offset scheme as before, except emit_ingredients_t slots are 4*S_v wide (k,v,g,beta)
    // instead of S_v*S_v (a full state matrix); state_out points at the base of THIS
    // (seq,head)'s slot-0 storage either way. `state` (unoffset) is kept around separately so
    // the trailing final-state block (always S_v*S_v-shaped, at a fixed offset past all K
    // ingredient slots) can be addressed too.
    const int64_t state_in_offset = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float * state_out;
    if constexpr (emit_ingredients_t) {
        state_out = state + (sequence * H + h_idx) * (4 * S_v);
    } else {
        state_out = state + (sequence * H + h_idx) * S_v * S_v;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const T_in * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const T_in * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const T_in * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const T_in * beta_t = beta + gb_offset;
        const T_in * g_t     = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = gdn_to_float(*beta_t);

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = gdn_to_float(k_t[i]);
            q_reg[r] = gdn_to_float(q_t[i]);
        }

        if constexpr (!KDA) {
            const float g_val = expf(gdn_to_float(*g_t));

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (gdn_to_float(v_t[col]) - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = gdn_from_float<T_in>(attn_col * scale);
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(gdn_to_float(g_t[i])) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (gdn_to_float(v_t[col]) - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(gdn_to_float(g_t[i])) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = gdn_from_float<T_in>(attn_col * scale);
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // emit_ingredients_t==false: most-recent-first (slot 0 = final state), matching the
            // s_copy/rs_idx row-selection convention used elsewhere. emit_ingredients_t==true:
            // chronological (slot 0 = oldest of the K retained tokens) -- ingredients are only
            // ever consumed by this op's own replay call site, which needs a straight forward
            // prefix view, not a reversed one; see the CPU reference for the full rationale.
            const int target_slot = emit_ingredients_t ? ((int) t - ((int) n_tokens - K)) : ((int) n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                if constexpr (emit_ingredients_t) {
                    // ingredients: k (offset 0, full vector), v (offset S_v, this warp's column
                    // only), g (offset 2*S_v, raw/non-exponentiated), beta (offset 3*S_v). k/g/beta
                    // don't vary by column, so only one warp (col == 0) needs to write them --
                    // every warp doing so redundantly would cost O(S_v) writes each, i.e. O(S_v^2)
                    // total per component, which is what this design is supposed to avoid.
                    float * ingr_slot = state_out + (int64_t) target_slot * state_slot_stride;
                    if (col == 0) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            const int i    = r * warp_size + lane;
                            ingr_slot[i]   = k_reg[r];
                            if constexpr (KDA) {
                                ingr_slot[2 * S_v + i] = gdn_to_float(g_t[i]);
                            } else {
                                ingr_slot[2 * S_v + i] = gdn_to_float(*g_t);
                            }
                            ingr_slot[3 * S_v + i] = beta_val;
                        }
                    }
                    if (lane == 0) {
                        ingr_slot[S_v + col] = gdn_to_float(v_t[col]);
                    }
                } else {
                    float * curr_state = state_out + (int64_t) target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[col * S_v + i] = s_shard[r];
                    }
                }
            }

            if constexpr (emit_ingredients_t) {
                // state immediately before the K-token retained window starts, captured inline
                // (free relative to a second op call over the same prefix) when n_tokens > K.
                const int t_ckpt = (int) n_tokens - K - 1;
                if (t_ckpt >= 0 && t == t_ckpt) {
                    float * ckpt_slot = state + (int64_t) K * state_slot_stride +
                                         S_v * S_v * H * n_seqs + (sequence * H + h_idx) * S_v * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        ckpt_slot[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
    }

    if constexpr (emit_ingredients_t) {
        // fixed, once-per-call cost (not scaled by K): the true final state, for use as the next
        // decode's optimistic base state / this decode's own checkpoint update.
        float * final_slot = state + (int64_t) K * state_slot_stride + (sequence * H + h_idx) * S_v * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            final_slot[col * S_v + i] = s_shard[r];
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i              = r * warp_size + lane;
            state_out[col * S_v + i] = s_shard[r];
        }
    }
}


// same per-column math as gated_delta_net_cuda, but each warp owns NC state columns so the
// NC reduction chains interleave; multi-token, scalar-gate, S_v == 128 only (GGML_CUDA_SM86_GDN_COLS)
template <typename T_in = float, int S_v, bool keep_rs_t, int NC, bool PREFETCH>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda_ilp(const T_in * q,
                         const T_in * k,
                         const T_in * v,
                         const T_in * g,
                         const T_in * beta,
                         const float * curr_state,
                         T_in *       dst,
                         float *       state,
                         int64_t       H,
                         int64_t       n_tokens,
                         int64_t       n_seqs,
                         int64_t       sq1,
                         int64_t       sq2,
                         int64_t       sq3,
                         int64_t       sv1,
                         int64_t       sv2,
                         int64_t       sv3,
                         int64_t       sb1,
                         int64_t       sb2,
                         int64_t       sb3,
                         const uint3   neqk1_magic,
                         const uint3   rq3_magic,
                         float         scale,
                         int64_t       state_slot_stride,
                         int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * NC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset + col0 * S_v;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;

    float s_shard[NC][rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    const T_in * q_base = q + iq3 * sq3 + iq1 * sq1;
    const T_in * k_base = k + iq3 * sq3 + iq1 * sq1;
    const T_in * v_base = v + sequence * sv3 + h_idx * sv1 + col0;
    const T_in * b_base = beta + sequence * sb3 + h_idx * sb1;
    const T_in * g_base = g    + sequence * sb3 + h_idx * sb1;

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float v_reg[NC];
    float g_raw;
    float beta_val;

    auto load_tok = [&](int t, float * kr, float * qr, float * vr, float & gr, float & br) {
        const T_in * q_t = q_base + t * sq2;
        const T_in * k_t = k_base + t * sq2;
        const T_in * v_t = v_base + t * sv2;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kr[r] = gdn_to_float(k_t[r * warp_size + lane]);
            qr[r] = gdn_to_float(q_t[r * warp_size + lane]);
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            vr[c] = gdn_to_float(v_t[c]);
        }
        gr = gdn_to_float(g_base[t * sb2]);
        br = gdn_to_float(b_base[t * sb2]);
    };

    if constexpr (PREFETCH) {
        load_tok(0, k_reg, q_reg, v_reg, g_raw, beta_val);
    }

    for (int t = 0; t < n_tokens; t++) {
        float kn_reg[rows_per_lane];
        float qn_reg[rows_per_lane];
        float vn_reg[NC];
        float gn_raw = 0.0f;
        float bn_val = 0.0f;
        if constexpr (PREFETCH) {
            if (t + 1 < n_tokens) {
                load_tok(t + 1, kn_reg, qn_reg, vn_reg, gn_raw, bn_val);
            }
        } else {
            load_tok(t, k_reg, q_reg, v_reg, g_raw, beta_val);
        }

        const float g_val = expf(g_raw);

        // kv[col] = sum_i S[i][col] * k[i]   (same order as the original kernel, per column)
        float kv_shard[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_shard[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard[c] += s_shard[c][r] * k_reg[r];
            }
        }
        float kv_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_col[c] = warp_reduce_sum<warp_size>(kv_shard[c]);
        }

        float attn_partial[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float delta_col = (v_reg[c] - g_val * kv_col[c]) * beta_val;
            attn_partial[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r]    = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                attn_partial[c] += s_shard[c][r] * q_reg[r];
            }
        }
        float attn_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            attn_col[c] = warp_reduce_sum<warp_size>(attn_partial[c]);
        }

        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; c++) {
                attn_data[col0 + c] = gdn_from_float<T_in>(attn_col[c] * scale);
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * snap = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < NC; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        snap[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
                    }
                }
            }
        }

        if constexpr (PREFETCH) {
            if (t + 1 < n_tokens) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    k_reg[r] = kn_reg[r];
                    q_reg[r] = qn_reg[r];
                }
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    v_reg[c] = vn_reg[c];
                }
                g_raw    = gn_raw;
                beta_val = bn_val;
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

// default 4 (qualified on sm86); 1 selects the original kernel
static int ggml_cuda_sm86_gdn_cols() {
    static const int cols = [] {
        const char * s = getenv("GGML_CUDA_SM86_GDN_COLS");
        const int v = s ? atoi(s) : 4;
        return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 4;
    }();
    return cols;
}

static bool ggml_cuda_sm86_gdn_prefetch() {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_SM86_GDN_PREFETCH");
        return s != nullptr && atoi(s) != 0;
    }();
    return on;
}

template <typename T_in = float, bool keep_rs_t, int NC, bool PREFETCH>
static void launch_gated_delta_net_ilp_inst(
        const T_in * q_d, const T_in * k_d, const T_in * v_d,
        const T_in * g_d, const T_in * b_d, const float * s_d,
        T_in * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    constexpr int S_v = 128;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    GGML_ASSERT(S_v % (num_warps * NC) == 0);
    dim3 grid_dims(H, n_seqs, S_v / (num_warps * NC));
    dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda_ilp<T_in, S_v, keep_rs_t, NC, PREFETCH>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
}

// returns false when this path does not apply; caller falls back to the original kernel
template <typename T_in = float, bool keep_rs_t>
static bool launch_gated_delta_net_ilp(
        const T_in * q_d, const T_in * k_d, const T_in * v_d,
        const T_in * g_d, const T_in * b_d, const float * s_d,
        T_in * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int  nc       = ggml_cuda_sm86_gdn_cols();
    const bool prefetch = ggml_cuda_sm86_gdn_prefetch();
    if (S_v != 128 || n_tokens < 2 || (nc == 1 && !prefetch)) {
        return false;
    }
#define GDN_ILP_LAUNCH(NC_, PF_) \
    launch_gated_delta_net_ilp_inst<T_in, keep_rs_t, NC_, PF_>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream)
    if (nc == 8) {
        if (prefetch) { GDN_ILP_LAUNCH(8, true); } else { GDN_ILP_LAUNCH(8, false); }
    } else if (nc == 4) {
        if (prefetch) { GDN_ILP_LAUNCH(4, true); } else { GDN_ILP_LAUNCH(4, false); }
    } else if (nc == 2) {
        if (prefetch) { GDN_ILP_LAUNCH(2, true); } else { GDN_ILP_LAUNCH(2, false); }
    } else {
        GDN_ILP_LAUNCH(1, true);
    }
#undef GDN_ILP_LAUNCH
    return true;
}

template <typename T_in = float, bool KDA, bool keep_rs_t, bool emit_ingredients_t>
static void launch_gated_delta_net(
        const T_in * q_d, const T_in * k_d, const T_in * v_d,
        const T_in * g_d, const T_in * b_d, const float * s_d,
        T_in * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<T_in, 16, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<T_in, 32, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<T_in, 64, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<T_in, 128, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    // F16 path: detect type and launch appropriate kernel
    if (src_q->type == GGML_TYPE_F16) {
        const __half * q_d_f16 = (const __half *) src_q->data;
        const __half * k_d_f16 = (const __half *) src_k->data;
        const __half * v_d_f16 = (const __half *) src_v->data;
        const __half * g_d_f16 = (const __half *) src_g->data;
        const __half * b_d_f16 = (const __half *) src_beta->data;
        const float  * s_d_f16 = (const float *)  src_state->data;
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

        // state offset: dst buffer is F16, state is F32 after the attention scores
        float * state_d_f16 = (float *) (dst_d_f16 + S_v * H * n_tokens * n_seqs);
        int64_t state_slot_stride_f16 = emit_ingr_f16 ? (4 * S_v * H * n_seqs) : (S_v * S_v * H * n_seqs);
        if (cache != nullptr) {
            state_d_f16           = cache->data;
            state_slot_stride_f16 = cache->slot_stride;
        }

        // launch F16 kernel
        if (kda) {
            if (emit_ingr_f16) {
                launch_gated_delta_net<__half, true, true, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                    S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                    sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16);
            } else if (keep_rs_f16) {
                launch_gated_delta_net<__half, true, true, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                    S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                    sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16);
            } else {
                launch_gated_delta_net<__half, true, false, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                    S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                    sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16);
            }
        } else {
            if (emit_ingr_f16) {
                launch_gated_delta_net<__half, false, true, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                    S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                    sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16);
            } else if (keep_rs_f16) {
                if (launch_gated_delta_net_ilp<__half, true>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16)) {
                    return;
                }
                launch_gated_delta_net<__half, false, true, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                    S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                    sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16);
            } else {
                if (launch_gated_delta_net_ilp<__half, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                        S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                        sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16)) {
                    return;
                }
                launch_gated_delta_net<__half, false, false, false>(q_d_f16, k_d_f16, v_d_f16, g_d_f16, b_d_f16, s_d_f16, dst_d_f16, state_d_f16,
                    S_v, H, n_tokens, n_seqs, sq1_f16, sq2_f16, sq3_f16, sv1_f16, sv2_f16, sv3_f16,
                    sb1_f16, sb2_f16, sb3_f16, neqk1, rq3, scale_f16, state_slot_stride_f16, K_f16, stream_f16);
            }
        }
        return;
    }

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
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

    // K (snapshot slot count) and emit_mode (0=full snapshots, 1=ingredients) are op params;
    // state holds s0 only [S_v, S_v, H, n_seqs].
    const int  K         = ggml_get_op_params_i32(dst, 0);
    const int  emit_mode = ggml_get_op_params_i32(dst, 1);
    const bool keep_rs   = (K > 1) || (emit_mode != 0);
    const bool emit_ingr = (emit_mode == 1);

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing.
    // emit_ingr slots are 4*S_v wide (k,v,g,beta) instead of S_v*S_v (a full state matrix);
    // the fusion path (external cache) never engages for emit_mode==1 (see
    // ggml_cuda_try_gdn_cache_fusion's explicit guard), so cache->slot_stride is always the
    // emit_mode==0 layout when cache != nullptr.
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = emit_ingr ? (4 * S_v * H * n_seqs) : (S_v * S_v * H * n_seqs);
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (emit_ingr) {
            launch_gated_delta_net<float, true, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else if (keep_rs) {
            launch_gated_delta_net<float, true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<float, true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (emit_ingr) {
            launch_gated_delta_net<float, false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else if (keep_rs) {
            if (launch_gated_delta_net_ilp<float, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream)) {
                return;
            }
            launch_gated_delta_net<float, false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            if (launch_gated_delta_net_ilp<float, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream)) {
                return;
            }
            launch_gated_delta_net<float, false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
