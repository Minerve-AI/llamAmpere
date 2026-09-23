#pragma once
// INT8-QK FlashAttention kernel for Ampere (sm_80+)
// Supports Q8_0 KV cache (K pre-quantized to INT8, V dequantized in-kernel)
//
// Strategy (HyperQwen / SageAttention):
//   - Q: quantized per-row to INT8 in-kernel
//   - K: pre-quantized to INT8 (from Q8_0 or FP16) via fattn-k-quant.cuh
//   - V: Q8_0 → dequantized to FP16 in-kernel (shared memory)
//   - QK^T: mma.m16n8k16.s8.s8.s32 (INT8 tensor cores, 2× FP16)
//   - Dequant: S_int32 * (q_scale * k_scale) → float
//   - Softmax: online softmax
//   - PV: FP16 (V dequantized from Q8_0)

#include "common.cuh"
#include "mma.cuh"
#include "ggml.h"

namespace ggml_cuda_fattn_i8qk {

    // =====================================================================
    // Config
    // =====================================================================
    constexpr int DKQ = 128;
    constexpr int DV  = 128;
    constexpr int NTHREADS = 128;
    constexpr int NQ = 8;    // queries per block
    constexpr int NK = 16;   // keys per block

    // Q8_0 block (matches ggml layout)
    struct __align__(16) block_q8_0_cuda {
        float d;
        int8_t qs[32];
    };

    // =====================================================================
    // INT8 pack helper
    // =====================================================================
    static __device__ __forceinline__ int pack_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
        int result;
        uint32_t * p = (uint32_t *)&result;
        *p = (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
             ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
        return result;
    }

    // =====================================================================
    // Kernel
    // =====================================================================
    //
    // Grid: (n_heads, n_q_blocks)
    // Block: 128 threads (4 warps)
    //
    // Q:  [n_heads, seq_q, DKQ/2] half2
    // K:  [n_kv_heads, seq_k, DKQ] int8 (pre-quantized)
    // K_scale: [n_tiles, n_kv_heads] float
    // V:  [n_kv_heads, seq_k, DV/32] block_q8_0  (Q8_0 format)
    // O:  [n_heads, seq_q, DV/2] half2
    //
    __global__ void __launch_bounds__(NTHREADS, 4)
    flash_attn_i8qk_kernel(
            const half2 * __restrict__ Q_h2,
            const int8_t * __restrict__ K_int8,
            const float * __restrict__ K_scale,
            const block_q8_0_cuda * __restrict__ V_q8,
            half2 * __restrict__ O_h2,
            const int seq_q, const int seq_k,
            const int n_heads, const int n_kv_heads,
            const float sm_scale) {

        const int head = blockIdx.x;
        const int q_block = blockIdx.y;
        const int q0 = q_block * NQ;
        const int kv_head = head / (n_heads / n_kv_heads);
        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid % 32;

        // Shared memory
        __shared__ int8_t  s_Q[NQ * DKQ];          // Q INT8 [8 × 128]
        __shared__ int8_t  s_K[DKQ * NK];          // K INT8 [128 × 16] transposed
        __shared__ half2   s_V[DKQ * NK];          // V FP16 [128 × 16] transposed (dequantized)
        __shared__ float   s_S[NQ * NK];           // Logits [8 × 16]
        __shared__ float   s_q_scale[NQ];
        __shared__ float   s_k_scale;
        __shared__ float   s_m[NQ];
        __shared__ float   s_l[NQ];
        __shared__ half2   s_O[NQ * DV/2];         // [8 × 64] half2
        __shared__ float   s_row_max[NQ];
        __shared__ float   s_row_sum[NQ];
        __shared__ float   s_alpha[NQ];

        // =================================================================
        // Phase 0: Load Q, compute per-row scale, quantize to INT8
        // =================================================================
        {
            const int n_half2 = DKQ / 2;
            const int r = tid / 16;
            const int c = (tid % 16) * 4; // 4 half2 per thread per row

            float q_vals[8];
            const int row = q0 + r;
            if (row < seq_q) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const half2 v = Q_h2[(int64_t)head * seq_q * n_half2 + row * n_half2 + c/2 + i];
                    const float2 f = __half22float2(v);
                    q_vals[i*2] = f.x;
                    q_vals[i*2+1] = f.y;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) q_vals[i] = 0.0f;
            }

            float local_amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) local_amax = fmaxf(local_amax, fabsf(q_vals[i]));

            __shared__ float s_amax_part[NQ * 16];
            s_amax_part[r * 16 + (tid % 16)] = local_amax;
            __syncthreads();

            if (tid % 16 == 0) {
                float m = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) m = fmaxf(m, s_amax_part[r * 16 + i]);
                s_q_scale[r] = m / 127.0f;
            }
            __syncthreads();

            const float inv_qs = (s_q_scale[r] > 1e-10f) ? (1.0f / s_q_scale[r]) : 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                s_Q[r * DKQ + c + i] = (int8_t)roundf(q_vals[i] * inv_qs);
            }
        }

        // Init
        if (tid < NQ) { s_m[tid] = -1e30f; s_l[tid] = 0.0f; }
        for (int i = tid; i < NQ * DV/2; i += NTHREADS) s_O[i] = __float2half2_rn(0.0f);
        __syncthreads();

        // =================================================================
        // Phase 1: Main loop
        // =================================================================
        const int n_kv_blocks = (seq_k + NK - 1) / NK;

        for (int kv_b = 0; kv_b < n_kv_blocks; ++kv_b) {
            const int k0 = kv_b * NK;
            const int n_keys = min(NK, seq_k - k0);

            // ---------------------------------------------------------
            // Load K tile (INT8): s_K[d * NK + k] = K_int8[k0+k, d]
            // ---------------------------------------------------------
            for (int i = tid; i < DKQ * NK; i += NTHREADS) {
                const int d = i / NK;
                const int k = i % NK;
                s_K[i] = (k < n_keys) ? K_int8[(int64_t)kv_head * seq_k * DKQ + (k0 + k) * DKQ + d] : (int8_t)0;
            }

            // ---------------------------------------------------------
            // Load V tile (Q8_0 → FP16): s_V[d * NK + k] = V[k0+k, d]
            // V layout: [n_kv_heads, seq_k, DV/32] block_q8_0
            // V_q8[h * seq_k * (DV/32) + s * (DV/32) + d/32].qs[d%32] * .d
            // ---------------------------------------------------------
            {
                const int nblocks_v = DV / 32; // 4 blocks per row
                // We need s_V[d * NK + k] for d=0..127, k=0..15
                // 128*16 = 2048 half2 values, 128 threads → 16 per thread
                for (int i = tid; i < DKQ * NK; i += NTHREADS) {
                    const int d = i / NK;
                    const int k = i % NK;
                    if (k < n_keys) {
                        const int v_idx = (int64_t)kv_head * seq_k * nblocks_v + (k0 + k) * nblocks_v + d / 32;
                        const block_q8_0_cuda blk = V_q8[v_idx];
                        s_V[d * NK + k] = __float2half_rn((float)blk.qs[d % 32] * blk.d);
                    } else {
                        s_V[d * NK + k] = __float2half_rn(0.0f);
                    }
                }
            }

            // Load K scale
            if (tid == 0) {
                s_k_scale = K_scale[(k0 / 64) * n_heads + kv_head];
            }
            __syncthreads();

            // ---------------------------------------------------------
            // QK^T via INT8 MMA (mma.m16n8k16.row.col.s32.s8.s8.s32)
            //
            // A[m][k] = Q_int8[m][d0+k]  (m=0..15 queries, k=0..15 dim)
            // B[k][n] = K_int8[n][d0+k]  (k=0..15 dim, n=0..7 keys)
            // C[m][n] = sum_k A[m][k]*B[k][n] = S_int32[m][n]
            //
            // Warp 0: keys 0..7, Warp 1: keys 8..15
            // 8 d-chunks × 1 MMA each = 8 MMA per warp
            // ---------------------------------------------------------
            if (warp_id < 2) {
                const int n_offset = warp_id * 8;
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

                #pragma unroll
                for (int d_chunk = 0; d_chunk < DKQ/16; ++d_chunk) {
                    const int d0 = d_chunk * 16;

                    // Load A: Q_int8 (16×16)
                    // Only 8 rows valid (NQ=8), rows 8..15 = 0
                    int a0, a1;
                    {
                        const int row = lane / 4;
                        const int col = (lane % 4) * 4;
                        if (lane < 16) {
                            // a0: rows 0..7
                            a0 = pack_i8(
                                s_Q[row * DKQ + d0 + col],
                                s_Q[row * DKQ + d0 + col + 1],
                                s_Q[row * DKQ + d0 + col + 2],
                                s_Q[row * DKQ + d0 + col + 3]);
                            a1 = 0; // rows 8..15 don't exist
                        } else {
                            // a1: rows 8..15 → we map to rows 0..7 (but these are zero)
                            // Actually: lanes 16-31 hold a1 for rows 8..15
                            // Since we only have 8 queries, a1 = 0
                            a0 = 0;
                            a1 = 0;
                        }
                    }

                    // Load B: K_int8 (16×8, col-major)
                    // B[k][n] = s_K[(d0+k) * NK + n_offset + n]
                    int b0;
                    {
                        const int col_b = lane / 4;       // n (0..7)
                        const int row_b = (lane % 4) * 4; // k offset (0,4,8,12)
                        b0 = pack_i8(
                            s_K[(d0 + row_b) * NK + n_offset + col_b],
                            s_K[(d0 + row_b + 1) * NK + n_offset + col_b],
                            s_K[(d0 + row_b + 2) * NK + n_offset + col_b],
                            s_K[(d0 + row_b + 3) * NK + n_offset + col_b]);
                    }

                    // INT8 MMA
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(b0));
                }

                // Store C to shared (only rows 0..7 are valid)
                {
                    const int row = lane / 4;
                    const int col = 2 * (lane % 4);
                    if (row < NQ) {
                        s_S[row * NK + n_offset + col]     = (float)c0;
                        s_S[row * NK + n_offset + col + 1] = (float)c2;
                    }
                }
            }
            __syncthreads();

            // ---------------------------------------------------------
            // Dequant + causal mask
            // ---------------------------------------------------------
            for (int i = tid; i < NQ * NK; i += NTHREADS) {
                const int r = i / NK;
                const int c = i % NK;
                float s_val = s_S[i] * (s_q_scale[r] * s_k_scale * sm_scale);
                if (k0 + c > q0 + r) s_val = -1e30f;
                s_S[i] = s_val;
            }
            __syncthreads();

            // ---------------------------------------------------------
            // Online softmax
            // ---------------------------------------------------------
            for (int r = 0; r < NQ; ++r) {
                if (tid >= r * 16 && tid < (r+1) * 16) {
                    float v = s_S[r * NK + (tid - r*16)];
                    #pragma unroll
                    for (int o = 8; o > 0; o /= 2)
                        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o, 16));
                    if ((tid - r*16) == 0) s_row_max[r] = v;
                }
            }
            __syncthreads();

            for (int r = tid; r < NQ; r += NTHREADS) {
                const float m_old = s_m[r];
                const float m_new = fmaxf(m_old, s_row_max[r]);
                s_m[r] = m_new;
                s_alpha[r] = (m_old > -1e29f) ? expf(m_old - m_new) : 0.0f;
            }
            __syncthreads();

            for (int i = tid; i < NQ * NK; i += NTHREADS) {
                const int r = i / NK;
                s_S[i] = (s_S[i] > -1e29f) ? expf(s_S[i] - s_m[r]) : 0.0f;
            }
            __syncthreads();

            for (int r = 0; r < NQ; ++r) {
                if (tid >= r * 16 && tid < (r+1) * 16) {
                    float v = s_S[r * NK + (tid - r*16)];
                    #pragma unroll
                    for (int o = 8; o > 0; o /= 2)
                        v += __shfl_xor_sync(0xffffffff, v, o, 16);
                    if ((tid - r*16) == 0) s_row_sum[r] = v;
                }
            }
            __syncthreads();

            for (int r = tid; r < NQ; r += NTHREADS) {
                s_l[r] = s_l[r] * s_alpha[r] + s_row_sum[r];
            }
            __syncthreads();

            // Rescale O
            for (int i = tid; i < NQ * DV/2; i += NTHREADS) {
                const int r = i / (DV/2);
                const float2 o = __half22float2(s_O[i]);
                s_O[i] = __float2half2_rn(make_float2(o.x * s_alpha[r], o.y * s_alpha[r]));
            }
            __syncthreads();

            // ---------------------------------------------------------
            // O += P × V (FP16 FMA)
            // P: s_S [8×16] float
            // V: s_V [128×16] half2 (transposed: s_V[d*NK + k] = V[k,d])
            // O: s_O [8×64] half2
            // ---------------------------------------------------------
            {
                const int r = tid / 16;
                const int d_base = (tid % 16) * 4;

                #pragma unroll
                for (int dd = 0; dd < 4; ++dd) {
                    const int d = d_base + dd;
                    float2 o_acc = __half22float2(s_O[r * (DV/2) + d]);

                    #pragma unroll
                    for (int c = 0; c < NK; ++c) {
                        const float p = s_S[r * NK + c];
                        if (p > 0.0f) {
                            const float2 v = __half22float2(s_V[d * NK + c]);
                            o_acc.x += p * v.x;
                            o_acc.y += p * v.y;
                        }
                    }

                    s_O[r * (DV/2) + d] = __float2half2_rn(o_acc);
                }
            }
            __syncthreads();
        }

        // =================================================================
        // Phase 2: Normalize + write
        // =================================================================
        for (int i = tid; i < NQ * DV/2; i += NTHREADS) {
            const int r = i / (DV/2);
            const float l = s_l[r];
            if (l > 1e-30f) {
                const float2 o = __half22float2(s_O[i]);
                s_O[i] = __float2half2_rn(make_float2(o.x / l, o.y / l));
            } else {
                s_O[i] = __float2half2_rn(0.0f);
            }
        }
        __syncthreads();

        {
            const int n_half2 = DV / 2;
            const int r = tid / 16;
            const int d_base = (tid % 16) * 4;
            const int row = q0 + r;
            if (row < seq_q) {
                #pragma unroll
                for (int dd = 0; dd < 4; ++dd) {
                    O_h2[(int64_t)head * seq_q * n_half2 + row * n_half2 + d_base + dd]
                        = s_O[r * n_half2 + d_base + dd];
                }
            }
        }
    }

    // =====================================================================
    // Host launcher
    // =====================================================================
    inline void launch_flash_attn_i8qk(
            const half2 * Q_h2,
            const int8_t * K_int8,
            const float * K_scale,
            const block_q8_0_cuda * V_q8,
            half2 * O_h2,
            int seq_q, int seq_k,
            int n_heads, int n_kv_heads,
            float sm_scale,
            cudaStream_t stream = 0) {
        dim3 grid(n_heads, (seq_q + NQ - 1) / NQ);
        dim3 block(NTHREADS);
        const size_t smem = NQ * DKQ + DKQ * NK + DKQ * NK * sizeof(half2)
                          + NQ * NK * sizeof(float) + NQ * sizeof(float) + sizeof(float)
                          + NQ * sizeof(float) + NQ * sizeof(float)
                          + NQ * (DV/2) * sizeof(half2)
                          + NQ * sizeof(float) + NQ * sizeof(float) + NQ * sizeof(float);
        flash_attn_i8qk_kernel<<<grid, block, smem, stream>>>(
            Q_h2, K_int8, K_scale, V_q8, O_h2,
            seq_q, seq_k, n_heads, n_kv_heads, sm_scale);
    }

} // namespace ggml_cuda_fattn_i8qk
