#pragma once
// INT8-QK FlashAttention kernel for Ampere (sm_86)
// HyperQwen/SageAttention-style: INT8 QK^T + FP16 PV on tensor cores
//
// Strategy:
//   - Q: quantized per-row to INT8 in-kernel (absmax scale)
//   - K: pre-quantized to INT8 (mean-subtracted, per 64-key tile scale)
//   - V: Q8_0 → dequantized to FP16 in-kernel (shared memory)
//   - QK^T: mma.m16n8k16.row.col.s32.s8.s8.s32 (INT8 tensor cores)
//   - PV:   mma.m16n8k16.row.col.f32.f16.f16.f32 (FP16 tensor cores)
//   - Online softmax between QK^T and PV
//
// Block: 256 threads (8 warps), NQ=16 queries, NK=64 keys
// Grid: (n_heads, ceil(seq_q / NQ))

#include "common.cuh"
#include "mma.cuh"
#include "ggml.h"

namespace ggml_cuda_fattn_i8qk {

    constexpr int DKQ = 128;
    constexpr int DV  = 128;
    constexpr int NTHREADS = 256;
    constexpr int NQ = 16;   // queries per block
    constexpr int NK = 64;   // keys per block
    constexpr int NWARPS = NTHREADS / 32; // 8

    struct __align__(16) block_q8_0_cuda {
        float d;
        int8_t qs[32];
    };

    // Pack 4 int8 into one .b32
    static __device__ __forceinline__ int pack_i4(int8_t a, int8_t b, int8_t c, int8_t d) {
        int r;
        uint32_t * p = (uint32_t *)&r;
        *p = (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
             ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
        return r;
    }

    // Pack 2 half into one .b32 (for MMA A/B operands)
    static __device__ __forceinline__ int pack_h2(half lo, half hi) {
        int r;
        uint32_t * p = (uint32_t *)&r;
        *p = (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
        return r;
    }

    // =====================================================================
    // Kernel
    // =====================================================================
    //
    // Q:   [n_heads, seq_q, DKQ/2] half2
    // K:   [n_kv_heads, seq_k, DKQ] int8 (pre-quantized, mean-subtracted)
    // Ks:  [n_tiles, n_kv_heads] float (per 64-key tile scale)
    // V:   [n_kv_heads, seq_k, DV/32] block_q8_0
    // O:   [n_heads, seq_q, DV/2] half2
    //
    __global__ void __launch_bounds__(NTHREADS, 2)
    flash_attn_i8qk_kernel(
            const half2 * __restrict__ Q_h2,
            const int8_t * __restrict__ K_int8,
            const float * __restrict__ K_scale,
            const block_q8_0_cuda * __restrict__ V_q8,
            half2 * __restrict__ O_h2,
            const int seq_q, const int seq_k,
            const int n_heads, const int n_kv_heads,
            const float sm_scale) {

        const int head    = blockIdx.x;
        const int q_block = blockIdx.y;
        const int q0      = q_block * NQ;
        const int n_gqa   = n_heads / n_kv_heads;
        const int kv_head = head / n_gqa;
        const int tid     = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane    = tid % 32;

        // =================================================================
        // Shared memory
        // =================================================================
        __shared__ int8_t  s_Q[NQ * DKQ];            // [16][128] INT8
        __shared__ int8_t  s_K[DKQ * NK];            // [128][64] INT8 (dim-major)
        __shared__ half    s_V[NK * DV];             // [64][128] FP16 (key-major)
        __shared__ float   s_S[NQ * NK];             // [16][64] logits/softmax
        __shared__ half    s_P[NQ * NK];             // [16][64] P for PV MMA
        __shared__ float   s_q_scale[NQ];
        __shared__ float   s_k_scale;
        __shared__ float   s_m[NQ];
        __shared__ float   s_l[NQ];
        __shared__ float   s_alpha[NQ];
        __shared__ float   s_O[NQ * DV];             // [16][128] FP32 accumulator

        // =================================================================
        // Phase 0: Load Q, quantize to INT8
        // =================================================================
        {
            const int n_h2 = DKQ / 2;
            // 16 rows × 64 half2 = 1024 half2, 256 threads → 4 per thread
            for (int i = tid; i < NQ * n_h2; i += NTHREADS) {
                const int r = i / n_h2;
                const int c = i % n_h2;
                const int row = q0 + r;
                float2 v;
                if (row < seq_q) {
                    v = __half22float2(Q_h2[(int64_t)head * seq_q * n_h2 + row * n_h2 + c]);
                } else {
                    v = make_float2(0.0f, 0.0f);
                }
                // Store to a temp area in s_O (reuse before init)
                // Actually, let's store directly and compute amax
                s_O[r * DV + c * 2]     = v.x;
                s_O[r * DV + c * 2 + 1] = v.y;
            }
            __syncthreads();

            // Compute per-row absmax: 16 rows, each 128 values
            // 256 threads: 16 threads per row
            {
                const int r = tid / 16;
                const int c = (tid % 16) * 8; // 8 values per thread
                float local_max = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    local_max = fmaxf(local_max, fabsf(s_O[r * DV + c + i]));
                }
                // Warp-level reduce within the 16-thread group
                #pragma unroll
                for (int o = 8; o > 0; o /= 2)
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o, 16));
                if ((tid % 16) == 0) {
                    s_q_scale[r] = fmaxf(local_max, 1e-10f) / 127.0f;
                }
            }
            __syncthreads();

            // Quantize to INT8
            for (int i = tid; i < NQ * DKQ; i += NTHREADS) {
                const int r = i / DKQ;
                const int d = i % DKQ;
                const float inv = 1.0f / s_q_scale[r];
                s_Q[r * DKQ + d] = (int8_t)roundf(s_O[r * DV + d] * inv);
            }
            // Init s_O to zero (it was used as temp)
            for (int i = tid; i < NQ * DV; i += NTHREADS) s_O[i] = 0.0f;
        }

        // Init online softmax state
        if (tid < NQ) { s_m[tid] = -1e30f; s_l[tid] = 0.0f; s_alpha[tid] = 1.0f; }
        __syncthreads();

        // =================================================================
        // Phase 1: Main loop over KV blocks
        // =================================================================
        const int n_kv_blocks = (seq_k + NK - 1) / NK;

        for (int kv_b = 0; kv_b < n_kv_blocks; ++kv_b) {
            const int k0 = kv_b * NK;
            const int n_keys = min(NK, seq_k - k0);

            // ---------------------------------------------------------
            // Load K tile: s_K[d * NK + k] = K_int8[kv_head, k0+k, d]
            // 128 × 64 = 8192 int8, 256 threads → 32 per thread
            // ---------------------------------------------------------
            for (int i = tid; i < DKQ * NK; i += NTHREADS) {
                const int d = i / NK;
                const int k = i % NK;
                s_K[i] = (k < n_keys) ?
                    K_int8[(int64_t)kv_head * seq_k * DKQ + (k0 + k) * DKQ + d] : (int8_t)0;
            }

            // ---------------------------------------------------------
            // Load V tile: s_V[k * DV + d] = dequant(V_q8[kv_head, k0+k, d])
            // 64 × 128 = 8192 half, 256 threads → 32 per thread
            // V_q8 layout: [kv_head, seq, DV/32] blocks of 32
            // ---------------------------------------------------------
            {
                const int nblk = DV / 32; // 4 blocks per row
                for (int i = tid; i < NK * DV; i += NTHREADS) {
                    const int k = i / DV;
                    const int d = i % DV;
                    if (k < n_keys) {
                        const int vidx = (int64_t)kv_head * seq_k * nblk + (k0 + k) * nblk + d / 32;
                        const block_q8_0_cuda blk = V_q8[vidx];
                        s_V[k * DV + d] = __float2half((float)blk.qs[d % 32] * blk.d);
                    } else {
                        s_V[k * DV + d] = __float2half(0.0f);
                    }
                }
            }

            // Load K scale (one per 64-key tile)
            if (tid == 0) {
                s_k_scale = K_scale[(k0 / 64) * n_kv_heads + kv_head];
            }
            __syncthreads();

            // ---------------------------------------------------------
            // QK^T via INT8 MMA: S[16][64] = Q[16][128] × K[128][64]
            //
            // mma.m16n8k16.row.col.s32.s8.s8.s32
            // A[16×16] = Q_int8[m][d0+k], B[16×8] = K_int8[n][d0+k]
            // C[16×8] = S[m][n]
            //
            // 8 warps: warp w → n-chunk w (keys w*8..w*8+7), 8 k-chunks
            // ---------------------------------------------------------
            {
                const int n_off = warp_id * 8; // this warp's 8 keys
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

                #pragma unroll
                for (int d_chunk = 0; d_chunk < DKQ / 16; ++d_chunk) {
                    const int d0 = d_chunk * 16;

                    // A: 2 .b32
                    // a0 = Q[t/4][d0 + 4*(t%4) .. +3]
                    // a1 = Q[t/4+8][d0 + 4*(t%4) .. +3]
                    const int a_row0 = lane / 4;
                    const int a_row1 = a_row0 + 8;
                    const int a_col  = 4 * (lane % 4);
                    int a0 = pack_i4(
                        s_Q[a_row0 * DKQ + d0 + a_col],
                        s_Q[a_row0 * DKQ + d0 + a_col + 1],
                        s_Q[a_row0 * DKQ + d0 + a_col + 2],
                        s_Q[a_row0 * DKQ + d0 + a_col + 3]);
                    int a1 = pack_i4(
                        s_Q[a_row1 * DKQ + d0 + a_col],
                        s_Q[a_row1 * DKQ + d0 + a_col + 1],
                        s_Q[a_row1 * DKQ + d0 + a_col + 2],
                        s_Q[a_row1 * DKQ + d0 + a_col + 3]);

                    // B: 1 .b32
                    // b0 = K[n0 + 4*(t%4)][d0 + t/4] ... K[n0 + 4*(t%4)+3][d0 + t/4]
                    // B[k][n] = K_int8[n][d] → s_K[d * NK + n]
                    // b0 = {B[4*(lane%4)][lane/4], B[4*(lane%4)+1][lane/4],
                    //        B[4*(lane%4)+2][lane/4], B[4*(lane%4)+3][lane/4]}
                    //     = {s_K[(d0+4*(lane%4)) * NK + n_off + lane/4],
                    //        s_K[(d0+4*(lane%4)+1) * NK + n_off + lane/4],
                    //        s_K[(d0+4*(lane%4)+2) * NK + n_off + lane/4],
                    //        s_K[(d0+4*(lane%4)+3) * NK + n_off + lane/4]}
                    const int b_row = 4 * (lane % 4);
                    const int b_col = lane / 4;
                    int b0 = pack_i4(
                        s_K[(d0 + b_row)     * NK + n_off + b_col],
                        s_K[(d0 + b_row + 1) * NK + n_off + b_col],
                        s_K[(d0 + b_row + 2) * NK + n_off + b_col],
                        s_K[(d0 + b_row + 3) * NK + n_off + b_col]);

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(b0));
                }

                // Store C: c0=C[t/4][2*(t%4)], c1=C[t/4+8][2*(t%4)],
                //          c2=C[t/4][2*(t%4)+1], c3=C[t/4+8][2*(t%4)+1]
                {
                    const int r0 = lane / 4;
                    const int r1 = r0 + 8;
                    const int c  = 2 * (lane % 4);
                    s_S[r0 * NK + n_off + c]     = (float)c0;
                    s_S[r1 * NK + n_off + c]     = (float)c1;
                    s_S[r0 * NK + n_off + c + 1] = (float)c2;
                    s_S[r1 * NK + n_off + c + 1] = (float)c3;
                }
            }
            __syncthreads();

            // ---------------------------------------------------------
            // Dequant + causal mask
            // S[r][c] = S_int32 * q_scale[r] * k_scale * sm_scale
            // ---------------------------------------------------------
            for (int i = tid; i < NQ * NK; i += NTHREADS) {
                const int r = i / NK;
                const int c = i % NK;
                float val = s_S[i] * (s_q_scale[r] * s_k_scale * sm_scale);
                if (k0 + c > q0 + r) val = -1e30f;
                s_S[i] = val;
            }
            __syncthreads();

            // ---------------------------------------------------------
            // Online softmax
            // ---------------------------------------------------------
            {
                // Row max: 16 rows × 64 cols, 256 threads → 16 threads/row
                const int r = tid / 16;
                const int c = (tid % 16) * 4;
                float row_max = -1e30f;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    row_max = fmaxf(row_max, s_S[r * NK + c + i]);
                #pragma unroll
                for (int o = 8; o > 0; o /= 2)
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, o, 16));
                if ((tid % 16) == 0) {
                    const float m_old = s_m[r];
                    const float m_new = fmaxf(m_old, row_max);
                    s_m[r] = m_new;
                    s_alpha[r] = (m_old > -1e29f) ? expf(m_old - m_new) : 0.0f;
                }
            }
            __syncthreads();

            // Exp + row sum
            for (int i = tid; i < NQ * NK; i += NTHREADS) {
                const int r = i / NK;
                float v = s_S[i];
                v = (v > -1e29f) ? expf(v - s_m[r]) : 0.0f;
                s_S[i] = v;
                s_P[i] = __float2half(v); // store as half for PV MMA
            }
            __syncthreads();

            // Row sum reduction
            {
                const int r = tid / 16;
                const int c = (tid % 16) * 4;
                float row_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    row_sum += s_S[r * NK + c + i];
                #pragma unroll
                for (int o = 8; o > 0; o /= 2)
                    row_sum += __shfl_xor_sync(0xffffffff, row_sum, o, 16);
                if ((tid % 16) == 0) {
                    s_l[r] = s_l[r] * s_alpha[r] + row_sum;
                }
            }
            __syncthreads();

            // Rescale O accumulator
            for (int i = tid; i < NQ * DV; i += NTHREADS) {
                const int r = i / DV;
                s_O[i] *= s_alpha[r];
            }
            __syncthreads();

            // ---------------------------------------------------------
            // PV via FP16 MMA: O[16][128] += P[16][64] × V[64][128]
            //
            // mma.m16n8k16.row.col.f32.f16.f16.f32
            // A[16×16] = P[m][k0+k], B[16×8] = V[k0+k][d0+n]
            // C[16×8] = O[m][d0+n]
            //
            // 8 warps: warp w → 2 n-chunks (dims w*16..w*16+15), 4 k-chunks
            // ---------------------------------------------------------
            {
                const int d_base = warp_id * 16; // this warp's 16 dims

                #pragma unroll
                for (int k_chunk = 0; k_chunk < NK / 16; ++k_chunk) {
                    const int k0m = k_chunk * 16;

                    #pragma unroll
                    for (int d_sub = 0; d_sub < 2; ++d_sub) {
                        const int d0 = d_base + d_sub * 8;
                        float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

                        // A: P[m][k] as half2 pairs → 4 .b32
                        // a0 = P[t/4][4*(t%4) .. +1] as half2
                        // a1 = P[t/4+8][4*(t%4) .. +1] as half2
                        // a2 = P[t/4][4*(t%4)+8 .. +9] as half2
                        // a3 = P[t/4+8][4*(t%4)+8 .. +9] as half2
                        const int a_r0 = lane / 4;
                        const int a_r1 = a_r0 + 8;
                        const int a_c  = 2 * (lane % 4);  // f16: 2 half per .b32
                        int a0 = pack_h2(s_P[a_r0 * NK + k0m + a_c],     s_P[a_r0 * NK + k0m + a_c + 1]);
                        int a1 = pack_h2(s_P[a_r1 * NK + k0m + a_c],     s_P[a_r1 * NK + k0m + a_c + 1]);
                        int a2 = pack_h2(s_P[a_r0 * NK + k0m + a_c + 8], s_P[a_r0 * NK + k0m + a_c + 9]);
                        int a3 = pack_h2(s_P[a_r1 * NK + k0m + a_c + 8], s_P[a_r1 * NK + k0m + a_c + 9]);

                        // B: V[k][d] as half2 pairs → 2 .b32
                        // b0 = V[2*(t%4)][t/4] and V[2*(t%4)+1][t/4] as half2
                        // b1 = V[2*(t%4)+8][t/4] and V[2*(t%4)+9][t/4] as half2
                        const int b_r = 2 * (lane % 4);
                        const int b_c = lane / 4;
                        int b0 = pack_h2(s_V[(k0m + b_r)     * DV + d0 + b_c],
                                         s_V[(k0m + b_r + 1) * DV + d0 + b_c]);
                        int b1 = pack_h2(s_V[(k0m + b_r + 8) * DV + d0 + b_c],
                                         s_V[(k0m + b_r + 9) * DV + d0 + b_c]);

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

                        // Accumulate into s_O
                        const int o_r0 = lane / 4;
                        const int o_r1 = o_r0 + 8;
                        const int o_c  = 2 * (lane % 4);
                        s_O[o_r0 * DV + d0 + o_c]     += c0;
                        s_O[o_r1 * DV + d0 + o_c]     += c1;
                        s_O[o_r0 * DV + d0 + o_c + 1] += c2;
                        s_O[o_r1 * DV + d0 + o_c + 1] += c3;
                    }
                }
            }
            __syncthreads();
        }

        // =================================================================
        // Phase 2: Normalize + write output
        // =================================================================
        for (int i = tid; i < NQ * DV; i += NTHREADS) {
            const int r = i / DV;
            const float l = s_l[r];
            s_O[i] = (l > 1e-30f) ? s_O[i] / l : 0.0f;
        }
        __syncthreads();

        {
            const int n_h2 = DV / 2;
            for (int i = tid; i < NQ * n_h2; i += NTHREADS) {
                const int r = i / n_h2;
                const int c = i % n_h2;
                const int row = q0 + r;
                if (row < seq_q) {
                    const float lo = s_O[r * DV + c * 2];
                    const float hi = s_O[r * DV + c * 2 + 1];
                    O_h2[(int64_t)head * seq_q * n_h2 + row * n_h2 + c] =
                        __float2half2_rn(make_float2(lo, hi));
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
        flash_attn_i8qk_kernel<<<grid, block, 0, stream>>>(
            Q_h2, K_int8, K_scale, V_q8, O_h2,
            seq_q, seq_k, n_heads, n_kv_heads, sm_scale);
    }

} // namespace ggml_cuda_fattn_i8qk
