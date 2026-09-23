#pragma once
// INT8-QK FlashAttention kernel for Ampere (sm_80+)
//
// Strategy (HyperQwen / SageAttention):
//   - Q: quantized per-row to INT8 in-kernel
//   - K: pre-quantized to INT8 per 64-key tile with mean subtraction
//   - QK^T: mma.m16n8k16.s8.s8.s32 (INT8 tensor cores)
//   - Dequant: S_int32 * (q_scale * k_scale) → float
//   - Softmax: online softmax
//   - PV: FP16 MMA (standard FlashAttention)
//
// Mean subtraction is softmax-invariant (constant offset per query).

#include "common.cuh"
#include "mma.cuh"

namespace ggml_cuda_fattn_i8qk {

    // =====================================================================
    // Config: D=128, standard llama.cpp tiling
    // =====================================================================
    constexpr int DKQ = 128;
    constexpr int DV  = 128;
    constexpr int NTHREADS = 128;
    constexpr int NWARPS = 4;
    constexpr int NQ = 8;    // queries per block (NCOLS1)
    constexpr int NK = 16;   // keys per block (NCOLS2)

    // =====================================================================
    // INT8 pack/unpack helpers
    // =====================================================================
    // Pack 4 int8 values into one int32 (little-endian)
    static __device__ __forceinline__ int pack_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
        int result;
        uint32_t * p = (uint32_t *)&result;
        *p = (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
             ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
        return result;
    }

    // =====================================================================
    // Load INT8 A tile (16×16) from shared memory into MMA register layout
    // s_A[m * stride + k] = A[m][k]
    // Returns: 2 int32 registers (a0, a1)
    // =====================================================================
    static __device__ __forceinline__ void load_A_i8(
            int & a0, int & a1,
            const int8_t * __restrict__ s_A, int stride,
            int d0) {
        const int lane = threadIdx.x % 32;
        const int row = lane / 4;
        const int col = (lane % 4) * 4;

        // a0: A[row][d0+col .. d0+col+3]
        a0 = pack_i8(
            s_A[row * stride + d0 + col],
            s_A[row * stride + d0 + col + 1],
            s_A[row * stride + d0 + col + 2],
            s_A[row * stride + d0 + col + 3]);

        // a1: A[row+8][d0+col .. d0+col+3]
        a1 = pack_i8(
            s_A[(row + 8) * stride + d0 + col],
            s_A[(row + 8) * stride + d0 + col + 1],
            s_A[(row + 8) * stride + d0 + col + 2],
            s_A[(row + 8) * stride + d0 + col + 3]);
    }

    // =====================================================================
    // Load INT8 B tile (16×8, col-major) from shared memory into MMA register
    // B[k][n] = s_B[(d0+k) * stride + n]  (K^T layout)
    // Returns: 1 int32 register (b0)
    // =====================================================================
    static __device__ __forceinline__ void load_B_i8(
            int & b0,
            const int8_t * __restrict__ s_B, int stride,
            int d0, int n_offset) {
        const int lane = threadIdx.x % 32;
        const int col = lane / 4;       // n dimension (0..7)
        const int row = (lane % 4) * 4; // k dimension offset (0,4,8,12)

        // b0: B[row..row+3][col] = s_B[(d0+row)*stride + col+n_offset], ...
        b0 = pack_i8(
            s_B[(d0 + row) * stride + col + n_offset],
            s_B[(d0 + row + 1) * stride + col + n_offset],
            s_B[(d0 + row + 2) * stride + col + n_offset],
            s_B[(d0 + row + 3) * stride + col + n_offset]);
    }

    // =====================================================================
    // Store INT32 C tile (16×8) from MMA registers to shared memory
    // s_C[m * stride + n] = C[m][n]
    // =====================================================================
    static __device__ __forceinline__ void store_C_i32(
            float * __restrict__ s_C, int stride,
            int c0, int c1, int c2, int c3,
            float scale, int m_offset, int n_offset) {
        const int lane = threadIdx.x % 32;
        const int row = lane / 4;
        const int col = 2 * (lane % 4);

        // c0: C[row][col], c1: C[row+8][col], c2: C[row][col+1], c3: C[row+8][col+1]
        s_C[(m_offset + row) * stride + n_offset + col]       = (float)c0 * scale;
        s_C[(m_offset + row + 8) * stride + n_offset + col]   = (float)c1 * scale;
        s_C[(m_offset + row) * stride + n_offset + col + 1]   = (float)c2 * scale;
        s_C[(m_offset + row + 8) * stride + n_offset + col + 1] = (float)c3 * scale;
    }

    // =====================================================================
    // Kernel
    // =====================================================================
    __global__ void __launch_bounds__(NTHREADS, 4)
    flash_attn_i8qk_kernel(
            const half2 * __restrict__ Q_h2,      // [n_heads, seq_q, DKQ/2]
            const int8_t * __restrict__ K_int8,    // [n_kv_heads, seq_k, DKQ]
            const float * __restrict__ K_scale,    // [n_tiles, n_kv_heads]
            const float * __restrict__ K_mean,     // [n_kv_heads, DKQ]
            const half2 * __restrict__ V_h2,       // [n_kv_heads, seq_k, DV/2]
            half2 * __restrict__ O_h2,             // [n_heads, seq_q, DV/2]
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
        __shared__ int8_t  s_K[DKQ * NK];          // K INT8 [128 × 16] (transposed)
        __shared__ half2   s_V[DKQ * NK];          // V FP16 [128 × 16] (transposed)
        __shared__ float   s_S[NQ * NK];           // Logits [8 × 16]
        __shared__ float   s_q_scale[NQ];
        __shared__ float   s_k_scale;
        __shared__ float   s_m[NQ];                // Running max
        __shared__ float   s_l[NQ];                // Running sum
        __shared__ half2   s_O[NQ * DV/2];         // Output accumulator [8 × 64]
        __shared__ float   s_row_max[NQ];
        __shared__ float   s_row_sum[NQ];
        __shared__ float   s_alpha[NQ];

        // =================================================================
        // Phase 0: Load Q, compute per-row scale, quantize to INT8
        // =================================================================
        {
            const int n_half2 = DKQ / 2; // 64
            // 8 rows × 64 half2 = 512 half2 = 1024 float values
            // 128 threads: 8 float per thread
            // Thread (r, c): r = tid/16, c = (tid%16)*4 → handles float[c..c+3] of row r

            const int r = tid / 16;
            const int c = (tid % 16) * 4;

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

            // Compute local absmax
            float local_amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) local_amax = fmaxf(local_amax, fabsf(q_vals[i]));

            // Reduce across 16 threads per row
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

            // Quantize and store
            const float inv_qs = (s_q_scale[r] > 1e-10f) ? (1.0f / s_q_scale[r]) : 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                s_Q[r * DKQ + c + i] = (int8_t)roundf(q_vals[i] * inv_qs);
            }
        }

        // Init state
        if (tid < NQ) { s_m[tid] = -1e30f; s_l[tid] = 0.0f; }
        for (int i = tid; i < NQ * DV/2; i += NTHREADS) s_O[i] = __float2half2_rn(0.0f);
        __syncthreads();

        // =================================================================
        // Phase 1: Main loop over K/V blocks
        // =================================================================
        const int n_kv_blocks = (seq_k + NK - 1) / NK;

        for (int kv_b = 0; kv_b < n_kv_blocks; ++kv_b) {
            const int k0 = kv_b * NK;
            const int n_keys = min(NK, seq_k - k0);

            // ---------------------------------------------------------
            // Load K tile: s_K[d * NK + k] = K_int8[k0+k, d]
            // 128×16 = 2048 int8, 128 threads → 16 per thread
            // ---------------------------------------------------------
            for (int i = tid; i < DKQ * NK; i += NTHREADS) {
                const int d = i / NK;
                const int k = i % NK;
                s_K[i] = (k < n_keys) ? K_int8[(int64_t)kv_head * seq_k * DKQ + (k0 + k) * DKQ + d] : (int8_t)0;
            }

            // Load V tile: s_V[d * NK + k] = V_h2[k0+k, d]
            // 128×16 half2 = 2048 half2, 128 threads → 16 per thread
            {
                const int n_v_half2 = DV / 2; // 64
                for (int i = tid; i < DKQ * NK/2; i += NTHREADS) {
                    const int d = i / NK;
                    const int k = (i % NK) * 2;
                    if (k < n_keys) {
                        s_V[d * NK + k/2]     = V_h2[(int64_t)kv_head * seq_k * n_v_half2 + (k0 + k) * n_v_half2 + d];
                        s_V[d * NK + k/2 + 1] = V_h2[(int64_t)kv_head * seq_k * n_v_half2 + (k0 + k) * n_v_half2 + d + 1];
                    }
                }
            }

            // Load K scale
            if (tid == 0) {
                s_k_scale = K_scale[(k0 / 64) * n_heads + kv_head];
            }
            __syncthreads();

            // ---------------------------------------------------------
            // QK^T via INT8 MMA
            //
            // We compute S[8×16] = Q[8×128] × K^T[128×16]
            // Using mma.m16n8k16.s8: D[16×8] = A[16×16] × B[16×8]
            //
            // A[m][k] = Q_int8[m][d0+k]  (m=query, k=head_dim chunk)
            // B[k][n] = K_int8[n][d0+k]  (k=head_dim chunk, n=key)
            // C[m][n] = S_int32[m][n]
            //
            // Warp 0: keys 0..7  (n_offset=0)
            // Warp 1: keys 8..15 (n_offset=8)
            // Warps 2,3: assist (or idle)
            //
            // 8 d-chunks of 16 → 8 MMA per active warp
            // ---------------------------------------------------------
            if (warp_id < 2) {
                const int n_offset = warp_id * 8;

                // Accumulator: 4 int32 per thread (C tile 16×8)
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

                #pragma unroll
                for (int d_chunk = 0; d_chunk < DKQ/16; ++d_chunk) {
                    const int d0 = d_chunk * 16;

                    // Load A: Q_int8 (16×16)
                    // Only 8 rows are valid (NQ=8), rows 8..15 are zero
                    int a0, a1;
                    if (lane < 16) {
                        // Rows 0..7
                        const int row = lane / 4;
                        const int col = (lane % 4) * 4;
                        a0 = pack_i8(
                            s_Q[row * DKQ + d0 + col],
                            s_Q[row * DKQ + d0 + col + 1],
                            s_Q[row * DKQ + d0 + col + 2],
                            s_Q[row * DKQ + d0 + col + 3]);
                        // a1: rows 8..15 → zero (we only have 8 queries)
                        a1 = 0;
                    } else {
                        // Lanes 16..31: rows 8..15 → zero
                        const int row = (lane - 16) / 4; // 0..7
                        const int col = ((lane - 16) % 4) * 4;
                        a0 = 0; // rows 8..15 don't exist
                        a1 = pack_i8(
                            s_Q[row * DKQ + d0 + col],
                            s_Q[row * DKQ + d0 + col + 1],
                            s_Q[row * DKQ + d0 + col + 2],
                            s_Q[row * DKQ + d0 + col + 3]);
                    }

                    // Load B: K_int8 (16×8, col-major)
                    // B[k][n] = s_K[(d0+k) * NK + n_offset + n]
                    int b0;
                    {
                        const int col_b = lane / 4;   // n (0..7)
                        const int row_b = (lane % 4) * 4; // k offset
                        b0 = pack_i8(
                            s_K[(d0 + row_b) * NK + n_offset + col_b],
                            s_K[(d0 + row_b + 1) * NK + n_offset + col_b],
                            s_K[(d0 + row_b + 2) * NK + n_offset + col_b],
                            s_K[(d0 + row_b + 3) * NK + n_offset + col_b]);
                    }

                    // INT8 MMA: mma.m16n8k16.s8.s8.s32
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(b0));
                }

                // Dequant and store to shared
                // c0: C[row][col], c1: C[row+8][col], c2: C[row][col+1], c3: C[row+8][col+1]
                // We only care about rows 0..7 (c0, c2)
                const int row = lane / 4;
                const int col = 2 * (lane % 4);
                if (row < NQ) {
                    s_S[row * NK + n_offset + col]     = (float)c0;
                    s_S[row * NK + n_offset + col + 1] = (float)c2;
                }
                // Rows 8..15 (c1, c3) are discarded
            }
            __syncthreads();

            // ---------------------------------------------------------
            // Dequant: multiply by (q_scale * k_scale * sm_scale)
            // Apply causal mask
            // ---------------------------------------------------------
            for (int i = tid; i < NQ * NK; i += NTHREADS) {
                const int r = i / NK;
                const int c = i % NK;
                float s_val = s_S[r * NK + c] * (s_q_scale[r] * s_k_scale * sm_scale);
                // Causal mask
                if (k0 + c > q0 + r) s_val = -1e30f;
                s_S[r * NK + c] = s_val;
            }
            __syncthreads();

            // ---------------------------------------------------------
            // Online softmax
            // ---------------------------------------------------------
            // Row max
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

            // Update m, compute alpha
            for (int r = tid; r < NQ; r += NTHREADS) {
                const float m_old = s_m[r];
                const float m_new = fmaxf(m_old, s_row_max[r]);
                s_m[r] = m_new;
                s_alpha[r] = (m_old > -1e29f) ? expf(m_old - m_new) : 0.0f;
            }
            __syncthreads();

            // Exp and store P
            for (int i = tid; i < NQ * NK; i += NTHREADS) {
                const int r = i / NK;
                s_S[i] = (s_S[i] > -1e29f) ? expf(s_S[i] - s_m[r]) : 0.0f;
            }
            __syncthreads();

            // Row sum
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

            // Update l: l = l * alpha + sum
            for (int r = tid; r < NQ; r += NTHREADS) {
                s_l[r] = s_l[r] * s_alpha[r] + s_row_sum[r];
            }
            __syncthreads();

            // Rescale O by alpha
            for (int i = tid; i < NQ * DV/2; i += NTHREADS) {
                const int r = i / (DV/2);
                const float2 o = __half22float2(s_O[i]);
                s_O[i] = __float2half2_rn(make_float2(o.x * s_alpha[r], o.y * s_alpha[r]));
            }
            __syncthreads();

            // ---------------------------------------------------------
            // O += P × V  (FP16, no tensor cores for simplicity)
            // O[r, d] += sum_c P[r, c] * V[c, d]
            // P is [8×16] float in s_S, V is [128×16] half2 in s_V (transposed)
            // ---------------------------------------------------------
            {
                const int r = tid / 16;
                const int d_base = (tid % 16) * 4; // 4 half2 per thread

                #pragma unroll
                for (int dd = 0; dd < 4; ++dd) {
                    const int d = d_base + dd;
                    float2 o_acc = __half22float2(s_O[r * (DV/2) + d]);

                    #pragma unroll
                    for (int c = 0; c < NK; ++c) {
                        const float p = s_S[r * NK + c];
                        if (p > 0.0f) {
                            // V[c, d] = s_V[d * NK + c] (half2)
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
        // Phase 2: Normalize and write output
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

        // Write
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
            const float * K_mean,
            const half2 * V_h2,
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
                          + NQ * (DV/2) * sizeof(half2) + NQ * sizeof(float) + NQ * sizeof(float) + NQ * sizeof(float);
        flash_attn_i8qk_kernel<<<grid, block, smem, stream>>>(
            Q_h2, K_int8, K_scale, K_mean, V_h2, O_h2,
            seq_q, seq_k, n_heads, n_kv_heads, sm_scale);
    }

} // namespace ggml_cuda_fattn_i8qk
