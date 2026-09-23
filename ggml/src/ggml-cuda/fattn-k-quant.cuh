#pragma once
// INT8-QK FlashAttention: K quantization pre-kernel
// Supports both FP16 and Q8_0 KV cache formats.
//
// Pipeline:
//   K (Q8_0 or FP16) → dequant to FP32 → compute per-(head,channel) mean
//                     → quantize to INT8 per 64-key tile (with mean subtraction)
//
// Mean subtraction is softmax-invariant:
//   softmax(Q·K) = softmax(Q·(K - mean)) because Q·mean is constant per query.

#include "common.cuh"
#include "ggml.h"

namespace ggml_cuda_fattn_i8qk {

    // =====================================================================
    // Q8_0 block dequant helper
    // =====================================================================
    // ggml Q8_0: block of 32 int8 values + 1 float scale
    // value[i] = qs[i] * d
    struct __align__(16) block_q8_0_cuda {
        float d;
        int8_t qs[32];
    };

    // =====================================================================
    // Kernel 1: Compute per-(head, channel) mean of K
    // =====================================================================
    // Grid: (n_heads * head_dim/256, 1, 1)
    // Block: 256 threads
    // Each block handles 256 (head, channel) pairs
    // Each thread accumulates over seq_len keys
    //
    // K layout (Q8_0): [n_kv_heads, seq_k, head_dim] as block_q8_0
    //   For each (kv_head, seq_pos): head_dim/32 blocks of Q8_0
    //   K[kv_head][s][c] = blocks[kv_head * seq_k * (head_dim/32) + s * (head_dim/32) + c/32].qs[c%32] * d
    //
    // K layout (FP16): [n_kv_heads, seq_k, head_dim] as half2
    //
    template<bool K_IS_Q8>
    __global__ void fattn_k_mean_kernel(
            const void * __restrict__ K,
            float * __restrict__ k_mean,
            int seq_len, int n_heads, int head_dim) {
        // Each thread computes the mean for one (head, channel) pair
        const int hc = threadIdx.x + blockIdx.x * blockDim.x;
        if (hc >= n_heads * head_dim) return;

        const int h = hc / head_dim;
        const int c = hc % head_dim;

        float sum = 0.0f;

        if constexpr (K_IS_Q8) {
            const int nblocks_per_row = head_dim / 32;
            const block_q8_0_cuda * K_q8 = (const block_q8_0_cuda *)K;
            // For each key s: K[h][s][c]
            for (int s = 0; s < seq_len; ++s) {
                const int block_idx = h * seq_len * nblocks_per_row + s * nblocks_per_row + c / 32;
                const block_q8_0_cuda blk = K_q8[block_idx];
                sum += (float)blk.qs[c % 32] * blk.d;
            }
        } else {
            const half2 * K_h2 = (const half2 *)K;
            const int n_half2 = head_dim / 2;
            // K is [n_heads, seq_len, head_dim] as half2
            // K[h][s][c] = K_h2[h * seq_len * n_half2 + s * n_half2 + c/2] (one component)
            for (int s = 0; s < seq_len; ++s) {
                const half2 v = K_h2[(int64_t)h * seq_len * n_half2 + s * n_half2 + c / 2];
                sum += (c % 2 == 0) ? __low2float(v) : __high2float(v);
            }
        }

        k_mean[hc] = sum / seq_len;
    }

    // =====================================================================
    // Kernel 2: Quantize K to INT8 (per 64-key tile, with mean subtraction)
    // =====================================================================
    // Grid: (n_tiles, n_heads, 1) where n_tiles = ceil(seq_len / 64)
    // Block: 256 threads
    //
    // For each (tile, head):
    //   1. Compute absmax of (K - mean) over 64 keys × head_dim
    //   2. scale = absmax / 127
    //   3. K_int8[k][c] = round((K[k][c] - mean[c]) / scale)
    //
    // Output layout:
    //   k_int8: [seq_len, n_heads, head_dim] int8 (same layout as K but INT8)
    //   k_scale: [n_tiles, n_heads] float
    //
    template<bool K_IS_Q8>
    __global__ void fattn_k_quant_kernel(
            const void * __restrict__ K,
            const float * __restrict__ k_mean,
            int8_t * __restrict__ k_int8,
            float * __restrict__ k_scale,
            int seq_len, int n_heads, int head_dim) {

        const int tile = blockIdx.x;
        const int h = blockIdx.y;
        const int k_start = tile * 64;
        const int k_end = min(k_start + 64, seq_len);
        const int n_keys = k_end - k_start;

        // Phase 1: Compute absmax (cooperative reduction)
        // 64 keys × head_dim values, 256 threads
        // Each thread handles (64*head_dim/256) values
        __shared__ float s_absmax[8]; // 8 partial maxes

        const int tid = threadIdx.x;
        float local_max = 0.0f;

        if constexpr (K_IS_Q8) {
            const int nblocks_per_row = head_dim / 32;
            const block_q8_0_cuda * K_q8 = (const block_q8_0_cuda *)K;

            for (int i = tid; i < n_keys * head_dim; i += blockDim.x) {
                const int k = i / head_dim;
                const int c = i % head_dim;
                const int block_idx = h * seq_len * nblocks_per_row + (k_start + k) * nblocks_per_row + c / 32;
                const block_q8_0_cuda blk = K_q8[block_idx];
                const float val = (float)blk.qs[c % 32] * blk.d - k_mean[h * head_dim + c];
                local_max = fmaxf(local_max, fabsf(val));
            }
        } else {
            const half2 * K_h2 = (const half2 *)K;
            const int n_half2 = head_dim / 2;

            for (int i = tid; i < n_keys * head_dim; i += blockDim.x) {
                const int k = i / head_dim;
                const int c = i % head_dim;
                const half2 v = K_h2[(int64_t)h * seq_len * n_half2 + (k_start + k) * n_half2 + c / 2];
                const float val = ((c % 2 == 0) ? __low2float(v) : __high2float(v)) - k_mean[h * head_dim + c];
                local_max = fmaxf(local_max, fabsf(val));
            }
        }

        // Warp reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }
        if ((tid % 32) == 0) {
            s_absmax[tid / 32] = local_max;
        }
        __syncthreads();

        // Final reduce (8 warps)
        if (tid < 8) {
            float m = s_absmax[tid];
            #pragma unroll
            for (int i = 1; i < 8; ++i) m = fmaxf(m, s_absmax[i]);
            if (tid == 0) {
                const float scale = m / 127.0f;
                k_scale[tile * n_heads + h] = scale;
            }
        }
        __syncthreads();

        const float scale = k_scale[tile * n_heads + h];
        const float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

        // Phase 2: Quantize
        if constexpr (K_IS_Q8) {
            const int nblocks_per_row = head_dim / 32;
            const block_q8_0_cuda * K_q8 = (const block_q8_0_cuda *)K;

            for (int i = tid; i < n_keys * head_dim; i += blockDim.x) {
                const int k = i / head_dim;
                const int c = i % head_dim;
                const int block_idx = h * seq_len * nblocks_per_row + (k_start + k) * nblocks_per_row + c / 32;
                const block_q8_0_cuda blk = K_q8[block_idx];
                const float val = (float)blk.qs[c % 32] * blk.d - k_mean[h * head_dim + c];
                k_int8[(int64_t)(k_start + k) * n_heads * head_dim + h * head_dim + c] =
                    (int8_t)roundf(val * inv_scale);
            }
        } else {
            const half2 * K_h2 = (const half2 *)K;
            const int n_half2 = head_dim / 2;

            for (int i = tid; i < n_keys * head_dim; i += blockDim.x) {
                const int k = i / head_dim;
                const int c = i % head_dim;
                const half2 v = K_h2[(int64_t)h * seq_len * n_half2 + (k_start + k) * n_half2 + c / 2];
                const float val = ((c % 2 == 0) ? __low2float(v) : __high2float(v)) - k_mean[h * head_dim + c];
                k_int8[(int64_t)(k_start + k) * n_heads * head_dim + h * head_dim + c] =
                    (int8_t)roundf(val * inv_scale);
            }
        }
    }

    // =====================================================================
    // Host launcher
    // =====================================================================
    struct k_quant_buffers {
        float  * k_mean;    // [n_heads, head_dim]
        int8_t * k_int8;    // [seq_len, n_heads, head_dim]
        float  * k_scale;   // [n_tiles, n_heads]
    };

    template<bool K_IS_Q8>
    inline void launch_k_quant(
            const void * K,
            k_quant_buffers & buf,
            int seq_len, int n_heads, int head_dim,
            cudaStream_t stream = 0) {

        // Kernel 1: Mean
        {
            const int total_hc = n_heads * head_dim;
            const int block = 256;
            const int grid = (total_hc + block - 1) / block;
            fattn_k_mean_kernel<K_IS_Q8><<<grid, block, 0, stream>>>(
                K, buf.k_mean, seq_len, n_heads, head_dim);
        }

        // Kernel 2: Quantize
        {
            const int n_tiles = (seq_len + 63) / 64;
            dim3 grid(n_tiles, n_heads);
            dim3 block(256);
            fattn_k_quant_kernel<K_IS_Q8><<<grid, block, 0, stream>>>(
                K, buf.k_mean, buf.k_int8, buf.k_scale,
                seq_len, n_heads, head_dim);
        }
    }

} // namespace ggml_cuda_fattn_i8qk
