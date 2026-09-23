#pragma once
// INT8-QK FlashAttention: K quantization pre-kernel
//
// Based on the SageAttention approach (ref: HyperQwen, syv-ai/HyperQwen).
// K is mean-smoothed per (head, channel) then quantized to INT8 per 64-key tile.
// The mean subtraction is softmax-invariant: for a fixed query q, q.dot(mean_k)
// is identical for every key, so all logits shift by the same constant.
//
// Output precision: cos > 0.99999 vs FP32 reference (measured by HyperQwen).

#include "common.cuh"

namespace ggml_cuda_fattn_i8qk {

    // =====================================================================
    // Step 1: Compute per-(head, channel) mean of K over the full sequence.
    //
    // Input:  K_fp16 [seq_len, n_kv_heads, head_dim] (row-major, half2)
    // Output: k_mean [n_kv_heads, head_dim] (float)
    //
    // Grid: (n_kv_heads * head_dim/256, 1, 1)
    // Block: 256 threads
    // =====================================================================
    __global__ void fattn_k_mean_kernel(
            const half2 * __restrict__ K,
            float * __restrict__ k_mean,
            int seq_len, int n_kv_heads, int head_dim) {
        // Each thread handles one (head, channel) pair
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = n_kv_heads * (head_dim / 2); // half2 elements
        if (idx >= total) return;

        const int h = idx / (head_dim / 2);
        const int c = (idx % (head_dim / 2)) * 2; // channel index (2 per half2)

        float sum0 = 0.0f, sum1 = 0.0f;
        for (int s = 0; s < seq_len; ++s) {
            const half2 val = K[(int64_t)s * n_kv_heads * (head_dim/2) + h * (head_dim/2) + c/2];
            const float2 f = __half22float2(val);
            sum0 += f.x;
            sum1 += f.y;
        }

        const float mean0 = sum0 / seq_len;
        const float mean1 = sum1 / seq_len;

        // Store as float (2 floats per half2 position)
        // Layout: k_mean[head][channel] where channel is 0..head_dim-1
        k_mean[h * head_dim + c]     = mean0;
        k_mean[h * head_dim + c + 1] = mean1;
    }

    // =====================================================================
    // Step 2: Quantize K to INT8 per 64-key tile, with mean subtraction.
    //
    // Input:  K_fp16 [seq_len, n_kv_heads, head_dim] (half2)
    //         k_mean [n_kv_heads, head_dim] (float)
    // Output: k_int8 [seq_len, n_kv_heads, head_dim] (int8)
    //         k_scale [seq_len/64, n_kv_heads] (float) - per-tile absmax/127
    //
    // Grid: (ceil(seq_len/64), n_kv_heads, 1)
    // Block: 256 threads
    // =====================================================================
    __global__ void fattn_k_quant_kernel(
            const half2 * __restrict__ K,
            const float * __restrict__ k_mean,
            int8_t * __restrict__ k_int8,
            float * __restrict__ k_scale,
            int seq_len, int n_kv_heads, int head_dim) {
        constexpr int TILE = 64; // keys per quantization tile

        const int tile_idx = blockIdx.x;
        const int h = blockIdx.y;
        const int tid = threadIdx.x;

        const int k_start = tile_idx * TILE;
        const int k_end = min(k_start + TILE, seq_len);
        const int n_keys = k_end - k_start;
        if (n_keys <= 0) return;

        // Each thread handles head_dim/256 channels (or 1 if head_dim <= 256)
        // For head_dim=256: 256 threads, each handles 1 channel pair (half2)
        const int n_half2 = head_dim / 2;
        if (tid >= n_half2) return;

        // Phase 1: compute absmax of (K - mean) for this tile
        float amax = 0.0f;
        for (int s = 0; s < n_keys; ++s) {
            const int k_idx = k_start + s;
            const half2 kval = K[(int64_t)k_idx * n_kv_heads * n_half2 + h * n_half2 + tid];
            const float2 kf = __half22float2(kval);
            const float2 km = make_float2(
                k_mean[h * head_dim + tid * 2],
                k_mean[h * head_dim + tid * 2 + 1]);
            const float d0 = fabsf(kf.x - km.x);
            const float d1 = fabsf(kf.y - km.y);
            amax = fmaxf(amax, fmaxf(d0, d1));
        }

        // Reduce amax across threads (each thread has a partial amax)
        // For head_dim=256, we have 128 threads with data
        // Use shared memory for reduction
        __shared__ float s_amax[256];
        s_amax[tid] = amax;
        __syncthreads();

        // Serial reduction (128 elements is fast)
        if (tid == 0) {
            float m = 0.0f;
            for (int i = 0; i < n_half2; ++i) {
                m = fmaxf(m, s_amax[i]);
            }
            s_amax[0] = m;
        }
        __syncthreads();

        const float scale = s_amax[0] / 127.0f;
        // Avoid division by zero
        const float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

        // Store scale for this tile
        if (tid == 0) {
            k_scale[tile_idx * n_kv_heads + h] = scale;
        }

        // Phase 2: quantize
        for (int s = 0; s < n_keys; ++s) {
            const int k_idx = k_start + s;
            const half2 kval = K[(int64_t)k_idx * n_kv_heads * n_half2 + h * n_half2 + tid];
            const float2 kf = __half22float2(kval);
            const float2 km = make_float2(
                k_mean[h * head_dim + tid * 2],
                k_mean[h * head_dim + tid * 2 + 1]);

            const int8_t q0 = (int8_t)roundf((kf.x - km.x) * inv_scale);
            const int8_t q1 = (int8_t)roundf((kf.y - km.y) * inv_scale);

            // Store INT8: layout [seq_len, n_kv_heads, head_dim]
            k_int8[(int64_t)k_idx * n_kv_heads * head_dim + h * head_dim + tid * 2]     = q0;
            k_int8[(int64_t)k_idx * n_kv_heads * head_dim + h * head_dim + tid * 2 + 1] = q1;
        }
    }

    // =====================================================================
    // Host-side launcher
    // =====================================================================
    struct k_quant_buffers {
        float  * k_mean;    // [n_kv_heads, head_dim]
        int8_t * k_int8;    // [seq_len, n_kv_heads, head_dim]
        float  * k_scale;   // [seq_len/64, n_kv_heads]
    };

    inline k_quant_buffers alloc_k_quant(int seq_len, int n_kv_heads, int head_dim, cudaStream_t stream = 0) {
        k_quant_buffers buf;
        const int n_tiles = (seq_len + 63) / 64;
        cudaMalloc(&buf.k_mean,  n_kv_heads * head_dim * sizeof(float));
        cudaMalloc(&buf.k_int8,  (int64_t)seq_len * n_kv_heads * head_dim * sizeof(int8_t));
        cudaMalloc(&buf.k_scale, n_tiles * n_kv_heads * sizeof(float));
        return buf;
    }

    inline void free_k_quant(k_quant_buffers & buf) {
        cudaFree(buf.k_mean);
        cudaFree(buf.k_int8);
        cudaFree(buf.k_scale);
    }

    inline void launch_k_quant(
            const half2 * K,
            k_quant_buffers & buf,
            int seq_len, int n_kv_heads, int head_dim,
            cudaStream_t stream = 0) {
        // Step 1: compute mean
        {
            const int total = n_kv_heads * (head_dim / 2);
            const int block = 256;
            const int grid = (total + block - 1) / block;
            fattn_k_mean_kernel<<<grid, block, 0, stream>>>(K, buf.k_mean, seq_len, n_kv_heads, head_dim);
        }
        // Step 2: quantize
        {
            dim3 grid((seq_len + 63) / 64, n_kv_heads, 1);
            const int block = 256;
            fattn_k_quant_kernel<<<grid, block, 0, stream>>>(
                K, buf.k_mean, buf.k_int8, buf.k_scale, seq_len, n_kv_heads, head_dim);
        }
    }

} // namespace ggml_cuda_fattn_i8qk
