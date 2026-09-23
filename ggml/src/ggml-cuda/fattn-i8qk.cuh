#pragma once
// INT8-QK FlashAttention: Integration layer
//
// This file provides the entry point for the INT8-QK attention path.
// Include this in ggml-cuda.cu after the existing fattn includes.
//
// Usage:
//   #include "fattn-i8qk.cuh"
//
// In the flash_attn_ext dispatch:
//   if (use_int8_qk) {
//       ggml_cuda_fattn_i8qk::flash_attn_i8qk(...);
//       return;
//   }

#include "fattn-k-quant.cuh"
#include "fattn-mma-i8.cuh"

namespace ggml_cuda_fattn_i8qk {

    // =====================================================================
    // Full attention entry point
    // Handles: K quantization → INT8 QK^T + FP16 PV
    //
    // Q:  [n_heads, seq_q, head_dim] FP16
    // K:  [n_kv_heads, seq_k, head_dim] FP16 (will be quantized internally)
    // V:  [n_kv_heads, seq_k, head_dim] FP16
    // O:  [n_heads, seq_q, head_dim] FP16
    //
    // =====================================================================
    struct i8qk_workspace {
        int8_t * k_int8;   // [seq_k, n_kv_heads, head_dim]
        float  * k_scale;  // [seq_k/64, n_kv_heads]
        float  * k_mean;   // [n_kv_heads, head_dim]
        bool     allocated;
    };

    inline i8qk_workspace i8qk_alloc(int seq_k, int n_kv_heads, int head_dim) {
        i8qk_workspace ws;
        ws.allocated = false;
        const int n_tiles = (seq_k + 63) / 64;
        cudaMalloc(&ws.k_int8,  (int64_t)seq_k * n_kv_heads * head_dim);
        cudaMalloc(&ws.k_scale, n_tiles * n_kv_heads * sizeof(float));
        cudaMalloc(&ws.k_mean,  n_kv_heads * head_dim * sizeof(float));
        ws.allocated = true;
        return ws;
    }

    inline void i8qk_free(i8qk_workspace & ws) {
        if (ws.allocated) {
            cudaFree(ws.k_int8);
            cudaFree(ws.k_scale);
            cudaFree(ws.k_mean);
            ws.allocated = false;
        }
    }

    // Main entry: runs K quant + INT8 attention
    inline void flash_attn_i8qk_full(
            const half2 * Q,
            const half2 * K_fp16,
            const half2 * V,
            half2 * O,
            int seq_q, int seq_k,
            int n_heads, int n_kv_heads, int head_dim,
            float sm_scale,
            i8qk_workspace & ws,
            cudaStream_t stream = 0) {

        // Step 1: Quantize K (mean + INT8)
        {
            struct k_quant_buffers {
                float * k_mean;
                int8_t * k_int8;
                float * k_scale;
            } kbuf = { ws.k_mean, ws.k_int8, ws.k_scale };
            launch_k_quant(K_fp16, kbuf, seq_k, n_kv_heads, head_dim, stream);
        }

        // Step 2: Run INT8-QK attention
        launch_flash_attn_i8qk(
            Q, ws.k_int8, ws.k_scale, ws.k_mean, V, O,
            seq_q, seq_k, n_heads, n_kv_heads, sm_scale, stream);
    }

} // namespace ggml_cuda_fattn_i8qk
