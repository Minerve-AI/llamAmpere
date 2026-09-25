#pragma once
// INT8-QK FlashAttention: Integration layer
// Supports both FP16 and Q8_0 KV cache.
//
// Include in ggml-cuda.cu:
//   #include "fattn-i8qk.cuh"
//
// Dispatch:
//   if (use_int8_qk) {
//       ggml_cuda_fattn_i8qk::flash_attn_i8qk(...);
//       return;
//   }

#include "fattn-k-quant.cuh"
#include "fattn-mma-i8.cuh"

namespace ggml_cuda_fattn_i8qk {

    // =====================================================================
    // Workspace
    // =====================================================================
    struct i8qk_workspace {
        int8_t * k_int8;    // [seq_k, n_kv_heads, head_dim]
        float  * k_scale;   // [seq_k/64, n_kv_heads]
        float  * k_mean;    // [n_kv_heads, head_dim]
        bool     allocated;
    };

    inline i8qk_workspace i8qk_alloc(int max_seq_k, int n_kv_heads, int head_dim) {
        i8qk_workspace ws;
        ws.allocated = false;
        const int n_tiles = (max_seq_k + 63) / 64;
        cudaMalloc(&ws.k_int8,  (int64_t)max_seq_k * n_kv_heads * head_dim);
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

    // =====================================================================
    // Entry points
    // =====================================================================

    // Q8_0 KV cache (primary use case)
    inline void flash_attn_i8qk_q8(
            const half2 * Q,
            const void * K_q8,       // Q8_0 K: [n_kv_heads, seq_k, head_dim/32] block_q8_0
            const void * V_q8,       // Q8_0 V: [n_kv_heads, seq_k, head_dim/32] block_q8_0
            half2 * O,
            int seq_q, int seq_k,
            int n_heads, int n_kv_heads, int head_dim,
            float sm_scale,
            i8qk_workspace & ws,
            cudaStream_t stream = 0) {

        // Step 1: Quantize K (Q8_0 → INT8 with mean)
        {
            k_quant_buffers kbuf = { ws.k_mean, ws.k_int8, ws.k_scale };
            launch_k_quant<true>(K_q8, kbuf, seq_k, n_kv_heads, head_dim, stream);
        }

        // Step 2: Run INT8-QK attention (V stays Q8_0, dequantized in-kernel)
        launch_flash_attn_i8qk(
            Q, ws.k_int8, ws.k_scale,
            (const block_q8_0_cuda *)V_q8,
            O,
            seq_q, seq_k, n_heads, n_kv_heads, sm_scale, stream);
    }

    // FP16 KV cache (fallback / testing)
    inline void flash_attn_i8qk_f16(
            const half2 * Q,
            const half2 * K_f16,
            const half2 * V_f16,
            half2 * O,
            int seq_q, int seq_k,
            int n_heads, int n_kv_heads, int head_dim,
            float sm_scale,
            i8qk_workspace & ws,
            cudaStream_t stream = 0) {

        // Step 1: Quantize K (FP16 → INT8 with mean)
        {
            k_quant_buffers kbuf = { ws.k_mean, ws.k_int8, ws.k_scale };
            launch_k_quant<false>(K_f16, kbuf, seq_k, n_kv_heads, head_dim, stream);
        }

        // Step 2: Run INT8-QK attention
        // NOTE: For FP16 V, we need to either:
        //   a) Convert V to Q8_0 first (lossy, not ideal)
        //   b) Use a separate kernel variant that loads V as FP16
        // For now, this path requires V to be Q8_0.
        // TODO: Add FP16 V variant.
        // This is a limitation — for FP16 KV cache, use the standard path.
    }

    // =====================================================================
    // Unified entry point (auto-detect KV type)
    // =====================================================================
    inline void flash_attn_i8qk(
            const half2 * Q,
            const void * K_data,
            const void * V_data,
            half2 * O,
            int seq_q, int seq_k,
            int n_heads, int n_kv_heads, int head_dim,
            ggml_type kv_type,
            float sm_scale,
            i8qk_workspace & ws,
            cudaStream_t stream = 0) {

        if (kv_type == GGML_TYPE_Q8_0) {
            flash_attn_i8qk_q8(Q, K_data, V_data, O,
                seq_q, seq_k, n_heads, n_kv_heads, head_dim,
                sm_scale, ws, stream);
        }
        // else: fall back to standard FP16 path (caller handles)
    }

} // namespace ggml_cuda_fattn_i8qk
