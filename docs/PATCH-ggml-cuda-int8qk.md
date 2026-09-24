// ============================================================================
// PATCH: ggml-cuda.cu — INT8-QK dispatch integration
// ============================================================================
//
// This file documents the changes needed in ggml/src/ggml-cuda/ggml-cuda.cu
// to enable the INT8-QK FlashAttention path.
//
// Apply these changes manually or via `git apply`.
//
// ============================================================================

// --- CHANGE 1: Add include (after existing fattn includes, ~line 50) ---
//
//   #include "fattn-mma-f16.cuh"
//   #include "fattn-mma-turbo.cuh"
// + #include "fattn-i8qk.cuh"
//   #include "fattn-tile.cuh"
//
// --- CHANGE 2: Add INT8-QK config flag (in the ggml_cuda_context or similar) ---
//
//   struct ggml_cuda_context {
//       ...
// +     bool use_int8_qk;           // Enable INT8 QK^T attention path
// +     ggml_cuda_fattn_i8qk::i8qk_workspace i8qk_ws;
//       ...
//   };
//
//   // In the context init:
// +   ctx.use_int8_qk = getenv("LLAMA_F16_INT8_QK") != nullptr;
// +   ctx.i8qk_ws.allocated = false;
//
// --- CHANGE 3: Add INT8-QK dispatch in flash_attn_ext (the main entry point) ---
//
//   In the function that dispatches to flash_attn_ext_f16 or similar,
//   add the INT8-QK path BEFORE the standard FP16 path:
//
//   void ggml_cuda_flash_attn_ext(...) {
//       ...
// +     // INT8-QK path (HyperQwen strategy)
// +     if (ctx.use_int8_qk &&
// +         head_dim == 128 &&           // Only D=128 for now
// +         type_K == GGML_TYPE_F16 &&   // K must be FP16 (we quantize internally)
// +         type_V == GGML_TYPE_F16) {
// +         // Allocate workspace if needed
// +         if (!ctx.i8qk_ws.allocated) {
// +             ctx.i8qk_ws = ggml_cuda_fattn_i8qk::i8qk_alloc(
// +                 seq_k, n_kv_heads, head_dim);
// +         }
// +         // Run INT8-QK attention
// +         ggml_cuda_fattn_i8qk::flash_attn_i8qk_full(
// +             (const half2 *) Q->data,
// +             (const half2 *) K->data,
// +             (const half2 *) V->data,
// +             (half2 *)      O->data,
// +             seq_q, seq_k,
// +             n_heads, n_kv_heads, head_dim,
// +             sm_scale,
// +             ctx.i8qk_ws,
// +             stream);
// +         return;
// +     }
//       // ... existing FP16 path ...
//   }
//
// --- CHANGE 4: Free workspace on context destruction ---
//
//   void ggml_cuda_free_context(ggml_cuda_context & ctx) {
//       ...
// +     ggml_cuda_fattn_i8qk::i8qk_free(ctx.i8qk_ws);
//       ...
//   }
//
// ============================================================================
// BUILD SYSTEM
// ============================================================================
//
// No CMake changes needed — the new files are headers (.cuh) that get
// #included by ggml-cuda.cu. They don't need separate compilation.
//
// The kernel is __global__ and will be compiled as part of ggml-cuda.cu.
//
// ============================================================================
// ENVIRONMENT VARIABLE
// ============================================================================
//
//   LLAMA_F16_INT8_QK=1  →  Enable INT8-QK path
//
//   Without the env var, the standard FP16 path is used (no behavior change).
//
// ============================================================================
// LIMITATIONS (current implementation)
// ============================================================================
//
// 1. Only head_dim=128 is supported (D=128). Other sizes fall back to FP16.
// 2. K must be in FP16 format (quantization happens in-kernel).
//    Q8_0 KV cache is NOT supported yet (would need dequant → quant).
// 3. The PV multiplication uses regular FMA (not FP16 MMA tensor cores).
//    This is a bottleneck — the next optimization step is to add FP16 MMA
//    for PV (same as the standard fattn-mma-f16 kernel).
// 4. No causal mask optimization (the mask is applied element-wise).
// 5. Single KV head per block (no multi-head batching in the kernel).
//
// ============================================================================
// NEXT STEPS (post-testing)
// ============================================================================
//
// 1. Add FP16 MMA for PV (use existing mma.cuh half2 primitives)
// 2. Add Q8_0 KV cache support (dequant K to FP16, then quantize to INT8)
// 3. Add multi-head batching (process 2-4 heads per block)
// 4. Add cp.async for K/V loading (overlap compute and memory)
// 5. Benchmark vs FP16 baseline
// 6. Add support for D=64, D=256
// 7. Integration with the existing template system (ncols1, ncols2, etc.)
