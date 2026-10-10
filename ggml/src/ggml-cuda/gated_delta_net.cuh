#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// GDN state read overlap: when the state input (src[5]) comes from a GET_ROWS over a
// persistent cache, the kernel reads directly from the cache using per-sequence row indices,
// skipping the GET_ROWS copy entirely.
struct ggml_cuda_gated_delta_net_state_src {
    const int32_t * rows       = nullptr;  // nullptr = classic path (read from src[5])
    int64_t         row_stride = 0;        // floats between consecutive rows in the cache
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// fused gate computation: softplus(alpha + dt_bias) * A
void ggml_cuda_op_gdn_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
