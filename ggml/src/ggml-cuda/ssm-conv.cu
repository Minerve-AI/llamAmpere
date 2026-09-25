#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}


// F16 version: read F16 input, F32 weights, compute in F32, write F16
template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f16(const __half * src0_ptr, const float * src1_ptr,
                                     const float * bias_ptr,
                                     const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                     __half * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                     const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const __half * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float  * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float  * GGML_CUDA_RESTRICT bias = bias_ptr;
    __half       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const __half * x_block = (const __half *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float  * w_block = (const float *)  ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    __half *       y_block = (__half *)       ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(__half);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(__half);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = __half2float(x_block[tid * stride_x + j]);
            }
        } else {
            x[(i - 1) % d_conv] = __half2float(x_block[tid * stride_x + i + d_conv - 1]);
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = __float2half(apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf);
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f16(const __half * __restrict__ src0, const float * __restrict__ src1,
                                                const float * __restrict__ bias,
                                                const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                                const int src1_nb1, __half * __restrict__ dst, const int dst_nb0,
                                                const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const __half * x_block = (const __half *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                              bidz * split_n_t * src0_nb0);
    const float  * w_block = (const float *)  ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    __half *       y_block =
        (__half *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(__half);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(__half);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = __half2float(x_block[row * stride_x + col]);
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = __float2half(apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf);
    }
}

template <bool apply_silu>
static void ssm_conv_f16_cuda(const __half * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                               const int src0_nb2, const int src1_nb1, __half * dst, const int dst_nb0, const int dst_nb1,
                               const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                               const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f16<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                         src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f16<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}


void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch
    cudaStream_t  stream = ctx.stream();

    // F16 path
    if (src0->type == GGML_TYPE_F16) {
        const __half * src0_d_f16 = (const __half *) src0->data;
        const float  * src1_d_f16 = (const float *)  src1->data;
        const float  * bias_d_f16 = fuse_bias ? (const float *) bias->data : nullptr;
        __half *       dst_d_f16  = (__half *) out->data;
        if (fuse_silu) {
            ssm_conv_f16_cuda<true>(src0_d_f16, src1_d_f16, bias_d_f16, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d_f16, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
        } else {
            ssm_conv_f16_cuda<false>(src0_d_f16, src1_d_f16, bias_d_f16, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d_f16, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
        }
        return;
    }

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// ============================================================================

// ============================================================================
// Fused CONCAT + SSM_CONV (+ SILU) kernel
// Reads directly from the two pre-concat sources (conv_states + qkv_mixed)
// eliminating the intermediate CONCAT tensor in DRAM.
//
// conv_states: [d_conv-1, conv_dim, n_s]  (previous window)
// qkv_mixed:   [n_t, conv_dim, n_s]       (current tokens, transposed)
// kernel:      [d_conv, conv_dim]         (depthwise conv weights)
// output:      [conv_dim, n_t, n_s]
// ============================================================================

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_fused_concat_f32(
    const float * conv_states_ptr,
    const float * qkv_mixed_ptr,
    const float * kernel_ptr,
    const float * bias_ptr,
    float * dst_ptr,
    const int cs_nb0, const int cs_nb1, const int cs_nb2,
    const int qk_nb0, const int qk_nb1, const int qk_nb2,
    const int src1_nb1,
    const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT cs = conv_states_ptr;
    const float * GGML_CUDA_RESTRICT qk = qkv_mixed_ptr;
    const float * GGML_CUDA_RESTRICT src1 = kernel_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;

    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int cs_s0 = cs_nb0 / sizeof(float);
    const int qk_s0 = qk_nb0 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const float * cs_ch = (const float *)((const char *)cs + bidx * cs_nb2 + (bidy * split_d_inner + tid) * cs_nb1);
    const float * qk_ch = (const float *)((const char *)qk + bidx * qk_nb2 + (bidy * split_d_inner + tid) * qk_nb1);
    const float * w_block = (const float *)((const char *)src1 + bidy * split_d_inner * src1_nb1);
    float * y_block = (float *)((char *)dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Initial window: d_conv-1 from conv_states, 1 from qkv_mixed[0]
#pragma unroll
    for (size_t j = 0; j < d_conv - 1; j++) {
        x[j] = cs_ch[j * cs_s0];
    }
    x[d_conv - 1] = qk_ch[0];

    for (int64_t i = 0; i < n_t; i++) {
        if (i > 0) {
            x[(i - 1) % d_conv] = qk_ch[i * qk_s0];
        }
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// Long-token variant (n_t > 32): uses shared memory
template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_fused_concat_long_token_f32(
    const float * __restrict__ cs_ptr,
    const float * __restrict__ qk_ptr,
    const float * __restrict__ src1_ptr,
    const float * __restrict__ bias_ptr,
    const int cs_nb0, const int cs_nb1, const int cs_nb2,
    const int qk_nb0, const int qk_nb1, const int qk_nb2,
    const int src1_nb1,
    float * __restrict__ dst,
    const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int cs_s0 = cs_nb0 / sizeof(float);
    const int qk_s0 = qk_nb0 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + local_n_t;
    const int     load_cols = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    // Cooperative load: smem[channel][time_pos]
    // time_pos 0..d_conv-2: from conv_states
    // time_pos d_conv-1..d_conv-1+split_n_t-1: from qkv_mixed
    {
        int row = tid / load_cols;
        int col = tid % load_cols;
        constexpr int total_elems = split_d_inner * load_cols;
        for (int idx = 0; idx < total_elems; idx += split_d_inner) {
            if (row < (int)split_d_inner) {
                const int ch = row;
                if (col < (int)(d_conv - 1)) {
                    // From conv_states: [col, ch, bidx]
                    smem[ch * n_cols + col] =
                        ((const float *)((const char *)cs_ptr + bidx * cs_nb2 + ch * cs_nb1))[col * cs_s0 + bidy * split_d_inner * cs_s0];
                } else {
                    const int tok = bidz * split_n_t + (col - (int)(d_conv - 1));
                    if (tok < n_t) {
                        smem[ch * n_cols + col] =
                            ((const float *)((const char *)qk_ptr + bidx * qk_nb2 + tok * qk_nb0 + ch * qk_nb1))[bidy * split_d_inner * qk_s0];
                    } else {
                        smem[ch * n_cols + col] = 0.0f;
                    }
                }
            }
            col += split_d_inner;
            row += col / load_cols;
            col  = col % load_cols;
            if (idx >= total_elems - tid - split_d_inner) break;
        }
    }
    __syncthreads();

    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = ((const float *)((const char *)src1_ptr + bidy * split_d_inner * src1_nb1))[tid * stride_w + j];
    }

    float b = bias_ptr != nullptr ? bias_ptr[bidy * split_d_inner + tid] : 0.0f;

    float * y_block = (float *)((char *)dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_fused_concat_f32_cuda(
    const float * cs, const float * qk, const float * src1, const float * bias,
    const int cs_nb0, const int cs_nb1, const int cs_nb2,
    const int qk_nb0, const int qk_nb1, const int qk_nb2,
    const int src1_nb1,
    float * dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int64_t nc, const int64_t nr, const int64_t n_t,
    const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_fused_concat_f32<apply_silu, threads, kNC>, launch_params,
                cs, qk, src1, bias, dst,
                cs_nb0, cs_nb1, cs_nb2, qk_nb0, qk_nb1, qk_nb2, src1_nb1,
                dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_fused_concat_long_token_f32<apply_silu, threads, kNC, split_n_t>
                <<<blocks, threads, smem_size, stream>>>(
                cs, qk, src1, bias,
                cs_nb0, cs_nb1, cs_nb2, qk_nb0, qk_nb1, qk_nb2, src1_nb1,
                dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

// Dispatch: fused CONCAT + SSM_CONV (+ ADD) + SILU
void ggml_cuda_op_ssm_conv_fused_concat(ggml_backend_cuda_context & ctx, ggml_tensor * ssm_conv,
                                        ggml_tensor * concat, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * cs   = concat->src[0];  // conv_states [d_conv-1, conv_dim, n_s]
    const struct ggml_tensor * qk   = concat->src[1];  // qkv_mixed   [n_t, conv_dim, n_s]
    const struct ggml_tensor * src1 = ssm_conv->src[1]; // kernel      [d_conv, conv_dim]

    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;
    GGML_ASSERT(!fuse_bias || fuse_silu);

    const struct ggml_tensor * bias = nullptr;
    if (fuse_bias) {
        bias = (bias_add_node->src[0] == ssm_conv) ? bias_add_node->src[1] : bias_add_node->src[0];
    }

    const struct ggml_tensor * out = fuse_silu ? silu_dst : ssm_conv;

    const int64_t nc  = src1->ne[0];
    const int64_t nr  = ssm_conv->ne[0];
    const int64_t n_t = out->ne[1];
    const int64_t n_s = out->ne[2];
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(cs->type == GGML_TYPE_F32);
    GGML_ASSERT(qk->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    GGML_ASSERT(nr % 128 == 0);

    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    const float * cs_d   = (const float *) cs->data;
    const float * qk_d   = (const float *) qk->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;

    if (fuse_silu) {
        ssm_conv_fused_concat_f32_cuda<true>(cs_d, qk_d, src1_d, bias_d,
            cs->nb[0], cs->nb[1], cs->nb[2], qk->nb[0], qk->nb[1], qk->nb[2], src1->nb[1],
            dst_d, out->nb[0], out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_fused_concat_f32_cuda<false>(cs_d, qk_d, src1_d, bias_d,
            cs->nb[0], cs->nb[1], cs->nb[2], qk->nb[0], qk->nb[1], qk->nb[2], src1->nb[1],
            dst_d, out->nb[0], out->nb[1], out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
