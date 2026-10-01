// Fused ADD + RMS_NORM + MUL + optional Q8_0 quantization (F16)
// Based on JakeATX/llamAmpere add_rms_norm_mul_f32, adapted for F16 inputs.
//
// The kernel computes:
//   residual = x + y          (written to `sum`)
//   normed   = rms_norm(residual) * w   (written to `dst` as F16)
//   q8       = quantize_q8_0(normed)    (written to `q8` if q8_mode != 0)
//
// This eliminates a separate quantization kernel launch when the next
// consumer is a Q4_K x Q8_0 matmul (MMVQ path).

#include "norm.cuh"
#include "common.cuh"
#include "ggml-common.h"

// q8_mode: 0 = no quantization, 1 = Q8_0
template <int block_size, int q8_mode>
static __global__ void add_rms_norm_mul_f16_q8(const __half * __restrict__ x,
                                               const __half * __restrict__ y,
                                               __half *           __restrict__ sum,
                                               const __half * __restrict__ w,
                                               __half *           __restrict__ dst,
                                               block_q8_1 *       __restrict__ q8,
                                               const int     ncols,
                                               const int     ncols_padded,
                                               const int64_t stride_row_x,
                                               const int64_t stride_ch_x,
                                               const int64_t stride_s_x,
                                               const int64_t stride_row_y,
                                               const int64_t stride_ch_y,
                                               const int64_t stride_s_y,
                                               const float   eps) {
    ggml_cuda_pdl_lc();
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    static_assert(block_size % QK8_1 == 0, "q8_1 blocks must not straddle warps");

    x   += sample*stride_s_x + channel*stride_ch_x + row*stride_row_x;
    y   += sample*stride_s_y + channel*stride_ch_y + row*stride_row_y;
    sum += ((sample*gridDim.y + channel)*gridDim.x + row)*ncols;
    dst += ((sample*gridDim.y + channel)*gridDim.x + row)*ncols;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    // Phase 1: compute residual, write it, accumulate sum of squares
    for (int col = tid; col < ncols; col += block_size) {
        const float v = __half2float(x[col]) + __half2float(y[col]);
        sum[col] = __float2half(v);
        tmp += v * v;
    }

    // Block reduce
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (q8_mode == 0) {
        // Phase 2a: write normalized F16 output only
        for (int col = tid; col < ncols; col += block_size) {
            const float v = __half2float(sum[col]);
            dst[col] = __float2half(scale * v * __half2float(w[col]));
        }
    } else {
        // Phase 2b: write normalized F16 output + Q8_0 quantized output
        const int lane = tid % QK8_1;
        // grid is (nrows, nchannels, nsamples), linear row index:
        block_q8_1 * yq = q8 + ((int64_t)(sample*gridDim.y + channel)*gridDim.x + row)*(ncols_padded/QK8_1);

        for (int col = tid; col < ncols_padded; col += block_size) {
            float v = 0.0f;
            if (col < ncols) {
                v = scale * __half2float(sum[col]) * __half2float(w[col]);
                dst[col] = __float2half(v);
            }

            // Q8_0 quantization: 32 elements per block, one warp per block
            float amax = fabsf(v);
            amax = warp_reduce_max<QK8_1>(amax);

            const float  d = amax / 127.0f;
            const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(v / d);

            const int ib = col / QK8_1;
            yq[ib].qs[lane] = q;

            const float qsum = warp_reduce_sum<QK8_1>((float)q);
            if (lane == 0) {
                yq[ib].ds = __halves2half2(__float2half(d), __float2half(d * qsum));
            }
        }
    }
}

template <int block_size, int q8_mode>
static void launch_add_rms_norm_mul_f16_q8(const __half * x, const __half * y, __half * sum,
                                            const __half * w, __half * dst, block_q8_1 * q8,
                                            int ncols, int ncols_padded,
                                            int nrows, int nchannels, int nsamples,
                                            int64_t sx01, int64_t sx02, int64_t sx03,
                                            int64_t sy01, int64_t sy02, int64_t sy03,
                                            float eps, cudaStream_t stream) {
    dim3 block_dims(block_size);
    dim3 grid_dims(nrows, nchannels, nsamples);
    size_t shared_mem_size = block_size * sizeof(float);
    add_rms_norm_mul_f16_q8<block_size, q8_mode><<<grid_dims, block_dims, shared_mem_size, stream>>>(
        x, y, sum, w, dst, q8, ncols, ncols_padded,
        sx01, sx02, sx03, sy01, sy02, sy03, eps);
}

static void add_rms_norm_mul_f16_q8_cuda(const __half * x, const __half * y, __half * sum,
                                          const __half * w, __half * dst, block_q8_1 * q8,
                                          int ncols, int ncols_padded,
                                          int nrows, int nchannels, int nsamples,
                                          int64_t sx01, int64_t sx02, int64_t sx03,
                                          int64_t sy01, int64_t sy02, int64_t sy03,
                                          float eps, cudaStream_t stream) {
    if (q8 == nullptr) {
        if (ncols <= 1024) {
            launch_add_rms_norm_mul_f16_q8<256, 0>(x, y, sum, w, dst, nullptr, ncols, ncols_padded,
                nrows, nchannels, nsamples, sx01, sx02, sx03, sy01, sy02, sy03, eps, stream);
        } else {
            launch_add_rms_norm_mul_f16_q8<1024, 0>(x, y, sum, w, dst, nullptr, ncols, ncols_padded,
                nrows, nchannels, nsamples, sx01, sx02, sx03, sy01, sy02, sy03, eps, stream);
        }
    } else {
        if (ncols <= 1024) {
            launch_add_rms_norm_mul_f16_q8<256, 1>(x, y, sum, w, dst, q8, ncols, ncols_padded,
                nrows, nchannels, nsamples, sx01, sx02, sx03, sy01, sy02, sy03, eps, stream);
        } else {
            launch_add_rms_norm_mul_f16_q8<1024, 1>(x, y, sum, w, dst, q8, ncols, ncols_padded,
                nrows, nchannels, nsamples, sx01, sx02, sx03, sy01, sy02, sy03, eps, stream);
        }
    }
}

// Public entry point: fused ADD + RMS_NORM + MUL (+ optional Q8_0)
// add_tensor:  GGML_OP_ADD, src[0]=x, src[1]=y, type=F16
// norm_tensor: GGML_OP_RMS_NORM, src[0]=add_tensor
// mul_tensor:  GGML_OP_MUL, src[0]=norm_tensor, src[1]=w (or vice versa)
// q8_dst:      output buffer for Q8_0 (nullptr if no quantization)
void ggml_cuda_op_add_rms_norm_mul_q8(ggml_backend_cuda_context & ctx,
                                      ggml_tensor *               add_tensor,
                                      ggml_tensor *               norm_tensor,
                                      ggml_tensor *               mul_tensor,
                                      void *                      q8_dst) {
    const ggml_tensor * x = add_tensor->src[0];
    const ggml_tensor * y = add_tensor->src[1];

    float eps = 0.0f;
    memcpy(&eps, norm_tensor->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    // Find the weight tensor in mul_tensor
    const ggml_tensor * w = nullptr;
    if (mul_tensor->src[0] == norm_tensor) {
        w = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == norm_tensor) {
        w = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false && "mul_tensor must have norm_tensor as a source");
    }

    GGML_ASSERT(x->type == GGML_TYPE_F16);
    GGML_ASSERT(y->type == GGML_TYPE_F16);
    GGML_ASSERT(w->type == GGML_TYPE_F16);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F16);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F16);

    const __half * x_d   = (const __half *) x->data;
    const __half * y_d   = (const __half *) y->data;
    const __half * w_d   = (const __half *) w->data;
    __half       * sum_d = (__half *) add_tensor->data;
    __half       * dst_d = (__half *) mul_tensor->data;
    block_q8_1 *   q8_d  = (block_q8_1 *) q8_dst;

    cudaStream_t stream = ctx.stream();

    const int64_t ncols     = x->ne[0];
    const int64_t nrows     = x->ne[1];
    const int64_t nchannels = x->ne[2];
    const int64_t nsamples  = x->ne[3];

    const int ncols_padded = (int) GGML_PAD(ncols, QK8_1);

    const size_t ts = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts);
    const int64_t sx01 = x->nb[1] / ts;
    const int64_t sx02 = x->nb[2] / ts;
    const int64_t sx03 = x->nb[3] / ts;

    GGML_ASSERT(y->nb[0] == ts);
    const int64_t sy01 = y->nb[1] / ts;
    const int64_t sy02 = y->nb[2] / ts;
    const int64_t sy03 = y->nb[3] / ts;

    GGML_ASSERT(w->ne[0] == ncols);

    add_rms_norm_mul_f16_q8_cuda(x_d, y_d, sum_d, w_d, dst_d, q8_d,
                                  (int)ncols, ncols_padded,
                                  (int)nrows, (int)nchannels, (int)nsamples,
                                  sx01, sx02, sx03, sy01, sy02, sy03,
                                  eps, stream);
}
