#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

// =============================================================================
// Fused GLU MMQ kernel - separate TU to avoid template explosion in other TUs
// =============================================================================

// Fused write-back: applies silu(gate) * up and writes the product
static __device__ __forceinline__ float mmq_silu(float x) {
    return x / (1.0f + __expf(-x));
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_fused_glu(
        const float * __restrict__ sum, const float * __restrict__ sum_gate,
        const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max,
        ggml_glu_op glu_op, float glu_limit) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    const bool y_scale_used = y_scale != nullptr;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) return;

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (fallback && i > i_max) continue;

            const int idx = (j0 / nwarps) * (I / warp_size) + i0 / warp_size;
            float val     = sum[idx];
            float val_g   = sum_gate[idx];

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (y_scale_used) {
                    val   *= y_scale[j];
                    val_g *= y_scale[j];
                }
            }

            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    val *= mmq_silu(val_g);
                    break;
                case GGML_GLU_OP_GEGLU:
                    val *= 0.5f * val_g * (1.0f + tanhf(0.7978845608028654f * val_g * (1.0f + 0.044715f * val_g * val_g)));
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    val *= mmq_silu(val_g);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    val_g = fminf(fmaxf(val_g, -glu_limit), glu_limit);
                    val *= mmq_silu(val_g);
                    break;
                default:
                    val *= val_g;
                    break;
            }

            dst[ids_dst[j] * stride + i] = val;
            GGML_UNUSED(y_scale_used);
        }
    }
}

// Fused process tile: computes both up and gate GEMMs, applies GLU in write-back.
// Uses the same smem layout as the original (reuses tile_x for gate weights).
// K-loop structure:
//   1. Load activations half 1 → tile_y
//   2. Load UP weights → tile_x, compute vec_dot → sum_up[0]
//   3. Load activations half 2 → tile_y, compute vec_dot → sum_up[NE_K]
//   4. Load GATE weights → tile_x (overwrite), reload act half 1, compute → sum_gate[0]
//   5. Reload act half 2, compute → sum_gate[NE_K]
template <ggml_type type, int J, bool fallback, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile_fused(
        const char * __restrict__ x, const char * __restrict__ x_gate,
        const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        ggml_glu_op glu_op, float glu_limit) {

    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps     = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int I          = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;
    constexpr int ne_block        = QK8_1_MMQ;
    constexpr int sz              = sizeof(block_q8_1_mmq) / sizeof(int);

    float sum[J * I / (nwarps * warp_size)] = {0.0f};
    float sum_gate[J * I / (nwarps * warp_size)] = {0.0f};

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        // --- UP pass ---
        // Load UP weights
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        // Load activations half 1
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y, sum, 0);
        __syncthreads();
        // Load activations half 2
        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();

        // --- GATE pass (reuse tile_x) ---
        // Load GATE weights (overwrite tile_x)
        load_tiles(x_gate, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        // Reload activations half 1 (L2 hit)
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y, sum_gate, 0);
        __syncthreads();
        // Reload activations half 2 (L2 hit)
        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y, sum_gate, MMQ_TILE_NE_K);
        __syncthreads();
    }

    // Write-back with GLU
    if (fixup) {
        // Write to fixup buffer: first J*I floats = sum_up, next J*I = sum_gate
        // The fixup kernel will combine and apply GLU
        ggml_cuda_mmq_write_back_fused_glu<type, J, fallback>(
            sum, sum_gate, ids_dst, tmp_fixup, y_scale, 2 * I, I, J, glu_op, glu_limit);
        // Note: for fixup, stride is 2*I because the fixup buffer stores [sum_up | sum_gate] per tile
        // The fixup kernel handles the combination.
    } else {
        ggml_cuda_mmq_write_back_fused_glu<type, J, fallback>(
            sum, sum_gate, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j, glu_op, glu_limit);
    }
}

// Fused stream-K fixup kernel: combines partial sums and applies GLU
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
static __global__ void mul_mat_q_stream_k_fixup_fused(
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
        const int stride_sample_dst, const uint3 ntx, ggml_glu_op glu_op, float glu_limit) {
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback) / 2) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J / nwarps] = {0.0f};
    float sum_gate[J / nwarps] = {0.0f};
    const int i = blockIdx.y * warp_size + threadIdx.x;

    const int nty = (nrows_x + I - 1) / I;
    const int bidx0 = blockIdx.x;

    int kbc0      = int64_t(blockIdx.x)    * (nsamples_y.z * nchannels_y.z * ntx.z * nty * blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1) * (nsamples_y.z * nchannels_y.z * ntx.z * nty * blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx) * (nsamples_y.z * nchannels_y.z * ntx.z * nty * blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) {
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

        // Fixup buffer layout per block: [sum_up (J*I floats) | sum_gate (J*I floats)]
        constexpr int fixup_stride = 2 * J * I;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            sum[j0/nwarps]      += tmp_last_tile[bidx * fixup_stride + j * I + i];
            sum_gate[j0/nwarps] += tmp_last_tile[bidx * fixup_stride + J * I + j * I + i];
        }

        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) return;

    int tmp = fastdiv(kbc0, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    if (!ids_dst) {
        const int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * J * stride_col_dst + it * I;
        dst += offset_dst;
        const int i_max = nrows_x - it * I - 1;
        const int j_max = ncols_dst - jt * J - 1;
        if (fallback && i > i_max) return;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            if (j > j_max) return;
            float val = sum[j0/nwarps];
            float val_g = sum_gate[j0/nwarps];
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU: val *= mmq_silu(val_g); break;
                case GGML_GLU_OP_GEGLU: val *= 0.5f * val_g * (1.0f + tanhf(0.7978845608028654f * val_g * (1.0f + 0.044715f * val_g * val_g))); break;
                case GGML_GLU_OP_SWIGLU_OAI: val *= mmq_silu(val_g); break;
                case GGML_GLU_OP_SWIGLU_CLAMP: val_g = fminf(fmaxf(val_g, -glu_limit), glu_limit); val *= mmq_silu(val_g); break;
                default: val *= val_g; break;
            }
            dst[j * stride_col_dst + i] = val;
        }
        return;
    }

    __shared__ int ids_dst_shared[J];
    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    for (int j = threadIdx.y * warp_size + threadIdx.x; j < J; j += nwarps * warp_size) {
        ids_dst_shared[j] = ids_dst[col_low + jt * J + j];
    }
    __syncthreads();

    const int offset_dst = it * I;
    dst += offset_dst;
    const int i_max = nrows_x - it * I - 1;
    const int j_max = col_diff - jt * J - 1;
    if (fallback && i > i_max) return;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) return;
        float val = sum[j0/nwarps];
        float val_g = sum_gate[j0/nwarps];
        switch (glu_op) {
            case GGML_GLU_OP_SWIGLU: val *= mmq_silu(val_g); break;
            case GGML_GLU_OP_GEGLU: val *= 0.5f * val_g * (1.0f + tanhf(0.7978845608028654f * val_g * (1.0f + 0.044715f * val_g * val_g))); break;
            case GGML_GLU_OP_SWIGLU_OAI: val *= mmq_silu(val_g); break;
            case GGML_GLU_OP_SWIGLU_CLAMP: val_g = fminf(fmaxf(val_g, -glu_limit), glu_limit); val *= mmq_silu(val_g); break;
            default: val *= val_g; break;
        }
        dst[ids_dst_shared[j] * stride_col_dst + i] = val;
    }
}

// Fused MMQ kernel (same structure as mul_mat_q, calls fused process_tile)
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
static __global__ void mul_mat_q_fused(
        const char * __restrict__ x, const char * __restrict__ x_gate, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx, ggml_glu_op glu_op, float glu_limit) {

    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    const uint32_t nty = (nrows_x + I - 1) / I;

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > J && j >= J) break;
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
        // Non-stream-K path (simple tiling)
        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
        const int wt = tmp2.x;
        const int zt = tmp2.y;
        const int jt = blockIdx.y;
        const int it = blockIdx.x;

        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt * stride_sample_y   + zt * stride_channel_y;
        int offset_dst     = wt * stride_sample_dst + zt * stride_channel_dst + jt * J * stride_col_dst;
        int offset_y_scale = 0;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt * nchannels_y.z * ncols_y + zt * ncols_y;
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;
            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) offset_y_scale = 0;
            if (jt * J >= col_diff) return;
            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
                const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
                if (j0 + nwarps * warp_size > J && j >= J) break;
                ids_dst_shared[j] = ids_dst[col_low + jt * J + j];
            }
            __syncthreads();
        }

        offset_y   += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it * I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt * J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x - it * I - 1;
        const int tile_y_max_j = col_diff - jt * J - 1;
        const int offset_x = fastdiv(wt, sample_ratio) * stride_sample_x + fastdiv(zt, channel_ratio) * stride_channel_x + it * I * stride_row_x;

        constexpr bool fixup = false;
        mul_mat_q_process_tile_fused<type, J, fallback, fixup>(
            x, x_gate, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
            stride_row_x, ncols_y, stride_col_dst,
            tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z, glu_op, glu_limit);
        return;
    }

    // Stream-K path
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    int kbc      = int64_t(blockIdx.x)    * (nsamples_y.z * nchannels_y.z * ntx.z * nty * blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1) * (nsamples_y.z * nchannels_y.z * ntx.z * nty * blocks_per_ne00.z) / gridDim.x;
    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));

    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        const int jt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nchannels_y);
        const int zt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nsamples_y);
        const int wt = tmp2.y;
        const int it = tmp2.x;

        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt * stride_sample_y   + zt * stride_channel_y;
        int offset_dst     = wt * stride_sample_dst + zt * stride_channel_dst + jt * J * stride_col_dst;
        int offset_y_scale = 0;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt * nchannels_y.z * ncols_y + zt * ncols_y;
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;
            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) offset_y_scale = 0;
            if (jt * J >= col_diff) {
                kbc += blocks_per_ne00.z;
                kbc -= fastmodulo(kbc, blocks_per_ne00);
                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
                continue;
            }
            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
                const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
                if (j0 + nwarps * warp_size > J && j >= J) break;
                ids_dst_shared[j] = ids_dst[col_low + jt * J + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it * I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt * J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x - it * I - 1;
        const int tile_y_max_j = col_diff - jt * J - 1;
        const int offset_x = fastdiv(wt, sample_ratio) * stride_sample_x + fastdiv(zt, channel_ratio) * stride_channel_x + it * I * stride_row_x;

        constexpr bool fixup = false;
        mul_mat_q_process_tile_fused<type, J, fallback, fixup>(
            x, x_gate, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
            stride_row_x, ncols_y, stride_col_dst,
            tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, glu_op, glu_limit);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);
        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) return;

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y       = wt * stride_sample_y   + zt * stride_channel_y;
    int offset_dst     = wt * stride_sample_dst + zt * stride_channel_dst + jt * J * stride_col_dst;
    int offset_y_scale = 0;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale = wt * nchannels_y.z * ncols_y + zt * ncols_y;
    }

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;
        offset_y   = 0;
        offset_dst = 0;
        if constexpr (type == GGML_TYPE_NVFP4) offset_y_scale = 0;
        if (jt * J >= col_diff) return;
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
            const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
            if (j0 + nwarps * warp_size > J && j >= J) break;
            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it * I;
    const float * y_scale_tile = nullptr;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale += col_low + jt * J;
        y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
    }

    const int tile_x_max_i = nrows_x - it * I - 1;
    const int tile_y_max_j = col_diff - jt * J - 1;
    const int offset_x = fastdiv(wt, sample_ratio) * stride_sample_x + fastdiv(zt, channel_ratio) * stride_channel_x + it * I * stride_row_x;

    // Last tile: write to fixup buffer (doubled size for sum_up + sum_gate)
    constexpr bool fixup = true;
    mul_mat_q_process_tile_fused<type, J, fallback, fixup>(
        x, x_gate, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
        stride_row_x, ncols_y, stride_col_dst,
        tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, glu_op, glu_limit);
}

// Launch helper for fused kernel
template <ggml_type type, int J, bool fallback>
static void launch_mul_mat_q_fused(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc); // Same as original (reuse tile_x)

    const dim3 block_dims(warp_size, nwarps, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_fused<type, J, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_fused<type, J, true>), nbytes_shared);

    const int nty  = (args.nrows_x   + config.I - 1) / config.I;
    const int ntx  = (args.ncols_max + config.J - 1) / config.J;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_nums_xy_tiling(nty, ntx, ntzw);

    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntx_fd             = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);

    if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cc)) {
        mul_mat_q_fused<type, J, fallback><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
            (args.x, args.x_gate, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale,
             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
             ntx_fd, args.glu_op, args.glu_limit);
        return;
    }

    const int ntiles_dst = ntx * nty * ntzw;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm * tiles_nwaves);
    const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);

    GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30));

    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool);
    if (fixup_needed) {
        // Doubled size: [sum_up (J*I) | sum_gate (J*I)] per block
        tmp_fixup.alloc(block_nums_stream_k.x * 2 * config.J * config.I);
    }

    const dim3 block_nums_fixup(block_nums_stream_k.x, config.I / warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y / 2, block_dims.z);

    mul_mat_q_fused<type, J, fallback><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
        (args.x, args.x_gate, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.y_scale,
         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
         ntx_fd, args.glu_op, args.glu_limit);

    if (!fixup_needed) return;

    CUDA_CHECK(cudaGetLastError());
    mul_mat_q_stream_k_fixup_fused<type, J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
        (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
         args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
         ntx_fd, args.glu_op, args.glu_limit);
}

// J selection for fused kernel
template <ggml_type type, bool fallback>
static void mul_mat_q_switch_J_fused(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id    = ggml_cuda_get_device();
    const int cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
#ifdef GGML_FUSED_J_8
        if (J == 8) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 8, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 8 - 1) / 8;
            if (ntiles_x < ntiles_J_best) {
                J_best = 8;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_16
        if (J == 16) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 16, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 16 - 1) / 16;
            if (ntiles_x < ntiles_J_best) {
                J_best = 16;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_24
        if (J == 24) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 24, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 24 - 1) / 24;
            if (ntiles_x < ntiles_J_best) {
                J_best = 24;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_32
        if (J == 32) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 32, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 32 - 1) / 32;
            if (ntiles_x < ntiles_J_best) {
                J_best = 32;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_40
        if (J == 40) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 40, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 40 - 1) / 40;
            if (ntiles_x < ntiles_J_best) {
                J_best = 40;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_48
        if (J == 48) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 48, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 48 - 1) / 48;
            if (ntiles_x < ntiles_J_best) {
                J_best = 48;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_56
        if (J == 56) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 56, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 56 - 1) / 56;
            if (ntiles_x < ntiles_J_best) {
                J_best = 56;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_64
        if (J == 64) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 64, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 64 - 1) / 64;
            if (ntiles_x < ntiles_J_best) {
                J_best = 64;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_72
        if (J == 72) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 72, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 72 - 1) / 72;
            if (ntiles_x < ntiles_J_best) {
                J_best = 72;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_80
        if (J == 80) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 80, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 80 - 1) / 80;
            if (ntiles_x < ntiles_J_best) {
                J_best = 80;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_88
        if (J == 88) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 88, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 88 - 1) / 88;
            if (ntiles_x < ntiles_J_best) {
                J_best = 88;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_96
        if (J == 96) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 96, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 96 - 1) / 96;
            if (ntiles_x < ntiles_J_best) {
                J_best = 96;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_104
        if (J == 104) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 104, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 104 - 1) / 104;
            if (ntiles_x < ntiles_J_best) {
                J_best = 104;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_112
        if (J == 112) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 112, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 112 - 1) / 112;
            if (ntiles_x < ntiles_J_best) {
                J_best = 112;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_120
        if (J == 120) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 120, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 120 - 1) / 120;
            if (ntiles_x < ntiles_J_best) {
                J_best = 120;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
#ifdef GGML_FUSED_J_128
        if (J == 128) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 128, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) continue;
            if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;

            const int ntiles_x = (args.ncols_opt + 128 - 1) / 128;
            if (ntiles_x < ntiles_J_best) {
                J_best = 128;
                ntiles_J_best = ntiles_x;
            }
        }
#endif
    }

    if (J_best == 0) {
        GGML_ABORT("fused MMQ: no J value compiled. Set -DGGML_CUDA_FUSED_J to include J values.\n");
    }

    switch (J_best) {
#ifdef GGML_FUSED_J_8
        case   8: launch_mul_mat_q_fused<type,   8, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_16
        case  16: launch_mul_mat_q_fused<type,  16, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_24
        case  24: launch_mul_mat_q_fused<type,  24, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_32
        case  32: launch_mul_mat_q_fused<type,  32, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_40
        case  40: launch_mul_mat_q_fused<type,  40, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_48
        case  48: launch_mul_mat_q_fused<type,  48, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_56
        case  56: launch_mul_mat_q_fused<type,  56, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_64
        case  64: launch_mul_mat_q_fused<type,  64, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_72
        case  72: launch_mul_mat_q_fused<type,  72, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_80
        case  80: launch_mul_mat_q_fused<type,  80, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_88
        case  88: launch_mul_mat_q_fused<type,  88, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_96
        case  96: launch_mul_mat_q_fused<type,  96, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_104
        case 104: launch_mul_mat_q_fused<type, 104, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_112
        case 112: launch_mul_mat_q_fused<type, 112, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_120
        case 120: launch_mul_mat_q_fused<type, 120, fallback>(ctx, args, stream); break;
#endif
#ifdef GGML_FUSED_J_128
        case 128: launch_mul_mat_q_fused<type, 128, fallback>(ctx, args, stream); break;
#endif
        default: GGML_ABORT("fused MMQ: J_best=%d not compiled. Set -DGGML_CUDA_FUSED_J to include it.\n", J_best); break;
    }
}

template <ggml_type type>
static void mul_mat_q_case_fused(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    if (args.nrows_x % 128 == 0) {
        mul_mat_q_switch_J_fused<type, false>(ctx, args, stream);
    } else {
        mul_mat_q_switch_J_fused<type, true>(ctx, args, stream);
    }
}

// Type dispatch for fused kernel
static void ggml_cuda_mul_mat_q_fused_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_K:    mul_mat_q_case_fused<GGML_TYPE_Q4_K>(ctx, args, stream); break;
        case GGML_TYPE_Q8_0:    mul_mat_q_case_fused<GGML_TYPE_Q8_0>(ctx, args, stream); break;
        default: GGML_ABORT("unsupported type %d for fused MMQ\n", (int)args.type_x); break;
    }
}

// Host entry point for fused GLU MMQ (definition in .cu to avoid multiple definition at link time)
void ggml_cuda_mul_mat_q_fused_glu(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * src0_gate, ggml_tensor * dst,
        ggml_glu_op glu_op, float glu_limit) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == src0_gate->type);
    GGML_ASSERT(src0->ne[0] == src0_gate->ne[0]);
    GGML_ASSERT(src0->ne[1] == src0_gate->ne[1]);

    // Debug: confirm fused GLU MMQ is being called
    static int fused_glu_call_count = 0;
    if (fused_glu_call_count < 5) {
        fprintf(stderr, "[FUSED-GLU-MMQ] call #%d: batch=%ld, hidden=%ld, inter=%ld, type=%d\n",
            fused_glu_call_count, (long)src1->ne[1], (long)src0->ne[0], (long)src0->ne[1], (int)src0->type);
    }
    fused_glu_call_count++;

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    const char  * src0_d      = (const char *) src0->data;
    const char  * src0_gate_d = (const char *) src0_gate->data;
    const float * src1_d      = (const float *) src1->data;
    float       *  dst_d      = (float *) dst->data;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const size_t y_block_size       = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = QK8_1_MMQ;

    // Quantize activations to Q8_1 (shared between up and gate)
    const size_t nbytes_src1_q8_1 = ne13 * ne12 * ne11 * ne10_padded * y_block_size / y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                               ne11, ne12, ne13, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12 * s12;

    mmq_args args;
    args.x = src0_d;
    args.type_x = src0->type;
    args.y = (const int *) src1_q8_1.ptr;
    args.ids_dst = nullptr;
    args.expert_bounds = nullptr;
    args.dst = dst_d;
    args.y_scale = nullptr;
    args.ncols_x = ne00; args.nrows_x = ne01; args.ncols_dst = ne1; args.stride_row_x = s01;
    args.ncols_y = ne11; args.nrows_dst = s1;
    args.nchannels_x = ne02; args.nchannels_y = ne12; args.stride_channel_x = s02; args.stride_channel_y = s12; args.stride_channel_dst = s2;
    args.nsamples_x = ne03; args.nsamples_y = ne13; args.stride_sample_x = s03; args.stride_sample_y = s13; args.stride_sample_dst = s3;
    args.ncols_max = ne1;
    args.ncols_opt = ne1;
    args.x_gate = src0_gate_d;
    args.glu_op = glu_op;
    args.glu_limit = glu_limit;
    args.fused_glu = true;

    ggml_cuda_mul_mat_q_fused_switch_type(ctx, args, stream);
}
