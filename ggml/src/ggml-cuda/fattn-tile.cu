#include "common.cuh"
#include "fattn-tile.cuh"
#include "fattn-kvarn-dispatch.cuh"
#include "fattn-kvarn-portable.cuh"

// KVarN magic constant: op_params[0] == 0x4B564152 ("KVAR") signals KVarN mode
static constexpr uint32_t KVARN_MAGIC = 0x4B564152u;

// Check if the flash attention op is using KVarN format
static inline bool ggml_cuda_fattn_is_kvarn(const ggml_tensor * dst) {
    const uint32_t magic = (uint32_t)dst->op_params[0];
    return magic == KVARN_MAGIC;
}

// Get KVarN parameters from the flash attention op
static inline void ggml_cuda_fattn_get_kvarn_params(
    const ggml_tensor * dst,
    int & key_bits, int & value_bits, size_t & record_bytes) {
    key_bits   = (int)dst->op_params[1];
    value_bits = (int)dst->op_params[2];
    record_bytes = (size_t)dst->op_params[3];
}

// KVarN-aware flash attention tile dispatch.
// When KVarN mode is active, this launches the portable KVarN kernel
// that reads quantized K/V records directly without dequantizing to F16 first.
template<int DKQ, int DV, int ncols2, bool use_logit_softcap>
static void launch_fattn_kvarn_tile(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    int key_bits, int value_bits, size_t record_bytes) {

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int warp_size = 32;

    // Select kernel configuration based on head size and Q width
    constexpr int cols_per_block = 32;
    const int nwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, cols_per_block, cc) / warp_size;
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);

    // Launch the KVarN portable kernel with extra parameters
    // The kernel signature includes kvarn_key_bits, kvarn_value_bits, kvarn_record_bytes,
    // kvarn_token_group (always 128), and kvarn_record_dim (head_slices * 128)
    constexpr size_t nbytes_shared = 0;

    // We need to cast to fattn_kernel_t and pass extra params through the launch
    // Since fattn_kernel_t has a fixed signature, we use the direct kernel launch
    // with the correct template instantiation.

    const int head_slices = DKQ / 128; // 1 for 128-dim, 2 for 256-dim, 4 for 512-dim
    const int record_dim = head_slices * 128;

    const dim3 block_dim(warp_size, nwarps, 1);
    const int ntiles_x = ((Q->ne[1] + ncols1) / ncols1);
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    const dim3 blocks_num(ntiles_x, 1, ntiles_z_gqa * K->ne[2] * Q->ne[3]);

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *)dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    flash_attn_kvarn_portable<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap>
        <<<blocks_num, block_dim, nbytes_shared, ctx.stream()>>>(
            (const char *)Q->data,
            (const char *)K->data,
            (const char *)V->data,
            mask ? (const char *)mask->data : nullptr,
            dst->src[4] ? (const char *)dst->src[4]->data : nullptr,
            nullptr, // KV_max (not used in portable path)
            (float *)dst->data,
            nullptr, // dst_meta
            scale, max_bias,
            powf(2.0f, -(max_bias) / n_head_log2),
            powf(2.0f, -(max_bias / 2.0f) / n_head_log2),
            n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K->ne[0], K->ne[1], K->ne[2], K->ne[3], K->nb[1], K->nb[2], K->nb[3],
            V->nb[1], V->nb[2], V->nb[3],
            mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
            mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
            key_bits, value_bits, record_bytes,
            KVAR_N_GROUP, record_dim
        );
    CUDA_CHECK(cudaGetLastError());
}

// Helper for ncols1 selection (mirrors fattn-tile.cuh pattern)
template<int DKQ, int DV, int ncols2, bool use_logit_softcap>
static void launch_fattn_kvarn_tile_switch_ncols1(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    int key_bits, int value_bits, size_t record_bytes) {

    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] > 16 / ncols2) {
        constexpr int cols_per_block = 32;
        launch_fattn_kvarn_tile<DKQ, DV, ncols2, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
        return;
    }

    if (ncols2 <= 16 && Q->ne[1] > 8 / ncols2) {
        launch_fattn_kvarn_tile<DKQ, DV, ncols2, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
        return;
    }

    if (ncols2 <= 8 && Q->ne[1] > 4 / ncols2) {
        launch_fattn_kvarn_tile<DKQ, DV, ncols2, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
        return;
    }

    GGML_ABORT("KVarN: unexpected Q width");
}

template<int DKQ, int DV, bool use_logit_softcap>
static void launch_fattn_kvarn_tile_switch_ncols2(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    int key_bits, int value_bits, size_t record_bytes) {

    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *)KQV->op_params + 1, sizeof(float));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    const bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    if (DKQ == 576) {
        if (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
    }

    if (DKQ == 192) {
        if (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
    }

    if (DKQ <= 512 && DKQ != 320 && DKQ != 192) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
        if (DV <= 256) {
            launch_fattn_kvarn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
            return;
        }
    }

    GGML_ABORT("KVarN: unexpected configuration");
}

// KVarN-aware tile case entry point
template<int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_tile_case_kvarn(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {

    float logit_softcap;
    memcpy(&logit_softcap, (const float *)dst->op_params + 2, sizeof(float));

    int key_bits, value_bits;
    size_t record_bytes;
    ggml_cuda_fattn_get_kvarn_params(dst, key_bits, value_bits, record_bytes);

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_kvarn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_kvarn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst, key_bits, value_bits, record_bytes);
    }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Check if KVarN mode is active
    if (ggml_cuda_fattn_is_kvarn(dst)) {
        // KVarN path: dispatch to KVarN-aware kernel
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];

        switch (K->ne[0]) {
            case  40: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn< 40,  40>(ctx, dst);
            } break;
            case  64: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn< 64,  64>(ctx, dst);
            } break;
            case  72: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn< 72,  72>(ctx, dst);
            } break;
            case  80: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn< 80,  80>(ctx, dst);
            } break;
            case  96: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn< 96,  96>(ctx, dst);
            } break;
            case 112: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<112, 112>(ctx, dst);
            } break;
            case 128: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<128, 128>(ctx, dst);
            } break;
            case 192: {
                GGML_ASSERT(V->ne[0] == 128);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<192, 128>(ctx, dst);
            } break;
            case 256: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<256, 256>(ctx, dst);
            } break;
            case 320: {
                GGML_ASSERT(V->ne[0] == 256);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<320, 256>(ctx, dst);
            } break;
            case 512: {
                GGML_ASSERT(V->ne[0] == K->ne[0]);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<512, 512>(ctx, dst);
            } break;
#ifndef GGML_USE_HIP
            case 576: {
                GGML_ASSERT(V->ne[0] == 512);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<576, 512>(ctx, dst);
            } break;
            case 640: {
                GGML_ASSERT(V->ne[0] == 512);
                ggml_cuda_flash_attn_ext_tile_case_kvarn<640, 512>(ctx, dst);
            } break;
#endif
            default: {
                GGML_ABORT("KVarN: Unsupported head size");
            } break;
        }
        return;
    }

    // Standard F16 path - fall through to existing implementation
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
#ifndef GGML_USE_HIP
        // D>=576 tile kernels exceed HIP local memory limit (67584 > 65536)
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        case 640: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<640, 512>(ctx, dst);
        } break;
#endif
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}
