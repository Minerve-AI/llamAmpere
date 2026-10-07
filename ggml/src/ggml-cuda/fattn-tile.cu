#include "common.cuh"
#include "fattn-tile.cuh"
#include "fattn-kvarn-dispatch.cuh"
#include "fattn-kvarn-portable.cuh"

// KVarN entry point for the tile dispatch.
// Delegates to the portable KVarN kernel which handles all configurations.
template<int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_tile_case_kvarn(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {

    int key_bits, value_bits;
    size_t record_bytes;
    ggml_cuda_fattn_get_kvarn_params(dst, key_bits, value_bits, record_bytes);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *)dst->op_params + 2, sizeof(float));

    // ncols2=1: process one KV head per block (safe default for portable kernel)
    if (logit_softcap == 0.0f) {
        launch_fattn_kvarn_portable<DKQ, DV, 1, false>(ctx, dst, key_bits, value_bits, record_bytes);
    } else {
        launch_fattn_kvarn_portable<DKQ, DV, 1, true>(ctx, dst, key_bits, value_bits, record_bytes);
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
