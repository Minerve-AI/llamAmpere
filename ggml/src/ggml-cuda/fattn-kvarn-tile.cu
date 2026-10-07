#include "common.cuh"
#include "fattn-kvarn-dispatch.cuh"
#include "fattn-kvarn-portable.cuh"

// KVarN entry point - separate .cu file to avoid nvcc OOM when combined with fattn-tile.cu
void ggml_cuda_flash_attn_ext_tile_kvarn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    int key_bits, value_bits;
    size_t record_bytes;
    ggml_cuda_fattn_get_kvarn_params(dst, key_bits, value_bits, record_bytes);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *)dst->op_params + 2, sizeof(float));

    const bool softcap = (logit_softcap != 0.0f);

    switch (K->ne[0]) {
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (!softcap) launch_fattn_kvarn_portable< 64,  64, 32, 1, false>(ctx, dst, key_bits, value_bits, record_bytes);
            else          launch_fattn_kvarn_portable< 64,  64, 32, 1, true >(ctx, dst, key_bits, value_bits, record_bytes);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (!softcap) launch_fattn_kvarn_portable< 96,  96, 32, 1, false>(ctx, dst, key_bits, value_bits, record_bytes);
            else          launch_fattn_kvarn_portable< 96,  96, 32, 1, true >(ctx, dst, key_bits, value_bits, record_bytes);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (!softcap) launch_fattn_kvarn_portable<112, 112, 32, 1, false>(ctx, dst, key_bits, value_bits, record_bytes);
            else          launch_fattn_kvarn_portable<112, 112, 32, 1, true >(ctx, dst, key_bits, value_bits, record_bytes);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (!softcap) launch_fattn_kvarn_portable<128, 128, 32, 1, false>(ctx, dst, key_bits, value_bits, record_bytes);
            else          launch_fattn_kvarn_portable<128, 128, 32, 1, true >(ctx, dst, key_bits, value_bits, record_bytes);
        } break;
        default:
            break;
    }
}
