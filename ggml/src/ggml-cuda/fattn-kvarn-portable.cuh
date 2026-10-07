#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// KVarN token group size (same as src/llama-kvarn.h)
constexpr int KVAR_N_GROUP = 128;

// ---------------------------------------------------------------------------
// KVarN dequantization utilities for flash attention
//
// KVarN record layout per head per token group:
//   - Payload: token_group * record_dim * bits / 8 bytes (bit-packed)
//   - s_row (per-token scale): token_group * 4 bytes
//   - zp    (per-dim zero-point): record_dim * 4 bytes
//   - s_col (per-dim scale): record_dim * 4 bytes
//
// Dequantize: value = (q - zp[col]) * s_col[col] * s_row[row]
// ---------------------------------------------------------------------------

// Unpack `bits` bits starting at bit_offset from payload.
static __device__ __forceinline__ uint32_t kvarn_unpack_bits(
    const uint8_t * payload,
    int bit_offset,
    int bits) {
    uint32_t val = 0;
    int bits_left = bits;
    int cur_byte = bit_offset / 8;
    int cur_bit  = bit_offset % 8;
    while (bits_left > 0) {
        uint8_t byte = payload[cur_byte];
        int available = 8 - cur_bit;
        int take = (bits_left < available) ? bits_left : available;
        val |= ((uint32_t)(byte >> cur_bit) & ((1u << take) - 1)) << (bits_left - take);
        bits_left -= take;
        cur_bit = 0;
        cur_byte++;
    }
    return val;
}

// Dequantize a single KVarN element.
// row = token position within group [0, token_group)
// col = dimension [0, record_dim)
static __device__ __forceinline__ float kvarn_dequant_element(
    const uint8_t * payload,
    const float * s_row,
    const float * zp,
    const float * s_col,
    int row,
    int col,
    int bits,
    int record_dim) {
    const int bit_offset = row * record_dim * bits + col * bits;
    const float q = (float)kvarn_unpack_bits(payload, bit_offset, bits);
    return (q - zp[col]) * s_col[col] * s_row[row];
}

// ---------------------------------------------------------------------------
// Portable KVarN flash attention kernel (full log-sum-exp)
//
// Reads KVarN-quantized K/V records, dequantizes to float in shared memory,
// then performs standard flash attention with the log-sum-exp trick.
//
// Grid: (n_q_cols/ncols1, n_kv_blocks, n_seq * n_heads)
// Block: 256 threads (8 warps)
// ---------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(256, 2)
static __global__ void flash_attn_kvarn_portable(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const unsigned int ne01_x, const unsigned int ne01_y, const unsigned int ne01_z, const int32_t ne02, const int32_t ne03,
                        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                        const int32_t nb11, const int32_t nb12, const int64_t nb13,
                        const int32_t nb21, const int32_t nb22, const int64_t nb23,
                        const int32_t ne31, const int32_t ne32, const int32_t ne33,
                        const int32_t nb31, const int32_t nb32, const int64_t nb33,
        // KVarN parameters
        const int32_t kvarn_key_bits,
        const int32_t kvarn_value_bits,
        const size_t  kvarn_record_bytes,
        const int32_t kvarn_token_group,
        const int32_t kvarn_record_dim) {

    const char * GGML_CUDA_RESTRICT Q    = Q_ptr;
    const char * GGML_CUDA_RESTRICT K    = K_ptr;
    const char * GGML_CUDA_RESTRICT V    = V_ptr;
    const char * GGML_CUDA_RESTRICT mask = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    if (DKQ != (int)ne00) return;

    const uint3 ne01 = make_uint3(ne01_x, ne01_y, ne01_z);

    constexpr int ncols     = ncols1 * ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = 8; // 256 threads / 32
    constexpr int cpw       = ncols > nwarps ? ncols / nwarps : 1;
    constexpr int np        = nwarps > ncols ? nwarps / ncols : 1;
    constexpr int DVp       = (DV + 2 * warp_size - 1) & ~(2 * warp_size - 1);

    const int tid     = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;

    const int col_Q_0 = blockIdx.x * ncols1;
    const int sequence = blockIdx.z / (ne02 / ncols2);
    const int head0    = blockIdx.z * ncols2 - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;
    const int kv_head   = head0 / gqa_ratio;

    const float * Q_f = (const float *)(Q + nb03 * sequence + nb02 * head0);
    const half  * maskh = mask ? (const half *)(mask + nb33 * (sequence % ne33)) : nullptr;
    const float slope = (ncols2 > 1) ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    // KVarN parameters
    const int token_group  = kvarn_token_group;   // 128
    const int record_dim   = kvarn_record_dim;    // head_slices * 128
    const int key_bits     = kvarn_key_bits;
    const int value_bits   = kvarn_value_bits;
    const size_t rec_bytes = kvarn_record_bytes;

    // Record layout offsets (within one record)
    const int k_payload_bytes = (token_group * record_dim * key_bits + 7) / 8;
    const int k_s_row_off     = k_payload_bytes;
    const int k_zp_off        = k_s_row_off + token_group * sizeof(float);
    const int k_s_col_off     = k_zp_off + record_dim * sizeof(float);

    const int v_payload_bytes = (token_group * record_dim * value_bits + 7) / 8;
    const int v_s_row_off     = v_payload_bytes;
    const int v_zp_off        = v_s_row_off + token_group * sizeof(float);
    const int v_s_col_off     = v_zp_off + record_dim * sizeof(float);

    // Stride between heads and token groups in K/V (in bytes)
    // K layout: [n_heads][n_token_groups][rec_bytes]
    // Stride per head = n_token_groups * rec_bytes (approximated by nb12 * ne12)
    // We use the actual tensor strides: nb12 is stride per head, nb13 is stride per sequence
    const int64_t stride_K_head = nb12;  // bytes per head
    const int64_t stride_K_tok  = nb11;  // bytes per token (within a head)
    const int64_t stride_V_head = nb22;
    const int64_t stride_V_tok  = nb21;

    // -----------------------------------------------------------------------
    // Load Q into shared memory
    // -----------------------------------------------------------------------
    constexpr int DKQp = (DKQ + 2 * warp_size - 1) & ~(2 * warp_size - 1);
    __shared__ float Q_tmp[ncols * DKQp];

    {
        constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
        constexpr int cpy_ne = cpy_nb / 4;
        constexpr int cpw_q  = ncols > 8 ? ncols / 8 : 1;
        constexpr int np_q   = 8 > ncols ? 8 / ncols : 1;

#pragma unroll
        for (int jc0 = 0; jc0 < cpw_q; ++jc0) {
            const int jc = jc0 + (warp_id / np_q) * cpw_q;
            const int j  = jc / ncols2;
            const int c  = jc % ncols2;
            constexpr int cpy_ne_D = cpy_ne < DKQp / warp_size ? cpy_ne : DKQp / warp_size;

#pragma unroll
            for (int i0 = 0; i0 < DKQp; i0 += np_q * warp_size * cpy_ne_D) {
                if (i0 + np_q * warp_size * cpy_ne_D <= DKQ ||
                    i0 + (warp_id % np_q) * (warp_size * cpy_ne_D) + lane_id * cpy_ne_D < DKQ) {
                    __align__(16) float tmp_f[cpy_ne_D] = {0.0f};
                    ggml_cuda_memcpy_1<sizeof(tmp_f)>(
                        tmp_f,
                        &Q_f[c * (nb02 / sizeof(float)) +
                             fastmodulo(col_Q_0 + j, ne01) * (nb01 / sizeof(float)) +
                             i0 + (warp_id % np_q) * (warp_size * cpy_ne_D) + lane_id * cpy_ne_D]);
#pragma unroll
                    for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                        tmp_f[i1] *= scale;
                    }
                    ggml_cuda_memcpy_1<sizeof(tmp_f)>(
                        &Q_tmp[jc * DKQp + i0 + (warp_id % np_q) * (warp_size * cpy_ne_D) + lane_id * cpy_ne_D],
                        tmp_f);
                }
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // State for log-sum-exp
    // -----------------------------------------------------------------------
    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < cpw; ++j0) {
        KQ_max[j0] = -FLT_MAX / 2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    // VKQ accumulators: VKQ[jc][dv]
    __align__(16) float2 VKQ[cpw * ((DVp / 2) / warp_size)] = {{0.0f, 0.0f}};

    // -----------------------------------------------------------------------
    // Shared memory for K/V dequantization tiles
    // We process one token group (128 tokens) at a time.
    // -----------------------------------------------------------------------
    constexpr int nbatch_fa = KVAR_N_GROUP; // 128 tokens per chunk
    __shared__ float K_tile[nbatch_fa * DKQ];
    __shared__ float V_tile[nbatch_fa * DV];

    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;

    // -----------------------------------------------------------------------
    // Main loop over KV cache in chunks of nbatch_fa tokens
    // -----------------------------------------------------------------------
    for (int k_VKQ_0 = blockIdx.y * nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y * nbatch_fa) {
        const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // valid tokens in this chunk

        // -------------------------------------------------------------------
        // Step 1: Dequantize K for this chunk into K_tile
        // -------------------------------------------------------------------
        {
            // Each thread handles multiple (token, dim) pairs
            // Total work: nbatch_fa * DKQ values
            // 256 threads, so each thread does (nbatch_fa * DKQ / 256) values
            constexpr int total_k = nbatch_fa * DKQ;
            constexpr int work_per_thread = (total_k + 255) / 256;

            for (int wi = 0; wi < work_per_thread; ++wi) {
                const int idx = tid * work_per_thread + wi;
                if (idx >= total_k) break;

                const int t   = idx / DKQ;  // token within chunk
                const int dim = idx % DKQ;  // dimension

                const int token_abs = k_VKQ_0 + t;
                if (token_abs >= k_VKQ_max) continue;

                const int token_group_idx = token_abs / token_group;
                const int pos_in_group    = token_abs % token_group;

                // K record pointer for this head and token group
                // K tensor: ne10=record_dim, ne11=n_tokens, ne12=n_heads, ne13=n_seq
                // We use the record-based layout: K + kv_head * stride_K_head + token_group_idx * (token_group * stride_K_tok)
                // But actually the KVarN records are stored contiguously per head per token group.
                // The stride between token groups = token_group * stride_K_tok (if stride_K_tok is per-token)
                // Or we can compute: record_offset = kv_head * (ne11 / token_group) * rec_bytes + token_group_idx * rec_bytes
                const char * k_record = K + kv_head * stride_K_head + (int64_t)token_group_idx * token_group * stride_K_tok;

                const uint8_t * k_payload = (const uint8_t *)k_record;
                const float * k_s_row     = (const float *)(k_record + k_s_row_off);
                const float * k_zp        = (const float *)(k_record + k_zp_off);
                const float * k_s_col     = (const float *)(k_record + k_s_col_off);

                K_tile[t * DKQ + dim] = kvarn_dequant_element(k_payload, k_s_row, k_zp, k_s_col, pos_in_group, dim, key_bits, record_dim);
            }
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Step 2: Dequantize V for this chunk into V_tile
        // -------------------------------------------------------------------
        {
            constexpr int total_v = nbatch_fa * DV;
            constexpr int work_per_thread = (total_v + 255) / 256;

            for (int wi = 0; wi < work_per_thread; ++wi) {
                const int idx = tid * work_per_thread + wi;
                if (idx >= total_v) break;

                const int t   = idx / DV;
                const int dim = idx % DV;

                const int token_abs = k_VKQ_0 + t;
                if (token_abs >= k_VKQ_max) continue;

                const int token_group_idx = token_abs / token_group;
                const int pos_in_group    = token_abs % token_group;

                const char * v_record = V + kv_head * stride_V_head + (int64_t)token_group_idx * token_group * stride_V_tok;

                const uint8_t * v_payload = (const uint8_t *)v_record;
                const float * v_s_row     = (const float *)(v_record + v_s_row_off);
                const float * v_zp        = (const float *)(v_record + v_zp_off);
                const float * v_s_col     = (const float *)(v_record + v_s_col_off);

                V_tile[t * DV + dim] = kvarn_dequant_element(v_payload, v_s_row, v_zp, v_s_col, pos_in_group, dim, value_bits, record_dim);
            }
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Step 3: Compute KQ = Q @ K^T for each Q column
        // -------------------------------------------------------------------
        float KQ_acc[cpw * (nbatch_fa / (np * warp_size))] = {0.0f};

#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (warp_id / np) * cpw;
            const int c  = jc % ncols2;

            // Each warp lane handles a subset of tokens
            constexpr int tokens_per_lane = nbatch_fa / (np * warp_size);
            for (int ti = 0; ti < tokens_per_lane; ++ti) {
                const int t = ti * (np * warp_size) + (warp_id % np) * warp_size + lane_id;
                if (t >= k_VKQ_sup) continue;

                float kq = 0.0f;
                for (int d = 0; d < DKQ; ++d) {
                    kq += Q_tmp[jc * DKQp + d] * K_tile[t * DKQ + d];
                }

                // Apply logit softcap
                if (use_logit_softcap && logit_softcap != 0.0f) {
                    kq = logit_softcap * tanhf(kq / logit_softcap);
                }

                // Apply mask (ALiBi or causal)
                if (maskh) {
                    const int j = fastmodulo(col_Q_0 + jc / ncols2, ne01);
                    kq += slope * __half2float(maskh[j * (nb31 / sizeof(half)) + k_VKQ_0 + t]);
                }

                KQ_acc[(ti / (np * warp_size)) * cpw + jc0] = kq;
            }
        }

        // -------------------------------------------------------------------
        // Step 4: Update KQ_max (with FATTN_KQ_MAX_OFFSET for numerical stability)
        // -------------------------------------------------------------------
        float KQ_max_new[cpw];
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            KQ_max_new[jc0] = KQ_max[jc0];
        }

#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            constexpr int tokens_per_lane = nbatch_fa / (np * warp_size);
            for (int ti = 0; ti < tokens_per_lane; ++ti) {
                const int t = ti * (np * warp_size) + (warp_id % np) * warp_size + lane_id;
                if (t >= k_VKQ_sup) continue;
                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(ti / (np * warp_size)) * cpw + jc0] + FATTN_KQ_MAX_OFFSET);
            }
        }

        // Warp reduce max
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
        }

        // Cross-warp sync (if np > 1)
        if constexpr (np > 1) {
            static_assert(cpw == 1, "bad cpw for np > 1");
            __shared__ float KQ_max_new_shared[nwarps];
            if (lane_id == 0) {
                KQ_max_new_shared[warp_id] = KQ_max_new[0];
            }
            __syncthreads();
            KQ_max_new[0] = KQ_max_new_shared[(warp_id & ~(np - 1)) + lane_id % np];
            KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
        }

        // -------------------------------------------------------------------
        // Step 5: Softmax + rescale
        // -------------------------------------------------------------------
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new[jc0]);
            KQ_max[jc0] = KQ_max_new[jc0];

            float KQ_sum_add = 0.0f;
            constexpr int tokens_per_lane = nbatch_fa / (np * warp_size);
            for (int ti = 0; ti < tokens_per_lane; ++ti) {
                const int t = ti * (np * warp_size) + (warp_id % np) * warp_size + lane_id;
                const float val = (t < k_VKQ_sup) ?
                    expf(KQ_acc[(ti / (np * warp_size)) * cpw + jc0] - KQ_max[jc0]) : 0.0f;
                KQ_sum_add += val;

                // Store softmax value for VKQ accumulation
                KQ_acc[(ti / (np * warp_size)) * cpw + jc0] = val;
            }
            KQ_sum[jc0] = KQ_sum[jc0] * KQ_max_scale + KQ_sum_add;

            // Rescale VKQ
#pragma unroll
            for (int i0 = 0; i0 < DVp / 2; i0 += warp_size) {
                VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size].x *= KQ_max_scale;
                VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size].y *= KQ_max_scale;
            }
        }

        // -------------------------------------------------------------------
        // Step 6: VKQ += softmax(KQ) @ V
        // -------------------------------------------------------------------
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            constexpr int tokens_per_lane = nbatch_fa / (np * warp_size);
            for (int ti = 0; ti < tokens_per_lane; ++ti) {
                const int t = ti * (np * warp_size) + (warp_id % np) * warp_size + lane_id;
                if (t >= k_VKQ_sup) continue;

                const float s = KQ_acc[(ti / (np * warp_size)) * cpw + jc0];
                if (s == 0.0f) continue;

#pragma unroll
                for (int i0 = 0; i0 < DVp / 2; i0 += warp_size) {
                    VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size].x += s * V_tile[t * DV + 2 * (i0 / warp_size) + 2 * lane_id];
                    VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size].y += s * V_tile[t * DV + 2 * (i0 / warp_size) + 2 * lane_id + 1];
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Final: warp reduce sum
    // -----------------------------------------------------------------------
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    // Cross-warp combine (if np > 1)
    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw for np > 1");
        __shared__ float VKQ_combine[nwarps * DVp];
        __shared__ float KQ_sum_combine[nwarps];

        if (warp_id % np != 0) {
            // Non-master warps write their partial results
            for (int i0 = 0; i0 < DVp; i0 += warp_size) {
                VKQ_combine[warp_id * DVp + i0 + lane_id] = (float)((float*)&VKQ[i0 / warp_size])[lane_id % 2 == 0 ? 0 : 1];
            }
            if (lane_id == 0) {
                KQ_sum_combine[warp_id] = KQ_sum[0];
            }
            return;
        }

        __syncthreads();

        for (int ip = 1; ip < np; ++ip) {
            for (int i0 = 0; i0 < DVp / 2; i0 += warp_size) {
                const int idx = (warp_id + ip) * DVp + 2 * (i0 / warp_size) + 2 * lane_id;
                VKQ[i0 / warp_size].x += VKQ_combine[idx];
                VKQ[i0 / warp_size].y += VKQ_combine[idx + 1];
            }
            KQ_sum[0] += KQ_sum_combine[warp_id + ip];
        }
    }

    // -----------------------------------------------------------------------
    // Attention sinks
    // -----------------------------------------------------------------------
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (warp_id / np) * cpw;
            const float sink = ((const float *)sinks)[head0 + jc % ncols2];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = expf(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0] * KQ_max_scale + val;

#pragma unroll
            for (int i0 = 0; i0 < DVp / 2; i0 += warp_size) {
                VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size].x *= KQ_max_scale;
                VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size].y *= KQ_max_scale;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Write back results: dst = VKQ / KQ_sum
    // -----------------------------------------------------------------------
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (warp_id / np) * cpw;
        const int j  = jc / ncols2;
        const int c  = jc % ncols2;

        if (ncols1 > 1 && col_Q_0 + j >= (int)ne01.z) {
            return;
        }

        const float out_scale = (gridDim.y == 1) ? 1.0f / KQ_sum[jc0] : 1.0f;
        const int j_dst = ((sequence * (int)ne01.z + col_Q_0 + j) * ne02 + head0 + c) * gridDim.y + blockIdx.y;

        constexpr int cpy_ne_D = cpy_ne < DVp / warp_size ? cpy_ne : DVp / warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size * cpy_ne_D) {
            if (i0 + warp_size * cpy_ne_D <= DV || i0 + lane_id * cpy_ne_D < DV) {
                __align__(16) float tmp[cpy_ne_D];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D / 2; ++i1) {
                    tmp[2 * i1]     = VKQ[jc0 * ((DVp / 2) / warp_size) + (i0 / (2 * warp_size)) + i1].x * out_scale;
                    tmp[2 * i1 + 1] = VKQ[jc0 * ((DVp / 2) / warp_size) + (i0 / (2 * warp_size)) + i1].y * out_scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D * 4>(
                    &dst[j_dst * DV + i0 + lane_id * cpy_ne_D],
                    tmp);
            }
        }

        if (gridDim.y != 1 && lane_id == 0) {
            dst_meta[j_dst] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
}

// ---------------------------------------------------------------------------
// Launch helper
// ---------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static void launch_fattn_kvarn_portable(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    int key_bits, int value_bits, size_t record_bytes) {

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int cols_per_block = 32;
    const int nbatch_fa = KVAR_N_GROUP;
    const int nwarps = 8;
    const int nthreads = nwarps * 32;

    // Grid dimensions
    const uint32_t n_q_cols = Q->ne[1];
    const uint32_t n_heads  = Q->ne[2];
    const uint32_t n_seq    = Q->ne[3];

    const int grid_x = (n_q_cols + ncols1 - 1) / ncols1;
    const int grid_z = n_seq * n_heads;

    // KV parallel blocks
    const int n_tokens = K->ne[1];
    const int grid_y = (n_tokens + nbatch_fa - 1) / nbatch_fa;

    dim3 blocks(grid_x, grid_y, grid_z);
    dim3 threads(nthreads);

    // KVarN parameters
    const int token_group = KVAR_N_GROUP;
    const int record_dim  = (int)(K->ne[0]); // head dimension

    // Fast division values for Q columns
    const uint3 ne01_fd = init_fastdiv_values(Q->ne[1]);

    // Shared memory
    constexpr size_t smem = (KVAR_N_GROUP * DKQ + KVAR_N_GROUP * DV + cols_per_block * DKQ) * sizeof(float);

    flash_attn_kvarn_portable<DKQ, DV, ncols1, ncols2, use_logit_softcap>
        <<<blocks, threads, smem, ctx.stream()>>>(
            (const char *)Q->data,
            (const char *)K->data,
            (const char *)V->data,
            mask ? (const char *)mask->data : nullptr,
            sinks ? (const char *)sinks->data : nullptr,
            (const int *)dst->extra,
            (float *)dst->data,
            (float2 *)((char *)dst->data + dst->nb[0] * dst->ne[0]),
            1.0f / sqrtf((float)DKQ),
            0.0f, 0.0f, 0.0f,
            0,
            0.0f,
            (int)Q->ne[0],
            ne01_fd.x, ne01_fd.y, ne01_fd.z,
            (int)Q->ne[2], (int)Q->ne[3],
            (int)Q->nb[1], (int)Q->nb[2], (int)Q->nb[3],
            (int)K->ne[0], (int)K->ne[1], (int)K->ne[2], (int)K->ne[3],
            (int)K->nb[1], (int)K->nb[2], (int64_t)K->nb[3],
            (int)V->nb[1], (int)V->nb[2], (int64_t)V->nb[3],
            (int)(mask ? mask->ne[1] : 0), (int)(mask ? mask->ne[2] : 0), (int)(mask ? mask->ne[3] : 0),
            (int)(mask ? mask->nb[1] : 0), (int)(mask ? mask->nb[2] : 0), (int64_t)(mask ? mask->nb[3] : 0),
            key_bits, value_bits, record_bytes, token_group, record_dim);
}
