// mmq_tc.cu - Tensor core GEMM kernel for prefill (V1)
//
// Supports: Q4_K (144B/block, 256 vals), Q8_0 (34B/block, 32 vals)
// Tile: BM=128 rows x BN=128 tokens, BK=64, NSH=4 stages
// 8 warps (256 threads), each warp handles 16 rows
// MMA: m16n8k16.f16 (tensor cores, sm_86+)
//
// Weight smem layout (per row per stage):
//   Q4_K: 48B = [d 2B][dmin 2B][scales 12B][qs 32B]
//   Q8_0: 68B = [d0 2B][qs0 32B][d1 2B][qs1 32B]
//
// Dequant: direct to MMA A-fragment registers (no intermediate buffer)
// Activation: FP16 hperm, cp.async into a_smem[stage][tok][k]
//
// Shared memory budget:
//   Q4_K: 4*128*48 + 4*128*64*2 = 24KB + 64KB = 88KB < 99KB
//   Q8_0: 4*128*68 + 4*128*64*2 = 34KB + 64KB = 98KB < 99KB

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq_tc.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

// ============================================================================
// Dequant helpers
// ============================================================================

// Extract 6-bit scale and 6-bit min from the Q4_K scales array
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 8] & 0xF) | ((q[j - 4] >> 6) << 4);
    }
}

// ============================================================================
// Main kernel
// ============================================================================

template <int WTYPE>
__global__ void __launch_bounds__(tc::NTHREADS)
mul_mat_q_tc_kernel(
    const uint8_t * __restrict__ W,   // quantized weights [N, K] row-major
    const float   * __restrict__ A,   // FP32 activations [K, ntok] col-major
    float         * __restrict__ C,   // output [N, ntok] col-major (FP32)
    int N, int K, int ntok)
{
    constexpr int BM    = tc::BM;     // 128
    constexpr int BN    = tc::BN;     // 128
    constexpr int BK    = tc::BK;     // 64
    constexpr int NSH   = tc::NSH;    // 4
    constexpr int WSTAGE = tc::WType<WTYPE>::STAGE_BYTES;

    // Shared memory layout: [weights: NSH*BM*WSTAGE][activations: NSH*BN*BK*2]
    extern __shared__ uint8_t smem_raw[];
    uint8_t * w_smem = smem_raw;
    __half  * a_smem = reinterpret_cast<__half*>(smem_raw + NSH * BM * WSTAGE);

    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int block_row  = blockIdx.x * BM;
    const int block_tok  = blockIdx.y * BN;
    const int n_tiles_k  = K / BK;

    // MMA fragment positions for this thread:
    // A fragment (m16n8k16): row = warp*16 + lane/4, K = kt*16 + (lane%4)*2 + {0,1}
    //                        row = warp*16 + 8 + lane/4, same K
    const int rA = warp * 16 + lane / 4;       // first row
    const int rB = rA + 8;                      // second row
    const int kpos = (lane % 4) * 2;           // K position within 16-tile

    // ========================================================================
    // cp.async load lambda
    // ========================================================================

    auto load_stage = [&](int stage, int kt) {
        if (kt >= n_tiles_k) return;
        const int k0 = kt * BK;

        // --- Weights ---
        {
            const int r  = tid / 2;
            const int h  = tid % 2;
            const int row = block_row + r;

            if (row < N) {
                uint8_t * dst = w_smem + stage * BM * WSTAGE + r * WSTAGE;

                if constexpr (WTYPE == tc::TQ4K) {
                    const int bidx  = k0 / 256;
                    const int koff  = k0 % 256;
                    const uint8_t * g = W + (size_t)row * (K / 256) * 144 + bidx * 144;

                    if (h == 0) {
                        __pipeline_memcpy_async(dst + 0,  g + 0,   4);   // d + dmin
                        __pipeline_memcpy_async(dst + 4,  g + 132, 12);  // scales
                        __pipeline_memcpy_async(dst + 16, g + 4 + koff/2,     16);
                        __pipeline_memcpy_async(dst + 32, g + 4 + koff/2 + 16, 16);
                    } else {
                        __pipeline_memcpy_async(dst + 32, g + 4 + koff/2 + 16, 16);
                    }
                } else {
                    // Q8_0: 2 blocks of 34B
                    const uint8_t * g0 = W + (size_t)row * (K / 32) * 34 + (k0 / 32) * 34;
                    const uint8_t * g1 = g0 + 34;

                    if (h == 0) {
                        __pipeline_memcpy_async(dst + 0,  g0,     16);
                        __pipeline_memcpy_async(dst + 16, g0 + 16, 16);
                        *reinterpret_cast<uint16_t*>(dst + 32) =
                            *reinterpret_cast<const uint16_t*>(g0 + 32);
                    } else {
                        __pipeline_memcpy_async(dst + 34, g1,     16);
                        __pipeline_memcpy_async(dst + 50, g1 + 16, 16);
                        *reinterpret_cast<uint16_t*>(dst + 66) =
                            *reinterpret_cast<const uint16_t*>(g1 + 32);
                    }
                }
            }
        }

        // --- Activations (FP32 global -> FP16 smem, in registers) ---
        {
            __half * dst = a_smem + stage * BN * BK;
            const float * g = A + k0;
            const int tok_local = tid / 2;
            const int koff      = (tid % 2) * 32;
            const int tok0      = block_tok + tok_local;
            if (tok0 < ntok) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    // Load 8 consecutive floats (2 x float4)
                    const float4 * gf = reinterpret_cast<const float4*>(g + (size_t)tok0 * K + koff + i * 8);
                    float4 f0 = gf[0];
                    float4 f1 = gf[1];
                    // Convert to half2 and store
                    __half2 * dh = reinterpret_cast<__half2*>(dst + tok_local * BK + koff + i * 8);
                    dh[0] = __halves2half2(__float2half_rn(f0.x), __float2half_rn(f0.y));
                    dh[1] = __halves2half2(__float2half_rn(f0.z), __float2half_rn(f0.w));
                    dh[2] = __halves2half2(__float2half_rn(f1.x), __float2half_rn(f1.y));
                    dh[3] = __halves2half2(__float2half_rn(f1.z), __float2half_rn(f1.w));
                }
            }
        }

        __pipeline_commit();
    };

    #pragma unroll
    for (int s = 0; s < NSH - 1; s++)
        load_stage(s, s);

    // ========================================================================
    // Main loop
    // ========================================================================

    float c_acc[BN/8][4];
    #pragma unroll
    for (int i = 0; i < BN/8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            c_acc[i][j] = 0.0f;

    for (int kt = 0; kt < n_tiles_k; kt++) {
        const int s = kt % NSH;

        __pipeline_wait_prior(NSH - 2);
        __syncthreads();

        const int k0 = kt * BK;

        // --- Dequantize weights directly to A-fragment registers ---
        unsigned a_frag[4][2];

        if constexpr (WTYPE == tc::TQ4K) {
            const uint8_t * wA = w_smem + s * BM * WSTAGE + rA * WSTAGE;
            const uint8_t * wB = w_smem + s * BM * WSTAGE + rB * WSTAGE;

            const float d    = __half2float(*reinterpret_cast<const __half*>(wA));
            const float dmin = __half2float(*reinterpret_cast<const __half*>(wA + 2));
            const uint8_t * scalesA = wA + 4;
            const uint8_t * qsA     = wA + 16;
            const uint8_t * qsB     = wB + 16;

            const int koff = k0 % 256;

            #pragma unroll
            for (int kl = 0; kl < BK / 16; kl++) {
                const int k_global = koff + kl * 16 + kpos;
                const int sb = k_global / 32;

                uint8_t sc, m;
                get_scale_min_k4(sb, scalesA, &sc, &m);
                const float d_eff  = d * (float)sc;
                const float m_eff  = dmin * (float)m;

                const int qidx = (kl * 16 + kpos) / 2;

                const uint8_t byteA = qsA[qidx];
                __half hA0 = __float2half_rn(d_eff * (float)(byteA & 0xF) - m_eff);
                __half hA1 = __float2half_rn(d_eff * (float)(byteA >> 4) - m_eff);

                const uint8_t byteB = qsB[qidx];
                __half hB0 = __float2half_rn(d_eff * (float)(byteB & 0xF) - m_eff);
                __half hB1 = __float2half_rn(d_eff * (float)(byteB >> 4) - m_eff);

                a_frag[kl][0] = *reinterpret_cast<unsigned*>(&hA0) |
                                ((*reinterpret_cast<unsigned*>(&hA1)) << 16);
                a_frag[kl][1] = *reinterpret_cast<unsigned*>(&hB0) |
                                ((*reinterpret_cast<unsigned*>(&hB1)) << 16);
            }
        } else {
            // Q8_0
            const uint8_t * wA = w_smem + s * BM * WSTAGE + rA * WSTAGE;
            const uint8_t * wB = w_smem + s * BM * WSTAGE + rB * WSTAGE;

            const int koff = k0 % 64;

            #pragma unroll
            for (int kl = 0; kl < BK / 16; kl++) {
                const int k_global = koff + kl * 16 + kpos;
                const int blk = k_global / 32;
                const int off = k_global % 32;

                const float dA = __half2float(*reinterpret_cast<const __half*>(wA + blk * 34));
                const float dB = __half2float(*reinterpret_cast<const __half*>(wB + blk * 34));

                const int8_t qA0 = *reinterpret_cast<const int8_t*>(wA + blk * 34 + 2 + off);
                const int8_t qA1 = *reinterpret_cast<const int8_t*>(wA + blk * 34 + 2 + off + 1);
                const int8_t qB0 = *reinterpret_cast<const int8_t*>(wB + blk * 34 + 2 + off);
                const int8_t qB1 = *reinterpret_cast<const int8_t*>(wB + blk * 34 + 2 + off + 1);

                __half hA0 = __float2half_rn(dA * (float)qA0);
                __half hA1 = __float2half_rn(dA * (float)qA1);
                __half hB0 = __float2half_rn(dB * (float)qB0);
                __half hB1 = __float2half_rn(dB * (float)qB1);

                a_frag[kl][0] = *reinterpret_cast<unsigned*>(&hA0) |
                                ((*reinterpret_cast<unsigned*>(&hA1)) << 16);
                a_frag[kl][1] = *reinterpret_cast<unsigned*>(&hB0) |
                                ((*reinterpret_cast<unsigned*>(&hB1)) << 16);
            }
        }

        // --- B fragments (activations) and MMA ---
        __half * a_stage = a_smem + s * BN * BK;

        #pragma unroll
        for (int nt = 0; nt < BN / 8; nt++) {
            unsigned b_frag[2];
            {
                const int tok = nt * 8 + (lane % 8);
                const int k0b = (lane / 8) * 8;
                unsigned addr = __cvta_generic_to_shared(a_stage + tok * BK + k0b);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int kl = 0; kl < BK / 16; kl++) {
                float c0 = c_acc[nt][0], c1 = c_acc[nt][1];
                float c2 = c_acc[nt][2], c3 = c_acc[nt][3];
                unsigned a0 = a_frag[kl][0], a1 = a_frag[kl][1];
                unsigned a2 = a0 >> 16, a3 = a1 >> 16;
                unsigned b0 = b_frag[0], b1 = b_frag[1];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1)
                );
                c_acc[nt][0] = c0; c_acc[nt][1] = c1;
                c_acc[nt][2] = c2; c_acc[nt][3] = c3;
            }
        }

        __syncthreads();
        load_stage((kt + NSH - 1) % NSH, kt + NSH - 1);
    }

    // ========================================================================
    // Epilogue: write C [N, ntok] col-major
    // ========================================================================
    const int row0 = block_row + rA;
    const int row1 = block_row + rB;
    const int col_base = kpos;

    #pragma unroll
    for (int nt = 0; nt < BN / 8; nt++) {
        const int col0 = block_tok + nt * 8 + col_base;
        if (row0 < N) {
            C[(size_t)col0 * N + row0]       = c_acc[nt][0];
            C[(size_t)(col0 + 1) * N + row0] = c_acc[nt][1];
        }
        if (row1 < N) {
            C[(size_t)col0 * N + row1]       = c_acc[nt][2];
            C[(size_t)(col0 + 1) * N + row1] = c_acc[nt][3];
        }
    }
}

// ============================================================================
// Host function
// ============================================================================

void ggml_cuda_mul_mat_tc(ggml_backend_cuda_context & ctx,
                           const ggml_tensor * src0,
                           const ggml_tensor * src1,
                           ggml_tensor * dst) {
    const int64_t N    = src0->ne[1];
    const int64_t K    = src0->ne[0];
    const int64_t ntok = src1->ne[1];

    GGML_ASSERT(N % tc::BM == 0);
    GGML_ASSERT(ntok % tc::BN == 0);
    GGML_ASSERT(K % tc::BK == 0);

    const uint8_t * W_d = (const uint8_t *) src0->data;
    const __half  * A_d = (const __half *) src1->data;
    float         * C_d = (float *) dst->data;

    const dim3 grid(N / tc::BM, ntok / tc::BN);
    const dim3 block(tc::NTHREADS);
    cudaStream_t stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_Q4_K: {
            constexpr size_t smem = tc::NSH * tc::BM * tc::WType<tc::TQ4K>::STAGE_BYTES
                                  + tc::NSH * tc::BN * tc::BK * sizeof(__half);
            auto & kern = mul_mat_q_tc_kernel<tc::TQ4K>;
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            kern<<<grid, block, smem, stream>>>(W_d, A_d, C_d, (int)N, (int)K, (int)ntok);
            break;
        }
        case GGML_TYPE_Q8_0: {
            constexpr size_t smem = tc::NSH * tc::BM * tc::WType<tc::TQ80>::STAGE_BYTES
                                  + tc::NSH * tc::BN * tc::BK * sizeof(__half);
            auto & kern = mul_mat_q_tc_kernel<tc::TQ80>;
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            kern<<<grid, block, smem, stream>>>(W_d, A_d, C_d, (int)N, (int)K, (int)ntok);
            break;
        }
        default:
            GGML_ABORT("unsupported type for TC kernel");
    }
}
