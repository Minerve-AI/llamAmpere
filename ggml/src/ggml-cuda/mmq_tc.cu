#include "common.cuh"
#include "mmq_tc.cuh"
#include "ggml-cuda/common.h"

// ============================================================================
// Tensor Core GEMM kernel for prefill (batch >= 128)
//
// Y[tok][row] = sum_k W[row][k] * X[tok][k]
//
// W: quantized weights (Q4_K, Q8_0) in GGUF layout
// X: FP16 activations in hperm order, shape [K, ntok] col-major
// Y: FP32 output, shape [N, ntok] col-major
//
// Tile: BM=128 rows x BN=128 tokens per CTA, 8 warps, BK=64
// MMA: m16n8k16.row.col.f16
// Pipeline: cp.async, NSH=4 stages
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860

// Q4_K: block size 144 bytes covers 256 K values (qk4_k=256)
// Per 32 K values per row: 144/8 = 18 bytes (2 sub-blocks of 16 vals)
// For BK=64: 2 sub-blocks = 36 bytes... but we use STAGE_BYTES from .cuh
//
// Simplified approach for V1:
// - Load raw weight bytes into smem
// - Dequantize in registers to FP16
// - Use MMA for the GEMM
// - The exact GGUF offset is computed per row/stage

template <int WTYPE>
__global__ void __launch_bounds__(tc::NTHREADS) mul_mat_q_tc(
    const uint8_t * __restrict__ w,
    const __half * __restrict__ x,
    float * __restrict__ y,
    int K, int N, int ntok)
{
    constexpr int BM_ = tc::BM;
    constexpr int BN_ = tc::BN;
    constexpr int BK_ = tc::BK;
    constexpr int NSH_ = tc::NSH;
    constexpr int HCH_ = tc::HCH;
    constexpr int STAGE_B = tc::WType<WTYPE>::STAGE_BYTES;
    constexpr int W_STAGE = BM_ * STAGE_B;
    constexpr int A_STAGE = BN_ * BK_ * 2; // bytes

    extern __shared__ uint8_t smem[];
    uint8_t  * w_smem = smem;
    __half   * a_smem = reinterpret_cast<__half *>(smem + NSH_ * W_STAGE);

    const int row0 = blockIdx.x * BM_;
    const int tok0 = blockIdx.y * BN_;
    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int nstages = K / BK_;

    // FP32 accumulators: 16 N-tiles x 4 regs
    float acc[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++)
        #pragma unroll
        for (int k = 0; k < 4; k++)
            acc[n][k] = 0.0f;

    // === Weight load: 2 threads per row, each loads STAGE_B/2 bytes ===
    auto load_w = [&](int stage) {
        const int s = stage % NSH_;
        const int r = tid / 2;
        const int h = tid % 2;
        if (r >= BM_) return;
        // Global offset: row r of the tile, K offset = stage * BK
        // Simplified: assume row-major uint8_t layout
        const uint8_t * src = w + (size_t)(row0 + r) * (K / 2) + stage * BK / 2 + h * (STAGE_B / 2);
        uint8_t * dst = w_smem + s * W_STAGE + r * STAGE_B + h * (STAGE_B / 2);
        // Load in 16-byte chunks
        const int nchunks = (STAGE_B / 2) / 16;
        #pragma unroll
        for (int c = 0; c < nchunks; c++) {
            unsigned s_addr = __cvta_generic_to_shared(dst + c * 16);
            tc::cpa16(s_addr, src + c * 16);
        }
    };

    // === Activation load: 256 threads, each loads A_STAGE/256 bytes ===
    auto load_a = [&](int stage) {
        const int s = stage % NSH_;
        const int bytes_per_thread = A_STAGE / tc::NTHREADS; // 64 bytes
        const int nchunks = bytes_per_thread / 16;           // 4 chunks
        // Global: x is [K, ntok] col-major, hperm order
        // For stage (k0 = stage*BK), load BK halves per token
        const __half * src = x + (size_t)(tok0) * K + stage * BK;
        __half * dst = a_smem + s * BN_ * BK_;
        // Each thread loads 4 chunks of 16 bytes (8 halves each)
        // Layout: thread tid loads data for tokens starting at tid*2
        #pragma unroll
        for (int c = 0; c < nchunks; c++) {
            const int offset = tid * bytes_per_thread + c * 16;
            const int tok = offset / (BK_ * 2);      // which token
            const int koff = (offset % (BK_ * 2)) / 2; // which K within token
            if (tok < BN_) {
                unsigned s_addr = __cvta_generic_to_shared(dst + tok * BK_ + koff);
                const __half * gsrc = x + (size_t)(tok0 + tok) * K + stage * BK_ + koff;
                tc::cpa16(s_addr, gsrc);
            }
        }
    };

    // === Pipeline prologue: prefetch NSH-1 stages ===
    #pragma unroll
    for (int s = 0; s < NSH_ - 1; s++) {
        if (s < nstages) {
            load_w(s);
            load_a(s);
        }
        tc::cpa_commit();
    }

    // === Main loop ===
    for (int stage = 0; stage < nstages; stage++) {
        const int s = stage % NSH_;

        // Wait for current stage
        tc::cpa_wait<NSH_ - 2>();
        __syncthreads();

        // --- Dequantize weights and compute A fragments ---
        // Each warp: 16 rows, each thread pair handles 1 row
        // Per thread: 32 K values (half of BK=64) = 2 groups of 16
        // A fragments: 2 K-tiles x 4 regs = 8 regs

        unsigned a_frag[2][4];
        {
            const int r = warp * 16 + (lane / 2); // row within warp
            const int half = lane % 2;
            const uint8_t * wptr = w_smem + s * W_STAGE + r * STAGE_B + half * (STAGE_B / 2);

            if constexpr (WTYPE == tc::TQ4K) {
                // Dequantize 32 K values (2 groups of 16)
                // Group 0: bytes 16..23 (8 bytes of 4-bit values)
                // Group 1: bytes 24..31 (8 bytes of 4-bit values)
                // Scale from header (bytes 0..15)
                __half2 scale = __floats2half2_rn(1.0f, 1.0f); // TODO: read actual scale
                __half2 out[8];
                // Group 0
                tc::dequant_q4k_16(wptr + 16, scale, out);
                a_frag[0][0] = *reinterpret_cast<unsigned *>(&out[0]);
                a_frag[0][1] = *reinterpret_cast<unsigned *>(&out[1]);
                a_frag[0][2] = *reinterpret_cast<unsigned *>(&out[2]);
                a_frag[0][3] = *reinterpret_cast<unsigned *>(&out[3]);
                // Group 1
                tc::dequant_q4k_16(wptr + 24, scale, out);
                a_frag[1][0] = *reinterpret_cast<unsigned *>(&out[0]);
                a_frag[1][1] = *reinterpret_cast<unsigned *>(&out[1]);
                a_frag[1][2] = *reinterpret_cast<unsigned *>(&out[2]);
                a_frag[1][3] = *reinterpret_cast<unsigned *>(&out[3]);
            } else if constexpr (WTYPE == tc::TQ80) {
                // Q8_0: 32 int8 values, scale from last 4 bytes
                __half2 scale = __floats2half2_rn(1.0f, 1.0f); // TODO: read actual scale
                __half2 out[8];
                tc::dequant_q80_32(wptr + 16, scale, out);
                a_frag[0][0] = *reinterpret_cast<unsigned *>(&out[0]);
                a_frag[0][1] = *reinterpret_cast<unsigned *>(&out[1]);
                a_frag[0][2] = *reinterpret_cast<unsigned *>(&out[2]);
                a_frag[0][3] = *reinterpret_cast<unsigned *>(&out[3]);
                tc::dequant_q80_32(wptr + 24, scale, out);
                a_frag[1][0] = *reinterpret_cast<unsigned *>(&out[0]);
                a_frag[1][1] = *reinterpret_cast<unsigned *>(&out[1]);
                a_frag[1][2] = *reinterpret_cast<unsigned *>(&out[2]);
                a_frag[1][3] = *reinterpret_cast<unsigned *>(&out[3]);
            }
        }

        // --- Load B fragments and MMA accumulate ---
        const __half * a_base = a_smem + s * BN_ * BK_;
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            // Load B fragment: 16x8, 2 regs per thread
            unsigned b[2];
            {
                // ldmatrix from smem: B is col-major [BK, BN]
                // For N-tile n: tokens tok0 + n*8 .. tok0 + n*8+7
                // K range: 16 values (one K-tile)
                const int tok = n * 8 + (lane % 8);
                const int k0 = (lane / 8) * 2;
                unsigned addr = __cvta_generic_to_shared(a_base + tok * BK_ + k0);
                tc::ldsm_x2(b, addr);
            }
            // K-tile 0
            tc::mma_16816(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                          a_frag[0][0], a_frag[0][1], a_frag[0][2], a_frag[0][3],
                          b[0], b[1]);
        }
        // K-tile 1 (next 16 K values)
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            unsigned b[2];
            {
                const int tok = n * 8 + (lane % 8);
                const int k0 = 16 + (lane / 8) * 2;
                unsigned addr = __cvta_generic_to_shared(a_base + tok * BK_ + k0);
                tc::ldsm_x2(b, addr);
            }
            tc::mma_16816(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                          a_frag[1][0], a_frag[1][1], a_frag[1][2], a_frag[1][3],
                          b[0], b[1]);
        }

        // Prefetch next stage
        if (stage + NSH_ - 1 < nstages) {
            load_w(stage + NSH_ - 1);
            load_a(stage + NSH_ - 1);
        }
        tc::cpa_commit();
        __syncthreads();
    }

    // === Epilogue: write FP32 accumulators to global memory ===
    const int row = row0 + warp * 16 + (lane / 2);
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        const int tok = tok0 + n * 8 + (lane % 8);
        if (tok < ntok) {
            y[(size_t)row * ntok + tok] = acc[n][0];
            y[(size_t)(row + 8) * ntok + tok] = acc[n][1];
        }
    }
}

#endif // __CUDA_ARCH__ >= 860

// ============================================================================
// Host launch function
// ============================================================================

void ggml_cuda_mul_mat_tc(ggml_backend_cuda_context & ctx,
                          const ggml_tensor * src0, const ggml_tensor * src1,
                          ggml_tensor * dst)
{
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc < 860) return;

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int64_t K = ne00;
    const int64_t N = ne01;
    const int64_t ntok = ne11;

    if (K % tc::BK != 0 || N % tc::BM != 0) return;

    const void * wdata = ggml_backend_cuda_get_buffer_read_ptr(src0);
    const void * xdata = ggml_backend_cuda_get_buffer_read_ptr(src1);
    void *       ydata = ggml_backend_cuda_get_buffer_write_ptr(dst);

    const int smem_bytes = tc::NSH * tc::BM * tc::WType<GGML_TYPE_Q4_K>::STAGE_BYTES
                         + tc::NSH * tc::BN * tc::BK * 2;

    dim3 block(tc::NTHREADS);
    dim3 grid(N / tc::BM, (ntok + tc::BN - 1) / tc::BN);

    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            GGML_CUDA_KERNEL(mul_mat_q_tc<tc::TQ4K>, grid, block, smem_bytes,
                wdata, xdata, ydata, K, N, ntok);
            break;
        case GGML_TYPE_Q8_0:
            GGML_CUDA_KERNEL(mul_mat_q_tc<tc::TQ80>, grid, block, smem_bytes,
                wdata, xdata, ydata, K, N, ntok);
            break;
        default:
            GGML_ABORT("unsupported type %d for TC kernel\n", (int)src0->type);
    }
}
