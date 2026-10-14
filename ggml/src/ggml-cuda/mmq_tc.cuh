// mmq_tc.cuh - Tensor Core GEMM helpers for k-quant weight dequantization
// Part of llamAmpere: FP16 GEMM with k-quant dequantization in registers
// Target: sm_86 (Ada Lovelace / RTX 40xx)

#ifndef GGML_CUDA_MMQ_TC_CUH
#define GGML_CUDA_MMQ_TC_CUH

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace tc {

// ============================================================================
// Tile constants
// ============================================================================

constexpr int BM  = 128;  // weight rows per CTA
constexpr int BN  = 128;  // tokens per CTA
constexpr int BK  = 64;   // K elements per pipeline stage
constexpr int NTHREADS = 256; // 8 warps
constexpr int NWARPS  = 8;
constexpr int HCH = 2;    // stages accumulated in FP16 before FP32
constexpr int NSH = 4;    // number of pipeline stages

// ============================================================================
// Weight type identifiers
// ============================================================================

constexpr int TQ4K = 12;  // Q4_K
constexpr int TQ5K = 13;  // Q5_K
constexpr int TQ6K = 14;  // Q6_K
constexpr int TQ80 = 8;   // Q8_0

// Per-type metadata: superblock size (bytes), bytes per row per stage, # chunks per BM
template <int TYPE> struct WType;

template <> struct WType<TQ4K> {
    static constexpr int BLK         = 144;  // superblock: [dmin 16 | dmax 16 | hs 1 | qs 256]
    static constexpr int STAGE_BYTES = 48;   // per row per stage: [hdr 16 | qs 32]
    static constexpr int NCH         = 3;    // chunks per BM rows
};

template <> struct WType<TQ5K> {
    static constexpr int BLK         = 176;  // superblock: [dmin 16 | dmax 16 | qh 64 | qs 256]
    static constexpr int STAGE_BYTES = 80;   // per row per stage: [hdr 16 | qh 32 | qs 32]
    static constexpr int NCH         = 5;
};

template <> struct WType<TQ6K> {
    static constexpr int BLK         = 224;  // superblock: [dmin 16 | dmax 16 | qh 32 | qs 256 | qh2 64]
    static constexpr int STAGE_BYTES = 56;   // per row per stage: [qlo 32 | qhi 16 | meta 8]
    static constexpr int NCH         = 3;
};

template <> struct WType<TQ80> {
    static constexpr int BLK         = 256;  // superblock: [units 64 | d 4]
    static constexpr int STAGE_BYTES = 68;   // per row per stage: [units 64 | d 4]
    static constexpr int NCH         = 4;
};

// ============================================================================
// Shared memory layout
// ============================================================================

struct SmemLayout {
    // Weight tile: BM rows * STAGE_BYTES per row per stage
    // Activation tile: BN tokens * BK halves per stage
    // (STAGE_BYTES defaults to Q4_K; specialized for other types via template)

    static constexpr int W_STAGE_Q4K = BM * WType<TQ4K>::STAGE_BYTES;
    static constexpr int W_STAGE_Q5K = BM * WType<TQ5K>::STAGE_BYTES;
    static constexpr int W_STAGE_Q6K = BM * WType<TQ6K>::STAGE_BYTES;
    static constexpr int W_STAGE_Q80 = BM * WType<TQ80>::STAGE_BYTES;

    static constexpr int A_STAGE = BN * BK * 2;  // sizeof(__half) = 2

    static constexpr int W_TOTAL_Q4K = W_STAGE_Q4K * NSH;
    static constexpr int W_TOTAL_Q5K = W_STAGE_Q5K * NSH;
    static constexpr int W_TOTAL_Q6K = W_STAGE_Q6K * NSH;
    static constexpr int W_TOTAL_Q80 = W_STAGE_Q80 * NSH;

    static constexpr int A_TOTAL = A_STAGE * NSH;

    static constexpr int TOTAL_Q4K = W_TOTAL_Q4K + A_TOTAL;
    static constexpr int TOTAL_Q5K = W_TOTAL_Q5K + A_TOTAL;
    static constexpr int TOTAL_Q6K = W_TOTAL_Q6K + A_TOTAL;
    static constexpr int TOTAL_Q80 = W_TOTAL_Q80 + A_TOTAL;
};

// ============================================================================
// Activation permutation (hperm)
// ============================================================================

// Permute activation indices so that 4 consecutive bytes in shared memory
// correspond to the 4 registers of an MMA A fragment.
//
// Layout of a 16-group (64 elements):
//   Original slots: 4t + {0,1,2,3}
//   Permuted slots:  2t + {0,1} and 8 + 2t + {0,1}
//
// Formula: hperm(e) = (e & ~15) | ((((e>>1)&1)<<2 | ((e>>2)&3))<<1) | (e&1)
__device__ __forceinline__ int hperm(int e) {
    return (e & ~15) | ((((e >> 1) & 1) << 2 | ((e >> 2) & 3)) << 1) | (e & 1);
}

// ============================================================================
// MMA intrinsics: m16n8k16 row.col
// ============================================================================

// Accumulate: C = C + A * B
// A: 4 uint32 registers (row-major fragment, each = 16 bits = 1 __half)
// B: 2 uint32 registers (col-major fragment, each = 16 bits = 1 __half)
// C/D: 4 float accumulators (row-major fragment)
__device__ __forceinline__ void mma_16816(float &c0, float &c1, float &c2, float &c3,
                                          unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                          unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ============================================================================
// ldmatrix: load 8x8 matrix fragments from shared memory
// ============================================================================

// Load 4 registers (m8n8.x4) = 64 bits each = 4 half values
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// Load 2 registers (m8n8.x2) = 64 bits each = 2 half values
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}

// ============================================================================
// cp.async intrinsics
// ============================================================================

__device__ __forceinline__ void cpa16(unsigned s, const void *g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(s), "l"(g) : "memory");
}

__device__ __forceinline__ void cpa8(unsigned s, const void *g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                 :: "r"(s), "l"(g) : "memory");
}

__device__ __forceinline__ void cpa4(unsigned s, const void *g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                 :: "r"(s), "l"(g) : "memory");
}

__device__ __forceinline__ void cpa_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cpa_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

// ============================================================================
// Dequantization helpers for Q4_K
// ============================================================================

// Decode one 16-element group of Q4_K to FP16.
// Input: 8 bytes of packed 4-bit values + scale half2
// Output: 8 __half2 (16 halves) in hperm order ready for MMA A fragments.
//
// Q4_K sub-block layout (per 32 values):
//   dmin[16] uint8_t  - per-element min scale (0..15 maps to actual min)
//   dmax[16] uint8_t  - per-element max scale
//   hs[1]   uint8_t   - half-scale selector
//   qs[32]  uint8_t   - quantized 4-bit values (0..15)
//
// Simplified prototype: uses a uniform scale per 32-value sub-block.
// val_fp16 ≈ (q_val - 8) * d_uniform  (where d_uniform = dmin + (dmax-dmin)/2)
__device__ __forceinline__ void dequant_q4k_16(
    const uint8_t *qs,   // 8 bytes: packed 4-bit values (low nibble of even, high nibble of odd)
    const __half2 d,     // scale half2: (d_uniform, d_uniform)
    __half2 out[8])      // 8 x half2 = 16 halves output, in hperm order
{
    const uint32_t M4 = 0x0F0F0F0Fu;

    // Unpack 4-bit values into uint32 slots (0..15 each)
    uint32_t v01  = ((uint32_t)qs[0] & M4) | (((uint32_t)qs[0] >> 4) & (M4 << 4));
    uint32_t v23  = ((uint32_t)qs[1] & M4) | (((uint32_t)qs[1] >> 4) & (M4 << 4));
    uint32_t v45  = ((uint32_t)qs[2] & M4) | (((uint32_t)qs[2] >> 4) & (M4 << 4));
    uint32_t v67  = ((uint32_t)qs[3] & M4) | (((uint32_t)qs[3] >> 4) & (M4 << 4));
    uint32_t v89  = ((uint32_t)qs[4] & M4) | (((uint32_t)qs[4] >> 4) & (M4 << 4));
    uint32_t v1011= ((uint32_t)qs[5] & M4) | (((uint32_t)qs[5] >> 4) & (M4 << 4));
    uint32_t v1213= ((uint32_t)qs[6] & M4) | (((uint32_t)qs[6] >> 4) & (M4 << 4));
    uint32_t v1415= ((uint32_t)qs[7] & M4) | (((uint32_t)qs[7] >> 4) & (M4 << 4));

    // Bias: subtract 8 (midpoint of 0..15) to get signed values
    // Then multiply by scale
    // Using half2 operations for parallelism

    // Precompute: (val - 8) * d
    // (val - 8) where val is 0..15 → range -8..+7
    // We approximate: cast each half to float, subtract 8.0, multiply by d

    const __half h8 = __float2half_rn(8.0f);
    const __half2 h8_2 = __half2half2(h8);

    out[0] = __hmul2(__hsub2(__half2half2(v01), h8_2), d);
    out[1] = __hmul2(__hsub2(__half2half2(v23), h8_2), d);
    out[2] = __hmul2(__hsub2(__half2half2(v45), h8_2), d);
    out[3] = __hmul2(__hsub2(__half2half2(v67), h8_2), d);
    out[4] = __hmul2(__hsub2(__half2half2(v89), h8_2), d);
    out[5] = __hmul2(__hsub2(__half2half2(v1011), h8_2), d);
    out[6] = __hmul2(__hsub2(__half2half2(v1213), h8_2), d);
    out[7] = __hmul2(__hsub2(__half2half2(v1415), h8_2), d);
}

// ============================================================================
// Dequantization helpers for Q8_0
// ============================================================================

// Decode one 32-element group of Q8_0 to FP16.
// Input: 32 uint8_t quantized values + 1 float scale
// Output: 16 __half2 (32 halves) in hperm order
__device__ __forceinline__ void dequant_q80_32(
    const uint8_t *qs,   // 32 bytes: quantized values (-128..127)
    const __half d,      // scale (single value)
    __half2 out[16])     // 16 x half2 = 32 halves output
{
    const __half2 hs = __half2half2(d);

    // Process 4 values at a time into half2 pairs
    for (int i = 0; i < 32; i += 4) {
        // Convert two int8 to half2
        __half h0 = __int2half_rn((int8_t)qs[i]);
        __half h1 = __int2half_rn((int8_t)qs[i + 1]);
        __half h2 = __int2half_rn((int8_t)qs[i + 2]);
        __half h3 = __int2half_rn((int8_t)qs[i + 3]);

        out[i / 2] = __hmul2(__halves2half2(h0, h1), hs);
        out[i / 2 + 2] = __hmul2(__halves2half2(h2, h3), hs);
    }
}

// ============================================================================
// Weight offset helpers
// ============================================================================

// Given a row index (0..BM-1), a superblock index sb, and a stage index p (0 or 1),
// compute the byte offset into the weight matrix for this row's stage data.
//
// For Q4_K: superblock = [dmin 16 | dmax 16 | hs 1 | qs 256]
//   Stage 0: header at offset 0, qs at offset 33
//   Stage 1: header at offset 0, qs at offset 289 (= 33 + 256)
//
// Simplified for prototype:
template <int TYPE>
__device__ __forceinline__ int weight_offset(int row, int sb, int p, int K) {
    constexpr int BLK = WType<TYPE>::BLK;
    constexpr int CHKS = WType<TYPE>::NCH;

    // Each BM rows processes CHKS superblocks
    // Base offset for this row's chunk
    int chunk_base = row / CHKS;
    int superblock_base = chunk_base * BLK + sb * BLK;

    if constexpr (TYPE == TQ4K) {
        // Stage p: qs start at 33 + 256*p within superblock
        return superblock_base + (p == 0 ? 33 : 289);
    } else if constexpr (TYPE == TQ5K) {
        // Stage p: qs start at 81 + 256*p within superblock (after dmin16+dmax16+qh64)
        return superblock_base + (p == 0 ? 81 : 337);
    } else if constexpr (TYPE == TQ6K) {
        // Stage p: qs start at 33 + 256*p (qlo) + meta offset
        return superblock_base + (p == 0 ? 33 : 289);
    } else { // TQ80
        // Stage p: units at 0 + 64*p, d at 64 + 4*p
        return superblock_base + (p == 0 ? 0 : 64);
    }
}

// ============================================================================
// Activation shared memory offset
// ============================================================================

// Given a token index (0..BN-1) and a stage index p (0..NSH-1),
// compute the byte offset in the activation shared memory buffer.
__device__ __forceinline__ int act_offset(int token, int p) {
    return BN * BK * 2 * p + token * BK * 2;
}

// ============================================================================
// Warp-level helpers
// ============================================================================

// Each warp handles BM/NWARPS = 16 rows and BN = 128 tokens.
// Thread within warp: threadIdx.x % 32
// Row within warp: threadIdx.x / 32 (but we have 16 rows per warp with 32 threads)
// So: each pair of threads handles one row's worth of data

__device__ __forceinline__ int warp_row_base() {
    return (threadIdx.x / 32) * (BM / NWARPS);
}

__device__ __forceinline__ int warp_token_base() {
    return 0; // Each warp handles all BN tokens
}

// ============================================================================
// Pipeline stage helpers
// ============================================================================

// Return the pipeline stage index for this iteration
__device__ __forceinline__ int stage_idx(int iter, int p) {
    return (iter + p) % NSH;
}

} // namespace tc

#endif // GGML_CUDA_MMQ_TC_CUH
