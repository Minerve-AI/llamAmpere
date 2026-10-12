#pragma once

// ============================================================================
// L2 cache policy helpers for sm_86+ (Ampere)
//
// Strategy: mark streaming weight data with .cs (cache streaming) so it gets
// low-priority allocation in both L1 and L2. The GDN recurrent state is
// prefetched with evict_last so it gets high-priority L2 allocation.
//
// .cs is a single-qualifier instruction with zero overhead vs a regular load —
// no createpolicy, no cache_hint operand. This is critical for the MMVQ inner
// loop where loads are only 2-4 bytes and any per-load overhead would dominate.
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860

// --- Streaming loads (low priority in L1 and L2) ---
//
// Use for weight data that is streamed once and should not evict
// KV cache or GDN recurrent state from L2.

static __device__ __forceinline__ int ld_cs_4(const void * p) {
    int r;
    asm("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

static __device__ __forceinline__ unsigned int ld_cs_8(const void * p) {
    unsigned int r;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

static __device__ __forceinline__ unsigned short ld_cs_16b(const void * p) {
    unsigned short r;
    asm("ld.global.cs.u16 %0, [%1];" : "=h"(r) : "l"(p));
    return r;
}

// --- Prefetch with evict_last (high priority in L2) ---
//
// Use to bring GDN recurrent state into L2 with high retention priority.
// The state is reused every token, so keeping it in L2 avoids 256B-2KB
// DRAM round trips per GDN layer per token.

static __device__ __forceinline__ void prefetch_l2_evict_last(const void * p, int bytes = 128) {
    asm volatile("prefetch.global.L2::evict_last [%0], %1;" :: "l"(p), "n"(bytes));
}

#else  // sm < 860: no-ops

static __device__ __forceinline__ int ld_cs_4(const void * p) { return *(const int *)p; }
static __device__ __forceinline__ unsigned int ld_cs_8(const void * p) { return *(const unsigned int *)p; }
static __device__ __forceinline__ unsigned short ld_cs_16b(const void * p) { return *(const unsigned short *)p; }
static __device__ __forceinline__ void prefetch_l2_evict_last(const void *, int) {}

#endif
