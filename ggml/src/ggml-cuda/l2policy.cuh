#pragma once
#include <cuda_runtime.h>

// L2 cache policy helpers for sm_86+ (Ampere and later).
//
// Strategy:
//   - Weights (streamed once per token) → L1::no_allocate + L2::evict_first
//     They pass through L2 but are evicted first, protecting KV cache and
//     recurrent state from being displaced by the weight stream.
//   - KV cache / GDN state (re-read across layers) → prefetch with L2::evict_last
//     They stay in L2 longer, reducing DRAM round-trips.
//
// On sm_86 (RTX 3090): 82 SMs stream ~16.6 GB of Q4_K weights per decode token
// at 936 GB/s. Without these hints, the weight stream evicts KV and GDN state
// from the 60 MB L2, causing extra DRAM traffic on subsequent layers.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860

// Generate a 64-bit L2 cache policy: "evict first" (lowest priority).
// Call once per kernel invocation and reuse for all weight loads.
static __device__ __forceinline__ unsigned long long l2_policy_evict_first() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}

// Load 4 bytes (int32) with L1 bypass + L2 evict-first policy.
// For weight data (Q4_K qs bytes, loaded as int).
static __device__ __forceinline__ int ld_w4(const void * p, unsigned long long pol) {
    int r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.s32 %0, [%1], %2;"
        : "=r"(r) : "l"(p), "l"(pol));
    return r;
}

// Load 8 bytes (int2) with L1 bypass + L2 evict-first policy.
static __device__ __forceinline__ int2 ld_w8(const void * p, unsigned long long pol) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.s32 {%0,%1}, [%2], %3;"
        : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(pol));
    return r;
}

// Load 16 bytes (int4) with L1 bypass + L2 evict-first policy.
static __device__ __forceinline__ int4 ld_w16(const void * p, unsigned long long pol) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(p), "l"(pol));
    return r;
}

// Load 2 bytes (uint16) with L1 bypass only (too small for cache_hint,
// but L1 bypass still saves L1 set for streaming data).
static __device__ __forceinline__ unsigned short ld_w16b(const void * p) {
    unsigned short r;
    asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
    return r;
}

// Prefetch a cache line into L2 with evict-last policy.
// Use for KV cache and GDN recurrent state before they are needed.
static __device__ __forceinline__ void prefetch_l2_evict_last(const void * p) {
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p));
}

#else

// Fallback for architectures without createpolicy (pre-Ampere): plain loads.
static __device__ __forceinline__ unsigned long long l2_policy_evict_first() { return 0; }
static __device__ __forceinline__ int ld_w4(const void * p, unsigned long long) { return *(const int *)p; }
static __device__ __forceinline__ int2 ld_w8(const void * p, unsigned long long) { return *(const int2 *)p; }
static __device__ __forceinline__ int4 ld_w16(const void * p, unsigned long long) { return *(const int4 *)p; }
static __device__ __forceinline__ unsigned short ld_w16b(const void * p) { return *(const unsigned short *)p; }
static __device__ __forceinline__ void prefetch_l2_evict_last(const void *) {}

#endif
