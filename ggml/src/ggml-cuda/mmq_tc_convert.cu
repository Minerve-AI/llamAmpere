// mmq_tc_convert.cu
//
// Pre-kernel that converts the FP32 activation into FP16 with the tensor-core
// "hperm" permutation, producing the operand layout expected by the MMQ
// tensor-core kernel.
//
//   input  : activation FP32,  [K, ntok] col-major (K is the fast dimension)
//   output : activation FP16,  [K, ntok] col-major, K-dim in hperm order
//
// hperm is a permutation of each 16-element group.  For an element index e
// (0..15) inside a 16-group, the element is stored at slot hperm(e):
//   4t + {0,1} -> 2t + {0,1}
//   4t + {2,3} -> 8 + 2t + {0,1}
//
// The kernel is a simple per-element scatter: for every source index k the
// value is written to dst at k_group + hperm(k_in_group).  hperm is a bijection
// on each 16-group, so every destination slot is written exactly once.

#include "common.cuh"

#include <cuda_fp16.h>

// Element e of a 16-group sits at slot hperm(e):
// 4t + {0,1} -> 2t + {0,1}, 4t + {2,3} -> 8 + 2t + {0,1}
__device__ __forceinline__ int hperm(int e) {
    return (e & ~15) | ((((e>>1)&1)<<2 | ((e>>2)&3))<<1) | (e&1);
}

__global__ void convert_act_to_hperm(
    const float * __restrict__ src,  // [K, ntok] FP32 col-major
    __half * __restrict__ dst,       // [K, ntok] FP16 hperm col-major
    int K, int ntok)
{
    // Each thread handles one element.
    // Total elements: K * ntok
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)K * ntok;
    if (idx >= total) return;

    const int tok = idx / K;
    const int k   = idx % K;

    // Apply hperm: the element at position k goes to position hperm(k)
    // within its 16-group, i.e. dst[k_group + hperm(k_in_group)] = src[idx].
    const int k_group    = k & ~15;  // base of the 16-group
    const int k_in_group = k & 15;
    const int k_out      = k_group + hperm(k_in_group);

    dst[(size_t)tok * K + k_out] = __float2half_rn(src[idx]);
}

void ggml_cuda_convert_act_hperm(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor * src,
                                  ggml_tensor * dst)
{
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src->ne[0] % 16 == 0); // K must be a multiple of 16 for hperm

    const int64_t K    = src->ne[0];
    const int64_t ntok = src->ne[1];

    const float  * src_d = (const float  *) src->data;
    __half       * dst_d = (__half       *) dst->data;

    const int64_t total = K * ntok;
    const int threads   = 256;
    const int64_t blocks = (total + threads - 1) / threads;

    convert_act_to_hperm<<<blocks, threads, 0, ctx.stream()>>>(src_d, dst_d, (int)K, (int)ntok);
}
