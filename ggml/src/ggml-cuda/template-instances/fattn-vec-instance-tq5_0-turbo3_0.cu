// TurboQuant CUDA flash attention vec kernel instantiation
// tq5_0/turbo3_0 — block size 128, D=128/256 only

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0);
