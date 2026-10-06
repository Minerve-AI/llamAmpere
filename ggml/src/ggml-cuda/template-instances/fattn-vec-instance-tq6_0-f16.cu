// TurboQuant CUDA flash attention vec kernel instantiation
// tq6_0/f16 — block size 128, D=128/256 only

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ6_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ6_0, GGML_TYPE_F16);
