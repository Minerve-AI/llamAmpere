#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// ---------------------------------------------------------------------------
// KVarN flash attention dispatch helpers (host-side)
//
// The KVarN mode is signaled via op_params[0] == KVARN_MAGIC.
// KVarN FA kernel not yet implemented - falls through to standard FA.
// ---------------------------------------------------------------------------

// KVarN magic constant: op_params[0] == 0x4B564152 ("KVAR") signals KVarN mode
static constexpr uint32_t KVARN_MAGIC = 0x4B564152u;

// Check if the flash attention op is using KVarN format
static inline bool ggml_cuda_fattn_is_kvarn(const ggml_tensor * dst) {
    const uint32_t magic = (uint32_t)dst->op_params[0];
    return magic == KVARN_MAGIC;
}

// Get KVarN parameters from the flash attention op
// op_params[1] = key_bits
// op_params[2] = value_bits
// op_params[3] = record_bytes (total size of one KVarN record)
static inline void ggml_cuda_fattn_get_kvarn_params(
    const ggml_tensor * dst,
    int & key_bits, int & value_bits, size_t & record_bytes) {
    key_bits     = (int)dst->op_params[1];
    value_bits   = (int)dst->op_params[2];
    record_bytes = (size_t)dst->op_params[3];
}
