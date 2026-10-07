#pragma once

#include "llama.h"
#include "llama-arch.h"

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t KVAR_N_GROUP = 128;

// Stage indices carry both the logical record cell and, when present, the
// host-selected physical F16 slot. A zero high word retains the legacy
// stateless mapping for non-unified callers; explicit slots are one-based in
// the packed representation so slot zero remains distinguishable.
constexpr int64_t llama_kvarn_encode_store_cell(uint32_t cell, uint32_t stage_slot) {
    return int64_t((uint64_t(stage_slot) + 1u) << 32u | uint64_t(cell));
}

constexpr int64_t llama_kvarn_encode_stage_cell(uint32_t cell) {
    return -int64_t(cell) - 2;
}

constexpr int64_t llama_kvarn_encode_stage_cell(uint32_t cell, uint32_t stage_slot) {
    return -llama_kvarn_encode_store_cell(cell, stage_slot) - 2;
}

constexpr uint64_t llama_kvarn_index_payload(int64_t index) {
    return index < -1 ? uint64_t(-(index + 2)) : uint64_t(index);
}

constexpr uint32_t llama_kvarn_decode_cell(int64_t index) {
    return uint32_t(llama_kvarn_index_payload(index));
}

constexpr int32_t llama_kvarn_decode_stage_slot(int64_t index) {
    const uint32_t encoded = uint32_t(llama_kvarn_index_payload(index) >> 32u);
    return encoded == 0 ? -1 : int32_t(encoded - 1u);
}

// SWA indices retain the absolute position in the low word and identify the
// independent KV stream in the high word.
constexpr int64_t llama_kvarn_encode_swa_position(uint32_t stream, uint32_t pos) {
    return int64_t(uint64_t(stream) << 32u | uint64_t(pos));
}

constexpr uint32_t llama_kvarn_decode_swa_stream(int64_t index) {
    return uint32_t(uint64_t(index) >> 32u);
}

struct llama_kvarn_type_desc {
    llama_kvarn_type type;
    const char * name;
    int key_bits;
    int value_bits;
    int group;
};

struct llama_kvarn_geometry {
    uint32_t token_group;
    uint32_t record_dim;
    uint32_t head_dim;
    uint32_t head_slices;
};

struct llama_kvarn_record_layout {
    uint32_t token_group;
    uint32_t record_dim;
    uint32_t rows;
    uint32_t cols;
    size_t payload_bytes;
    size_t scale_off;
    size_t zp_off;
    size_t other_off;
    size_t record_bytes;
};

struct llama_kvarn_tile_layout {
    size_t k_payload_off;
    size_t v_payload_off;
    size_t k_s_col_off;
    size_t k_zp_off;
    size_t k_s_row_off;
    size_t v_s_col_off;
    size_t v_s_row_off;
    size_t v_zp_off;

    size_t k_payload_bytes;
    size_t v_payload_bytes;
    size_t tile_bytes;
};

struct llama_kvarn_runtime_requirements {
    bool attention_supported;
    bool head_dims_supported;
    bool backend_ops_supported;
    uint32_t n_seq_max;
    bool kv_unified;
};

enum llama_kvarn_iswa_policy {
    LLAMA_KVARN_ISWA_DISABLED,
    LLAMA_KVARN_ISWA_ALL_LAYERS,
};

enum llama_kvarn_context_route {
    LLAMA_KVARN_CONTEXT_ROUTE_OWNED,
    LLAMA_KVARN_CONTEXT_ROUTE_SHARED_TARGET,
    LLAMA_KVARN_CONTEXT_ROUTE_UNSUPPORTED,
};

struct llama_kvarn_context_traits {
    llama_context_type ctx_type;
    llm_arch arch;
    bool has_ctx_other;
    bool dflash_has_dspark_head;
    bool dflash_has_selector;
};

llama_kvarn_context_route llama_kvarn_context_route_for(
        const llama_kvarn_context_traits & traits);

llama_kvarn_context_route llama_kvarn_context_route_for(
        llama_context_type ctx_type,
        llm_arch arch);

llama_kvarn_iswa_policy llama_kvarn_iswa_policy_for(
        bool enabled,
        bool has_swa,
        uint32_t n_seq_max);

size_t llama_kvarn_type_count();

const llama_kvarn_type_desc * llama_kvarn_type_desc_from_name(const char * name);
const llama_kvarn_type_desc * llama_kvarn_type_desc_from_type(llama_kvarn_type type);

llama_kvarn_tile_layout llama_kvarn_make_layout(
        uint32_t token_group,
        uint32_t record_dim,
        uint32_t head_dim,
        uint32_t head_slices,
        int      key_bits,
        int      value_bits);

llama_kvarn_record_layout llama_kvarn_record_layout(
        uint32_t token_group,
        uint32_t record_dim,
        uint32_t head_dim,
        uint32_t head_slices,
        int      bits,
        bool     value);

llama_kvarn_geometry llama_kvarn_geometry(
        uint32_t n_embd_head_k,
        uint32_t n_embd_head_v,
        uint32_t n_head_k,
        uint32_t n_head_v);

bool llama_kvarn_supported_head_dims(uint32_t head_dim);

bool llama_kvarn_backend_supports_non_causal_mask(ggml_backend_dev_t dev);

bool llama_kvarn_native_attention_allowed(bool causal_attn, llm_arch arch);

// Check if a context can use KVarN with the given traits
bool llama_kvarn_can_use_context(
        const llama_kvarn_context_traits & traits,
        const llama_kvarn_runtime_requirements & reqs);

// CPU quantization/dequantization (reference implementation)
void llama_kvarn_hadamard_64(float * values);
void llama_kvarn_hadamard_128(float * values);

void llama_kvarn_quantize_k_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record);

void llama_kvarn_quantize_v_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record);

void llama_kvarn_dequantize_k_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile);

void llama_kvarn_dequantize_v_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile);
