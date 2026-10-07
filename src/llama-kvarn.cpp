#include "llama-kvarn.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#define LLAMA_KVARN_DESC(KB, VB) { LLAMA_KVARN_K##KB##V##VB##_G128, "kvarn_k" #KB "v" #VB "_g128", KB, VB, 128 }

static constexpr std::array<llama_kvarn_type_desc, LLAMA_KVARN_TYPE_COUNT> KVAR_N_TYPES = {{
    { LLAMA_KVARN_TYPE_DISABLED, "off", 0, 0, 128 },

    LLAMA_KVARN_DESC(2, 2),
    LLAMA_KVARN_DESC(2, 3),
    LLAMA_KVARN_DESC(2, 4),

    LLAMA_KVARN_DESC(3, 2),
    LLAMA_KVARN_DESC(3, 3),
    LLAMA_KVARN_DESC(3, 4),

    LLAMA_KVARN_DESC(4, 2),
    LLAMA_KVARN_DESC(4, 3),
    LLAMA_KVARN_DESC(4, 4),

    LLAMA_KVARN_DESC(2, 5),
    LLAMA_KVARN_DESC(2, 6),
    LLAMA_KVARN_DESC(2, 8),

    LLAMA_KVARN_DESC(3, 5),
    LLAMA_KVARN_DESC(3, 6),
    LLAMA_KVARN_DESC(3, 8),

    LLAMA_KVARN_DESC(4, 5),
    LLAMA_KVARN_DESC(4, 6),
    LLAMA_KVARN_DESC(4, 8),

    LLAMA_KVARN_DESC(5, 2),
    LLAMA_KVARN_DESC(5, 3),
    LLAMA_KVARN_DESC(5, 4),
    LLAMA_KVARN_DESC(5, 5),
    LLAMA_KVARN_DESC(5, 6),
    LLAMA_KVARN_DESC(5, 8),

    LLAMA_KVARN_DESC(6, 2),
    LLAMA_KVARN_DESC(6, 3),
    LLAMA_KVARN_DESC(6, 4),
    LLAMA_KVARN_DESC(6, 5),
    LLAMA_KVARN_DESC(6, 6),
    LLAMA_KVARN_DESC(6, 8),

    LLAMA_KVARN_DESC(8, 2),
    LLAMA_KVARN_DESC(8, 3),
    LLAMA_KVARN_DESC(8, 4),
    LLAMA_KVARN_DESC(8, 5),
    LLAMA_KVARN_DESC(8, 6),
    LLAMA_KVARN_DESC(8, 8),
}};

//
// Type lookup
//

size_t llama_kvarn_type_count() {
    return KVAR_N_TYPES.size();
}

const llama_kvarn_type_desc * llama_kvarn_type_desc_from_name(const char * name) {
    if (name == nullptr) {
        return nullptr;
    }
    for (const auto & desc : KVAR_N_TYPES) {
        if (std::strcmp(desc.name, name) == 0) {
            return &desc;
        }
    }
    return nullptr;
}

const llama_kvarn_type_desc * llama_kvarn_type_desc_from_type(llama_kvarn_type type) {
    for (const auto & desc : KVAR_N_TYPES) {
        if (desc.type == type) {
            return &desc;
        }
    }
    return nullptr;
}

const char * llama_kvarn_type_name(llama_kvarn_type type) {
    const auto * desc = llama_kvarn_type_desc_from_type(type);
    return desc ? desc->name : "invalid";
}

llama_kvarn_type llama_kvarn_type_from_name(const char * name) {
    const auto * desc = llama_kvarn_type_desc_from_name(name);
    return desc ? desc->type : LLAMA_KVARN_TYPE_INVALID;
}

//
// Params
//

llama_kvarn_params llama_kvarn_default_params() {
    return {
        /*.type                =*/ LLAMA_KVARN_TYPE_DISABLED,
        /*.key_bits            =*/ 0,
        /*.value_bits          =*/ 0,
        /*.swa_key_bits        =*/ 0,
        /*.swa_value_bits      =*/ 0,
        /*.group               =*/ 128,
        /*.sinkhorn_iters      =*/ 16,
        /*.sink_tokens         =*/ 128,
        /*.window_chunk        =*/ 0,
        /*.fail_if_unsupported =*/ true,
    };
}

llama_kvarn_params llama_kvarn_params_for_type(llama_kvarn_type type) {
    llama_kvarn_params result = llama_kvarn_default_params();
    result.type = type;

    const auto * desc = llama_kvarn_type_desc_from_type(type);
    if (desc != nullptr) {
        result.key_bits   = desc->key_bits;
        result.value_bits = desc->value_bits;
        result.group      = desc->group;
    }
    return result;
}

//
// Validation
//

static bool kvarn_valid_bits(int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

static bool kvarn_valid_bit_pair(int key_bits, int value_bits) {
    for (const auto & desc : KVAR_N_TYPES) {
        if (desc.key_bits == key_bits && desc.value_bits == value_bits && desc.group == 128) {
            return desc.type != LLAMA_KVARN_TYPE_DISABLED && desc.type != LLAMA_KVARN_TYPE_INVALID;
        }
    }
    return false;
}

const char * llama_kvarn_validate_runtime(
        const llama_kvarn_params & params,
        const llama_kvarn_runtime_requirements & requirements) {
    if (params.type == LLAMA_KVARN_TYPE_DISABLED) {
        return nullptr;
    }

    const auto * desc = llama_kvarn_type_desc_from_type(params.type);
    if (desc == nullptr || desc->type == LLAMA_KVARN_TYPE_DISABLED) {
        return "invalid KVarN cache type";
    }
    if (params.key_bits != desc->key_bits || params.value_bits != desc->value_bits || params.group != desc->group) {
        return "KVarN cache parameters do not match the selected preset";
    }
    if (!kvarn_valid_bits(params.key_bits) || !kvarn_valid_bits(params.value_bits)) {
        return "KVarN supports only 2-, 3-, 4-, 5-, 6-, and 8-bit cache payloads";
    }
    if ((params.swa_key_bits != 0 && !kvarn_valid_bits(params.swa_key_bits)) ||
            (params.swa_value_bits != 0 && !kvarn_valid_bits(params.swa_value_bits))) {
        return "KVarN SWA overrides support only 2-, 3-, 4-, 5-, 6-, and 8-bit cache payloads";
    }
    if ((params.swa_key_bits == 0) != (params.swa_value_bits == 0)) {
        return "KVarN SWA override must specify both K and V bits";
    }
    if (params.swa_key_bits != 0 && !kvarn_valid_bit_pair(params.swa_key_bits, params.swa_value_bits)) {
        return "invalid KVarN SWA override bit combination";
    }
    if (params.group != 128) {
        return "KVarN currently requires a group size of 128 tokens";
    }
    if (params.sinkhorn_iters <= 0) {
        return "KVarN requires at least one Sinkhorn iteration";
    }
    if (params.sink_tokens != 128) {
        return "KVarN currently requires exactly 128 unquantized sink tokens";
    }
    if (!requirements.attention_supported) {
        return "KVarN is not supported by this attention/cache path";
    }
    if (!requirements.head_dims_supported) {
        return "KVarN requires 64-, 128-, 256-, or 512-dimensional key/value heads";
    }
    if (!requirements.backend_ops_supported) {
        return "KVarN requires a backend with KVarN store and materialization support";
    }
    return nullptr;
}

//
// Context routing
//

llama_kvarn_context_route llama_kvarn_context_route_for(
        const llama_kvarn_context_traits & traits) {
    if (traits.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
        return LLAMA_KVARN_CONTEXT_ROUTE_OWNED;
    }

    if (traits.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        // MTP draft contexts own their KV cache
        return LLAMA_KVARN_CONTEXT_ROUTE_OWNED;
    }

    return LLAMA_KVARN_CONTEXT_ROUTE_UNSUPPORTED;
}

llama_kvarn_context_route llama_kvarn_context_route_for(
        llama_context_type ctx_type,
        llm_arch arch) {
    return llama_kvarn_context_route_for({ ctx_type, arch, false, false, false });
}

llama_kvarn_iswa_policy llama_kvarn_iswa_policy_for(
        bool enabled,
        bool has_swa,
        uint32_t n_seq_max) {
    if (!enabled) {
        return LLAMA_KVARN_ISWA_DISABLED;
    }
    GGML_UNUSED(has_swa);
    GGML_UNUSED(n_seq_max);
    return LLAMA_KVARN_ISWA_ALL_LAYERS;
}

bool llama_kvarn_can_use_context(
        const llama_kvarn_context_traits & traits,
        const llama_kvarn_runtime_requirements & reqs) {
    auto route = llama_kvarn_context_route_for(traits);
    if (route == LLAMA_KVARN_CONTEXT_ROUTE_UNSUPPORTED) {
        return false;
    }
    return reqs.attention_supported && reqs.head_dims_supported && reqs.backend_ops_supported;
}

//
// Head dim support
//

bool llama_kvarn_supported_head_dims(uint32_t head_dim) {
    return head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512;
}

bool llama_kvarn_backend_supports_non_causal_mask(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return false;
    auto * fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_capabilities");
    (void)fn;
    // For now, assume any GPU backend that registers the capability supports non-causal
    return true;
}

bool llama_kvarn_native_attention_allowed(bool causal_attn, llm_arch arch) {
    return causal_attn || arch != LLM_ARCH_DFLASH;
}

//
// Geometry
//

struct llama_kvarn_geometry llama_kvarn_geometry(
        uint32_t n_embd_head_k,
        uint32_t n_embd_head_v,
        uint32_t n_head_k,
        uint32_t n_head_v) {
    GGML_UNUSED(n_embd_head_v);
    GGML_UNUSED(n_head_v);

    uint32_t head_dim = n_embd_head_k;
    uint32_t head_slices = 1;

    // For head dims that are multiples of 128, we can use head slicing
    if (head_dim == 256) {
        head_slices = 2;
    } else if (head_dim == 512) {
        head_slices = 4;
    }

    return {
        /*.token_group =*/ KVAR_N_GROUP,
        /*.record_dim  =*/ head_slices * 128,
        /*.head_dim    =*/ head_dim,
        /*.head_slices =*/ head_slices,
    };
}

//
// Layout
//

static size_t kvarn_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct llama_kvarn_record_layout llama_kvarn_record_layout(
        uint32_t token_group,
        uint32_t record_dim,
        uint32_t head_dim,
        uint32_t head_slices,
        int      bits,
        bool     value) {
    GGML_UNUSED(head_dim);

    const uint32_t rows = token_group;
    const uint32_t cols = record_dim;

    // payload: packed bits
    size_t payload_bytes = kvarn_align_up(size_t(rows) * cols * bits / 8, 4);

    // scales: per-column float32 (one per row group of 1)
    size_t s_col_bytes = kvarn_align_up(size_t(rows) * sizeof(float), 4);
    size_t s_row_bytes = kvarn_align_up(size_t(cols) * sizeof(float), 4);

    // zero-point: per-column float32
    size_t zp_bytes = kvarn_align_up(size_t(cols) * sizeof(float), 4);

    size_t offset = 0;
    size_t payload_off = offset; offset += payload_bytes;
    size_t s_col_off   = offset; offset += s_col_bytes;
    size_t s_row_off   = offset; offset += s_row_bytes;
    size_t zp_off      = offset; offset += zp_bytes;

    return {
        /*.token_group   =*/ token_group,
        /*.record_dim    =*/ record_dim,
        /*.rows          =*/ rows,
        /*.cols          =*/ cols,
        /*.payload_bytes =*/ payload_bytes,
        /*.scale_off     =*/ s_col_off,
        /*.zp_off        =*/ zp_off,
        /*.other_off     =*/ s_row_off,
        /*.record_bytes  =*/ offset,
    };
}

llama_kvarn_tile_layout llama_kvarn_make_layout(
        uint32_t token_group,
        uint32_t record_dim,
        uint32_t head_dim,
        uint32_t head_slices,
        int      key_bits,
        int      value_bits) {
    auto k = llama_kvarn_record_layout(token_group, record_dim, head_dim, head_slices, key_bits, false);
    auto v = llama_kvarn_record_layout(token_group, record_dim, head_dim, head_slices, value_bits, true);

    return {
        /*.k_payload_off =*/ 0,
        /*.v_payload_off =*/ k.record_bytes,
        /*.k_s_col_off   =*/ k.scale_off,
        /*.k_zp_off      =*/ k.zp_off,
        /*.k_s_row_off   =*/ k.other_off,
        /*.v_s_col_off   =*/ k.record_bytes + v.scale_off,
        /*.v_s_row_off   =*/ k.record_bytes + v.other_off,
        /*.v_zp_off      =*/ k.record_bytes + v.zp_off,

        /*.k_payload_bytes =*/ k.payload_bytes,
        /*.v_payload_bytes =*/ v.payload_bytes,
        /*.tile_bytes      =*/ k.record_bytes + v.record_bytes,
    };
}

//
// Hadamard transforms (reference CPU implementation)
//

void llama_kvarn_hadamard_64(float * values) {
    // In-place Walsh-Hadamard transform for 64 elements
    // Uses the iterative fast Hadamard transform (FHT)
    constexpr int N = 64;
    for (int h = 1; h < N; h *= 2) {
        for (int i = 0; i < N; i += 2 * h) {
            for (int j = 0; j < h; j++) {
                float u = values[i + j];
                float v = values[i + j + h];
                values[i + j]       = u + v;
                values[i + j + h]   = u - v;
            }
        }
    }
    // Normalize
    const float scale = 1.0f / std::sqrt(float(N));
    for (int i = 0; i < N; i++) {
        values[i] *= scale;
    }
}

void llama_kvarn_hadamard_128(float * values) {
    // In-place Walsh-Hadamard transform for 128 elements
    constexpr int N = 128;
    for (int h = 1; h < N; h *= 2) {
        for (int i = 0; i < N; i += 2 * h) {
            for (int j = 0; j < h; j++) {
                float u = values[i + j];
                float v = values[i + j + h];
                values[i + j]       = u + v;
                values[i + j + h]   = u - v;
            }
        }
    }
    // Normalize
    const float scale = 1.0f / std::sqrt(float(N));
    for (int i = 0; i < N; i++) {
        values[i] *= scale;
    }
}

//
// CPU quantization / dequantization (reference implementation)
//

// Sinkhorn-Knopp normalization for the quantization matrix
static void sinkhorn_normalize(float * mat, int rows, int cols, int iters) {
    for (int iter = 0; iter < iters; iter++) {
        // Normalize rows
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                sum += std::abs(mat[r * cols + c]);
            }
            if (sum > 1e-10f) {
                for (int c = 0; c < cols; c++) {
                    mat[r * cols + c] /= sum;
                }
            }
        }
        // Normalize columns
        for (int c = 0; c < cols; c++) {
            float sum = 0.0f;
            for (int r = 0; r < rows; r++) {
                sum += std::abs(mat[r * cols + c]);
            }
            if (sum > 1e-10f) {
                for (int r = 0; r < rows; r++) {
                    mat[r * cols + c] /= sum;
                }
            }
        }
    }
}

// Pack values into bit-packed format
static void pack_bits(const float * values, int n, int bits, uint8_t * out) {
    memset(out, 0, (n * bits + 7) / 8);
    const int max_val = (1 << bits) - 1;
    for (int i = 0; i < n; i++) {
        // values should already be in [0, max_val] range
        int val = static_cast<int>(values[i] + 0.5f);
        val = std::max(0, std::min(max_val, val));
        int bit_pos = i * bits;
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        for (int b = 0; b < bits; b++) {
            if (val & (1 << b)) {
                out[byte_idx + b / 8] |= (1 << (b % 8));
            }
        }
    }
}

// Unpack bit-packed format
static void unpack_bits(const uint8_t * in, int n, int bits, float * out) {
    for (int i = 0; i < n; i++) {
        int bit_pos = i * bits;
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        int val = 0;
        for (int b = 0; b < bits; b++) {
            if (in[byte_idx + b / 8] & (1 << (bit_off + b) % 8)) {
                val |= (1 << b);
            }
        }
        out[i] = static_cast<float>(val);
    }
}

void llama_kvarn_quantize_k_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record) {
    const uint32_t rows = KVAR_N_GROUP;
    const uint32_t cols = KVAR_N_GROUP; // record_dim for 128-dim heads
    const size_t n = size_t(rows) * cols;

    // 1. Apply Hadamard transform per row (128-dim)
    std::vector<float> rotated(n);
    for (uint32_t r = 0; r < rows; r++) {
        // Copy row
        for (uint32_t c = 0; c < cols; c++) {
            rotated[r * cols + c] = tile[r * cols + c];
        }
        // Apply 128-point Hadamard (for 128-dim heads)
        if (cols == 128) {
            llama_kvarn_hadamard_128(&rotated[r * cols]);
        } else if (cols == 64) {
            llama_kvarn_hadamard_64(&rotated[r * cols]);
        }
    }

    // 2. Sinkhorn normalization
    sinkhorn_normalize(rotated.data(), rows, cols, sinkhorn_iters);

    // 3. Compute per-column scale and zero-point
    std::vector<float> s_col(cols, 0.0f);
    std::vector<float> zp(cols, 0.0f);
    const float max_val = static_cast<float>((1 << bits) - 1);

    for (uint32_t c = 0; c < cols; c++) {
        float col_min = std::numeric_limits<float>::max();
        float col_max = std::numeric_limits<float>::min();
        for (uint32_t r = 0; r < rows; r++) {
            float v = rotated[r * cols + c];
            col_min = std::min(col_min, v);
            col_max = std::max(col_max, v);
        }
        float range = col_max - col_min;
        s_col[c] = (range > 1e-10f) ? range / max_val : 1.0f;
        zp[c] = -col_min / s_col[c];
    }

    // 4. Quantize and pack
    std::vector<float> quantized(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t c = i % cols;
        float q = rotated[i] / s_col[c] + zp[c];
        quantized[i] = std::max(0.0f, std::min(max_val, q));
    }

    // Pack into the record at the K payload offset
    pack_bits(quantized.data(), n, bits, record + layout.k_payload_off);

    // Store scales
    for (uint32_t c = 0; c < cols; c++) {
        reinterpret_cast<float *>(record + layout.k_s_col_off)[c] = s_col[c];
        reinterpret_cast<float *>(record + layout.k_zp_off)[c]    = zp[c];
    }
}

void llama_kvarn_quantize_v_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record) {
    const uint32_t rows = KVAR_N_GROUP;
    const uint32_t cols = KVAR_N_GROUP;
    const size_t n = size_t(rows) * cols;

    // V does NOT get Hadamard rotation - direct quantization
    std::vector<float> values(n);
    for (size_t i = 0; i < n; i++) {
        values[i] = tile[i];
    }

    // Sinkhorn normalization
    sinkhorn_normalize(values.data(), rows, cols, sinkhorn_iters);

    // Per-column scale and zero-point
    std::vector<float> s_col(cols, 0.0f);
    std::vector<float> zp(cols, 0.0f);
    const float max_val = static_cast<float>((1 << bits) - 1);

    for (uint32_t c = 0; c < cols; c++) {
        float col_min = std::numeric_limits<float>::max();
        float col_max = std::numeric_limits<float>::min();
        for (uint32_t r = 0; r < rows; r++) {
            float v = values[r * cols + c];
            col_min = std::min(col_min, v);
            col_max = std::max(col_max, v);
        }
        float range = col_max - col_min;
        s_col[c] = (range > 1e-10f) ? range / max_val : 1.0f;
        zp[c] = -col_min / s_col[c];
    }

    // Quantize and pack
    std::vector<float> quantized(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t c = i % cols;
        float q = values[i] / s_col[c] + zp[c];
        quantized[i] = std::max(0.0f, std::min(max_val, q));
    }

    pack_bits(quantized.data(), n, bits, record + layout.v_payload_off);

    for (uint32_t c = 0; c < cols; c++) {
        reinterpret_cast<float *>(record + layout.v_s_col_off)[c] = s_col[c];
        reinterpret_cast<float *>(record + layout.v_zp_off)[c]    = zp[c];
    }
}

void llama_kvarn_dequantize_k_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile) {
    const uint32_t rows = KVAR_N_GROUP;
    const uint32_t cols = KVAR_N_GROUP;
    const size_t n = size_t(rows) * cols;

    // Unpack
    std::vector<float> quantized(n);
    unpack_bits(record + layout.k_payload_off, n, bits, quantized.data());

    // Dequantize
    const float * s_col = reinterpret_cast<const float *>(record + layout.k_s_col_off);
    const float * zp    = reinterpret_cast<const float *>(record + layout.k_zp_off);

    for (uint32_t i = 0; i < n; i++) {
        uint32_t c = i % cols;
        tile[i] = (quantized[i] - zp[c]) * s_col[c];
    }

    // Inverse Hadamard (Hadamard is its own inverse up to scaling)
    for (uint32_t r = 0; r < rows; r++) {
        if (cols == 128) {
            llama_kvarn_hadamard_128(&tile[r * cols]);
        } else if (cols == 64) {
            llama_kvarn_hadamard_64(&tile[r * cols]);
        }
    }
}

void llama_kvarn_dequantize_v_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile) {
    const uint32_t rows = KVAR_N_GROUP;
    const uint32_t cols = KVAR_N_GROUP;
    const size_t n = size_t(rows) * cols;

    // Unpack
    std::vector<float> quantized(n);
    unpack_bits(record + layout.v_payload_off, n, bits, quantized.data());

    // Dequantize (no Hadamard for V)
    const float * s_col = reinterpret_cast<const float *>(record + layout.v_s_col_off);
    const float * zp    = reinterpret_cast<const float *>(record + layout.v_zp_off);

    for (uint32_t i = 0; i < n; i++) {
        uint32_t c = i % cols;
        tile[i] = (quantized[i] - zp[c]) * s_col[c];
    }
}
