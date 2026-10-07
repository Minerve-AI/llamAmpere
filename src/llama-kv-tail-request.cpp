#include "llama-kv-tail-request.h"

#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

uint32_t llama_kv_tail_parse_tokens(const std::string & value) {
    if (value == "auto" || value == "max") {
        return UINT32_MAX;
    }
    try {
        int32_t v = std::stoi(value);
        if (v < 0) {
            return UINT32_MAX; // negative = auto
        }
        return (uint32_t)v;
    } catch (...) {
        throw std::invalid_argument(
            "llama_kv_tail_parse_tokens: invalid value '" + value +
            "', expected integer or 'auto'");
    }
}

ggml_type llama_kv_tail_parse_type(const std::string & value) {
    if (value == "f16" || value == "fp16" || value == "half") {
        return GGML_TYPE_F16;
    }
    if (value == "bf16" || value == "bfloat16") {
        return GGML_TYPE_BF16;
    }
    if (value == "f32" || value == "float") {
        return GGML_TYPE_F32;
    }
    if (value == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (value == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (value == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    throw std::invalid_argument(
        "llama_kv_tail_parse_type: unsupported type '" + value +
        "', expected f16, bf16, f32, q8_0, q4_0, q4_1");
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

llama_kv_tail_request llama_kv_tail_request_parse(
        int32_t n_tokens,
        const std::string & type_str,
        uint32_t route,
        uint32_t coverage_pct) {
    llama_kv_tail_request req;
    req.n_tokens = (n_tokens < 0) ? UINT32_MAX : (uint32_t)n_tokens;
    req.type_k   = type_str.empty() ? GGML_TYPE_F16 : llama_kv_tail_parse_type(type_str);
    req.type_v   = req.type_k;
    req.route    = route;
    req.coverage_pct = std::min(coverage_pct, (uint32_t)100);
    return req;
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

llama_kv_tail_resolution llama_kv_tail_resolve(
        const llama_kv_tail_request & req,
        const llama_model * model,
        uint32_t n_ctx) {

    llama_kv_tail_resolution res;
    res.downgraded = false;

    // Resolve token count
    if (req.n_tokens == 0) {
        // Disabled
        res.n_tokens = 0;
    } else if (req.is_auto()) {
        // Auto: use the KVarN default (1024 tokens), capped by context window
        uint32_t default_tokens = 1024;
        if (n_ctx > 0) {
            // Use min(default, n_ctx/4) to avoid excessive memory usage
            default_tokens = std::min<uint32_t>(default_tokens, n_ctx / 4);
        }
        res.n_tokens = default_tokens;
    } else {
        // Explicit: use the requested count, capped by context window
        res.n_tokens = (n_ctx > 0) ? std::min(req.n_tokens, n_ctx) : req.n_tokens;
    }

    // Resolve data type
    // Tail types must be at least as precise as the body quantization.
    // For KVarN, the body is 2-8 bits, so F16 tail is always valid.
    res.type_k = req.type_k;
    res.type_v = req.type_v;

    // If model is provided, we could check backend capabilities here.
    // For now, all standard types (F16, BF16, F32) are accepted.
    (void)model;

    // Resolve route
    res.route = req.route;
    if (res.route == 0) {
        // Auto: use native (GPU) route
        res.route = 1;
    }

    if (res.downgraded) {
        res.reason = "tail type downgraded for backend compatibility";
    }

    return res;
}
