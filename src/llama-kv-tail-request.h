#pragma once

#include "llama.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// KV Tail Request
//
// A request specifies which tokens should be kept in high precision (the "tail")
// when using KVarN-quantized KV cache. The tail is a suffix of the KV cache
// that remains in F16/BF16 while the rest is quantized.
//
// The request is resolved into concrete storage parameters that are passed
// to the llama_kv_cache_kvarn constructor.
// ---------------------------------------------------------------------------

struct llama_kv_tail_request {
    // Number of tokens to keep in high precision at the tail.
    // 0 = disabled (all tokens quantized), UINT32_MAX = auto (use model default).
    uint32_t n_tokens;

    // Data type for the tail KV cache.
    // GGML_TYPE_COUNT = auto (use model default, typically F16).
    ggml_type type_k;
    ggml_type type_v;

    // Routing strategy: which backend handles the tail attention.
    // 0 = auto, 1 = native (GPU), 2 = generic (CPU fallback)
    uint32_t route;

    // Coverage threshold: minimum fraction of the tail that must be exact
    // before the tail is considered "complete". Values: 0..100 (percentage).
    // 100 = strict (all tail tokens must be exact), 0 = best-effort.
    uint32_t coverage_pct;

    bool is_enabled() const { return n_tokens > 0; }

    bool is_auto() const { return n_tokens == UINT32_MAX; }

    bool is_valid() const {
        return n_tokens == 0 || (n_tokens > 0 && type_k != GGML_TYPE_COUNT || type_k == GGML_TYPE_COUNT);
    }
};

// Default request: auto mode, F16 tail, native route, 100% coverage
inline llama_kv_tail_request llama_kv_tail_request_default() {
    llama_kv_tail_request req;
    req.n_tokens     = UINT32_MAX; // auto
    req.type_k       = GGML_TYPE_F16;
    req.type_v       = GGML_TYPE_F16;
    req.route        = 0;          // auto
    req.coverage_pct = 100;
    return req;
}

// Parse a KV tail request from command-line arguments.
// n_tokens: number of tokens (0 = disabled, -1 = auto)
// type_str: data type string ("f16", "bf16", "f32", etc.)
// route: routing strategy (0=auto, 1=native, 2=generic)
// coverage: coverage percentage (0-100)
llama_kv_tail_request llama_kv_tail_request_parse(
        int32_t n_tokens,
        const std::string & type_str,
        uint32_t route,
        uint32_t coverage_pct);

// Resolve a request against model parameters to produce concrete values.
// Returns the effective number of tail tokens and the resolved data type.
struct llama_kv_tail_resolution {
    uint32_t n_tokens;       // effective number of tail tokens
    ggml_type type_k;        // resolved K type
    ggml_type type_v;        // resolved V type
    uint32_t route;          // resolved route
    bool     downgraded;     // true if the requested type was downgraded
    std::string reason;      // human-readable explanation if downgraded
};

llama_kv_tail_resolution llama_kv_tail_resolve(
        const llama_kv_tail_request & req,
        const llama_model * model,
        uint32_t n_ctx);

// Parse the --kv-tail-tokens CLI argument value.
// Accepts: integer (0, 128, 1024, ...), "auto" for UINT32_MAX
uint32_t llama_kv_tail_parse_tokens(const std::string & value);

// Parse the --kv-tail-type CLI argument value.
ggml_type llama_kv_tail_parse_type(const std::string & value);
