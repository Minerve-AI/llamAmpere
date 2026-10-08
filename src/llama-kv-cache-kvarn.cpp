#include "llama-kv-cache-kvarn.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kvarn.h"
#include "ggml-backend-impl.h"

#include <cstring>

// Stub declarations for meta device functions (not available in llamAmpere)
static inline size_t ggml_backend_meta_device_count(ggml_backend_dev_t) { return 0; }
static inline ggml_backend_dev_t ggml_backend_meta_device_get(ggml_backend_dev_t, size_t) { return nullptr; }

//
// Backend capability queries
//

bool llama_kvarn_backend_supports_native_ops(ggml_backend_dev_t dev) {
    if (dev == nullptr) return false;
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return false;
    auto * fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_capabilities");
    return fn != nullptr;
}

bool llama_kvarn_backend_supports_ops(ggml_backend_dev_t dev, int head_dim) {
    if (dev == nullptr) return false;
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return false;
    auto * fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_capabilities");
    if (!fn) return false;
    (void)head_dim;
    return fn != nullptr;
}

bool llama_kvarn_backend_native_attention_uses_original_v(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return false;
}

uint32_t llama_kvarn_backend_native_rotated_max_query_tokens(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return 0;
}

bool llama_kvarn_backend_mixed_tail_native_preferred(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return false;
}

//
// llama_kv_cache_kvarn_context
//

llama_kv_cache_kvarn_context::llama_kv_cache_kvarn_context(
        llama_kv_cache_kvarn * cache,
        llama_memory_context_ptr base,
        llama_context * update_lctx,
        std::vector<int32_t> shared_graph_layers)
    : llama_kv_cache_context(LLAMA_MEMORY_STATUS_SUCCESS)
    , base_ctx(std::move(base))
    , cache(cache)
    , update_lctx(update_lctx)
    , shared_graph_layers(std::move(shared_graph_layers)) {
}

llama_kv_cache_context * llama_kv_cache_kvarn_context::base() const {
    return static_cast<llama_kv_cache_context *>(base_ctx.get());
}

int32_t llama_kv_cache_kvarn_context::graph_layer_for(int32_t il) const {
    GGML_UNUSED(il);
    return -1;
}

bool llama_kv_cache_kvarn_context::uses_shared_live_indices() const {
    return false;
}

bool llama_kv_cache_kvarn_context::next() {
    return base()->next();
}

bool llama_kv_cache_kvarn_context::apply() {
    return base()->apply();
}

void llama_kv_cache_kvarn_context::graph_compute_start() {
    base()->graph_compute_start();
}

void llama_kv_cache_kvarn_context::graph_compute_finish(ggml_status compute_status) {
    base()->graph_compute_finish(compute_status);
}

llama_memory_status llama_kv_cache_kvarn_context::get_status() const {
    return base()->get_status();
}

const llama_ubatch & llama_kv_cache_kvarn_context::get_ubatch() const {
    return base()->get_ubatch();
}

uint32_t llama_kv_cache_kvarn_context::get_n_kv() const {
    return base()->get_n_kv();
}

llama_kv_cache * llama_kv_cache_kvarn_context::get_kv() const {
    return base()->get_kv();
}

const llama_kv_cache::slot_info & llama_kv_cache_kvarn_context::current_sinfo() const {
    return base()->current_sinfo();
}

const llama_kv_cache_context::slot_info_vec_t & llama_kv_cache_kvarn_context::get_sinfos() const {
    return base()->get_sinfos();
}

void llama_kv_cache_kvarn_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    base()->get_prev_tokens(ubatch, n, res);
}

ggml_type llama_kv_cache_kvarn_context::type_k() const {
    return base()->type_k();
}

ggml_type llama_kv_cache_kvarn_context::type_v() const {
    return base()->type_v();
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k(ggml_context * ctx, int32_t il) const {
    return base()->get_k(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v(ggml_context * ctx, int32_t il) const {
    return base()->get_v(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_tail(ggml_context * ctx, int32_t il) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(il);
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_tail(ggml_context * ctx, int32_t il) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(il);
    return nullptr;
}

uint32_t llama_kv_cache_kvarn_context::get_tail_slots() const { return 0; }
ggml_type llama_kv_cache_kvarn_context::get_tail_type() const { return GGML_TYPE_F16; }
uint32_t llama_kv_cache_kvarn_context::get_tail_tokens() const { return 0; }
uint32_t llama_kv_cache_kvarn_context::get_tail_arena_stride() const { return 0; }
uint32_t llama_kv_cache_kvarn_context::get_tail_attention_stride(uint32_t n_query_tokens) const { return 0; }
uint32_t llama_kv_cache_kvarn_context::get_tail_body_execution_stride() const { return 0; }
uint32_t llama_kv_cache_kvarn_context::get_tail_body_execution_rows(int32_t il) const { return 0; }
bool llama_kv_cache_kvarn_context::has_compact_tail() const { return false; }
bool llama_kv_cache_kvarn_context::has_kv_body() const { return true; }
bool llama_kv_cache_kvarn_context::has_kv_body(int32_t il) const { return true; }
bool llama_kv_cache_kvarn_context::has_tail_current(int32_t il) const { return false; }
ggml_backend_dev_t llama_kv_cache_kvarn_context::get_tail_backend(int32_t il) const { return nullptr; }
llama_kv_tail_storage_kind llama_kv_cache_kvarn_context::get_tail_storage_kind() const { return LLAMA_KV_TAIL_STORAGE_DISABLED; }
uint32_t llama_kv_cache_kvarn_context::get_tail_rollback_tokens() const { return 0; }
llama_kv_tail_route llama_kv_cache_kvarn_context::get_tail_route(int32_t il) const { return LLAMA_KV_TAIL_ROUTE_NONE; }
const llama_kv_tail_layer_route * llama_kv_cache_kvarn_context::get_tail_layer_route(int32_t il) const { return nullptr; }
bool llama_kv_cache_kvarn_context::get_tail_explicit_bias(int32_t il) const { return false; }
bool llama_kv_cache_kvarn_context::can_pack_tail_body(const llama_ubatch & ubatch) const { return false; }

ggml_tensor * llama_kv_cache_kvarn_context::get_k_native(ggml_context * ctx, int32_t il) const {
    return base()->get_k(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_native(ggml_context * ctx, int32_t il) const {
    return base()->get_v(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_for_attention(ggml_context * ctx, int32_t il, bool native_attention) const {
    return base()->get_k(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_for_attention(ggml_context * ctx, int32_t il, bool native_attention) const {
    return base()->get_v(ctx, il);
}

bool llama_kv_cache_kvarn_context::uses_native_attention(int32_t il) const {
    GGML_UNUSED(il);
    return false;
}

bool llama_kv_cache_kvarn_context::has_qualified_dflash_mask() const { return false; }
ggml_backend_dev_t llama_kv_cache_kvarn_context::native_attention_backend(int32_t il) const { return nullptr; }
bool llama_kv_cache_kvarn_context::mixed_tail_native_preferred(int32_t il) const { return false; }
bool llama_kv_cache_kvarn_context::native_attention_uses_original_v(int32_t il) const { return false; }
uint32_t llama_kv_cache_kvarn_context::native_rotated_max_query_tokens(int32_t il) const { return 0; }
bool llama_kv_cache_kvarn_context::uses_compact_read_indices() const { return false; }
bool llama_kv_cache_kvarn_context::uses_materialization_indices() const { return false; }

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_rot(ggml_context * ctx, int n_rot) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(n_rot);
    return nullptr;
}

void llama_kv_cache_kvarn_context::set_input_kvarn_rot(ggml_tensor * dst) const {
    GGML_UNUSED(dst);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_mat_idxs(ggml_context * ctx) const {
    GGML_UNUSED(ctx);
    return nullptr;
}

void llama_kv_cache_kvarn_context::set_input_kvarn_mat_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return base()->cpy_k(ctx, k_cur, k_idxs, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return base()->cpy_v(ctx, v_cur, v_idxs, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k_with_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs,
        ggml_tensor * tail_idxs, int32_t il) const {
    GGML_UNUSED(tail_idxs);
    return base()->cpy_k(ctx, k_cur, k_idxs, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v_with_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs,
        ggml_tensor * tail_idxs, int32_t il) const {
    GGML_UNUSED(tail_idxs);
    return base()->cpy_v(ctx, v_cur, v_idxs, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(k_cur);
    GGML_UNUSED(tail_idxs);
    GGML_UNUSED(il);
    GGML_UNUSED(dependency);
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(v_cur);
    GGML_UNUSED(tail_idxs);
    GGML_UNUSED(il);
    GGML_UNUSED(dependency);
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ubatch);
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_tail_body_idxs(ggml_context * ctx) const {
    GGML_UNUSED(ctx);
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_rot(ggml_context * ctx) const {
    return base()->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_rot(ggml_context * ctx) const {
    return base()->build_input_v_rot(ctx);
}

void llama_kv_cache_kvarn_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_k_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_v_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
}

void llama_kv_cache_kvarn_context::set_input_tail_body_idxs(ggml_tensor * dst) const {
    GGML_UNUSED(dst);
}

void llama_kv_cache_kvarn_context::set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
}

void llama_kv_cache_kvarn_context::set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_shift(ggml_tensor * dst) const {
    base()->set_input_k_shift(dst);
}

void llama_kv_cache_kvarn_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    base()->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_kvarn_context::set_input_kq_mask_tail(
        ggml_tensor * body, ggml_tensor * exact,
        ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
        const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_UNUSED(body);
    GGML_UNUSED(exact);
    GGML_UNUSED(read_idxs);
    GGML_UNUSED(body_read_idxs);
    GGML_UNUSED(bias_read_idxs);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(causal_attn);
}

void llama_kv_cache_kvarn_context::set_input_tail_body_plan(
        ggml_tensor * query_order, ggml_tensor * run_desc,
        ggml_tensor * body_mask, const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_UNUSED(query_order);
    GGML_UNUSED(run_desc);
    GGML_UNUSED(body_mask);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(causal_attn);
}

void llama_kv_cache_kvarn_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_rot(ggml_tensor * dst) const {
    base()->set_input_k_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot(ggml_tensor * dst) const {
    base()->set_input_v_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_k_rot_backend(ggml_tensor * dst) const {
    GGML_UNUSED(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot_backend(ggml_tensor * dst) const {
    GGML_UNUSED(dst);
}

//
// llama_kv_cache_kvarn implementation
//

std::unique_ptr<llama_kv_cache> llama_kv_cache_kvarn::make_metadata_cache() const {
    llama_kv_cache_params params;
    params.n_seq_max  = n_seq_max;
    params.n_batch    = 0;
    params.n_ubatch   = metadata_n_ubatch;
    params.offload    = true;
    params.swa_full   = false;
    params.n_pad      = metadata_n_pad;
    params.n_swa      = metadata_n_swa;
    params.swa_type   = metadata_swa_type;
    params.type_k     = exact_tail_type_requested;
    params.type_v     = exact_tail_type_requested;
    params.kv_unified = false;

    auto mem = std::make_unique<llama_kv_cache>(
            model, hparams, params, kv_size);
    return mem;
}

llama_kv_cache_kvarn::llama_kv_cache_kvarn(
        const llama_model & model,
        const llama_hparams & hparams,
        llama_kvarn_params params,
        bool offload,
        bool unified,
        uint32_t kv_size,
        uint32_t n_seq_max,
        uint32_t n_batch,
        uint32_t n_ubatch,
        uint32_t n_pad,
        uint32_t n_swa,
        llama_swa_type swa_type,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse,
        uint32_t tail_tokens,
        ggml_type tail_type,
        uint32_t tail_tokens_requested,
        uint32_t tail_rollback_tokens)
    : model(model)
    , hparams(hparams)
    , params(params)
    , n_stream(1)
    , n_seq_max(n_seq_max)
    , kv_size(kv_size)
    , tail_groups(2)
    , stage_groups(swa_type != LLAMA_SWA_TYPE_NONE ? 2 : 3)
    , swa(n_swa > 0)
    , n_groups_per_stream(1)
    , exact_tail_tokens(tail_tokens)
    , metadata_n_pad(n_pad)
    , metadata_n_swa(n_swa)
    , metadata_swa_type(swa_type)
    , metadata_n_ubatch(n_ubatch)
    , exact_tail_tokens_requested(tail_tokens_requested)
    , exact_tail_type_requested(tail_type)
    , exact_tail_type(tail_type) {

    metadata = make_metadata_cache();

    // Set up layers
    const uint32_t n_layer = hparams.n_layer();
    layers.reserve(n_layer);
    for (uint32_t il = 0; il < n_layer; il++) {
        if (filter && !filter(il)) {
            continue;
        }
        layer l;
        l.il = il;
        l.n_head_kv = hparams.n_head_kv(il);
        l.head_dim_k = hparams.n_head(il) * hparams.rope_freq_base(il);
        l.head_dim_v = hparams.n_head(il);
        l.k_slices = 1;
        l.v_slices = 1;
        l.native_attention = false;
        l.native_attention_owner = nullptr;
        l.mixed_tail_native = false;
        l.native_original_v = false;
        l.native_rotated_max_query_tokens = 0;
        l.k_records = nullptr;
        l.v_records = nullptr;
        l.k_stage = nullptr;
        l.v_stage = nullptr;
        l.k_tail = nullptr;
        l.v_tail = nullptr;
        layers.push_back(std::move(l));
        map_layer_ids[il] = (int32_t)layers.size() - 1;
    }
}

const llama_kv_cache_kvarn::layer & llama_kv_cache_kvarn::layer_for(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        throw std::runtime_error("llama_kv_cache_kvarn::layer_for: unknown layer");
    }
    return layers[it->second];
}

uint32_t llama_kv_cache_kvarn::get_kv_n_stream() const { return n_stream; }
uint32_t llama_kv_cache_kvarn::get_kv_size() const { return kv_size; }
bool llama_kv_cache_kvarn::get_can_shift() const { return false; }
seq_rm_capability llama_kv_cache_kvarn::get_seq_rm_capability() const { return SEQ_RM_CELL; }
llama_kv_cache * llama_kv_cache_kvarn::get_metadata_cache() const { return metadata.get(); }
int32_t llama_kv_cache_kvarn::mapped_layer_id(int32_t il) const {
    auto it = map_layer_ids.find(il);
    return it != map_layer_ids.end() ? it->second : -1;
}
bool llama_kv_cache_kvarn::has_pending_stream_copies() const { return false; }
bool llama_kv_cache_kvarn::stream_is_exclusive_for(llama_seq_id) const { return false; }
bool llama_kv_cache_kvarn::apply_pending_stream_copies(llama_context *) { return false; }
bool llama_kv_cache_kvarn::uses_native_attention(int32_t) const { return false; }
bool llama_kv_cache_kvarn::has_qualified_dflash_mask() const { return false; }
ggml_backend_dev_t llama_kv_cache_kvarn::native_attention_backend(int32_t) const { return nullptr; }
bool llama_kv_cache_kvarn::mixed_tail_native_preferred(int32_t) const { return false; }
bool llama_kv_cache_kvarn::native_attention_uses_original_v(int32_t) const { return false; }
uint32_t llama_kv_cache_kvarn::native_rotated_max_query_tokens(int32_t) const { return 0; }
bool llama_kv_cache_kvarn::requires_state_for_partial_restore() const { return false; }
bool llama_kv_cache_kvarn::state_seq_can_save(llama_seq_id) const { return false; }
bool llama_kv_cache_kvarn::state_seq_can_restore(llama_seq_id) const { return false; }
bool llama_kv_cache_kvarn::state_seq_can_save(llama_seq_id, llama_state_seq_flags) const { return false; }
bool llama_kv_cache_kvarn::state_seq_can_restore(llama_seq_id, llama_state_seq_flags) const { return false; }
ggml_tensor * llama_kv_cache_kvarn::get_materialization_source(int32_t, bool) const { return nullptr; }
std::unique_ptr<llama_kv_cache> llama_kv_cache_kvarn::make_shared_metadata_cache(const llama_model &) const {
    return std::unique_ptr<llama_kv_cache>();
}
llama_kv_tail_route llama_kv_cache_kvarn::get_tail_route(int32_t) const { return LLAMA_KV_TAIL_ROUTE_BODY; }
bool llama_kv_cache_kvarn::get_tail_explicit_bias(int32_t) const { return false; }
void llama_kv_cache_kvarn::reset_kv_tail_planner_timing() {}
uint64_t llama_kv_cache_kvarn::get_kv_tail_planner_timing_ns() const { return 0; }
bool llama_kv_cache_kvarn::get_kv_tail_coverage(uint32_t, llama_seq_id, llama_kv_tail_coverage_info & out) const {
    out.covered = 0;
    out.total = 0;
    return true;
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_batch(llama_batch_allocr &, uint32_t, bool) {
    return nullptr;
}
llama_memory_context_ptr llama_kv_cache_kvarn::init_full() {
    return nullptr;
}
llama_memory_context_ptr llama_kv_cache_kvarn::init_update(llama_context *, bool) {
    return nullptr;
}
llama_memory_context_ptr llama_kv_cache_kvarn::init_kv_batch(const std::vector<llama_ubatch> &) {
    return nullptr;
}
void llama_kv_cache_kvarn::clear(bool data) {
    if (metadata) metadata->clear(data);
}
bool llama_kv_cache_kvarn::can_seq_rm(llama_seq_id, llama_pos, llama_pos) const { return false; }
bool llama_kv_cache_kvarn::seq_rm_plan(llama_seq_id, llama_pos, llama_pos, llama_pos & planned_p0, llama_pos & planned_p1) const {
    planned_p0 = 0;
    planned_p1 = 0;
    return false;
}
bool llama_kv_cache_kvarn::seq_rm(llama_seq_id, llama_pos, llama_pos) { return false; }
bool llama_kv_cache_kvarn::seq_rm_cell(llama_seq_id, uint32_t) { return false; }
int llama_kv_cache_kvarn::cells_at_pos(llama_seq_id, llama_pos, uint32_t *, int) { return 0; }
void llama_kv_cache_kvarn::seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) {}
void llama_kv_cache_kvarn::seq_keep(llama_seq_id) {}
void llama_kv_cache_kvarn::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
    throw std::runtime_error("llama_kv_cache_kvarn::seq_add not implemented");
}
void llama_kv_cache_kvarn::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
    throw std::runtime_error("llama_kv_cache_kvarn::seq_div not implemented");
}
llama_pos llama_kv_cache_kvarn::seq_pos_min(llama_seq_id) const { return 0; }
llama_pos llama_kv_cache_kvarn::seq_pos_max(llama_seq_id) const { return 0; }
std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kvarn::memory_breakdown() const {
    return {};
}
llama_kv_memory_stats llama_kv_cache_kvarn::kv_memory_stats() const {
    llama_kv_memory_stats stats;
    stats.n_used = 0;
    stats.n_free = 0;
    stats.n_full = 0;
    return stats;
}
void llama_kv_cache_kvarn::state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const {}
void llama_kv_cache_kvarn::state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) {}
void llama_kv_cache_kvarn::state_read_sinfo(llama_io_read_i &, llama_seq_id, llama_state_seq_flags, llama_kv_cache::slot_info_vec_t *, const llama_kv_cache::slot_info_vec_t *) {}
bool llama_kv_cache_kvarn::can_remove(llama_seq_id, llama_pos, llama_pos) const { return false; }
void llama_kv_cache_kvarn::copy_kvarn_stream(uint32_t, uint32_t) {}
ggml_tensor * llama_kv_cache_kvarn::store(ggml_context *, ggml_tensor *, ggml_tensor *, int32_t, const llama_kv_cache::slot_info &, bool) const { return nullptr; }
ggml_tensor * llama_kv_cache_kvarn::view(ggml_context *, ggml_tensor *, int32_t, uint32_t, const llama_kv_cache::slot_info &, bool, ggml_tensor *) const { return nullptr; }
ggml_tensor * llama_kv_cache_kvarn::materialize(ggml_context *, ggml_tensor *, int32_t, uint32_t, const llama_kv_cache::slot_info &, bool, ggml_tensor *, bool) const { return nullptr; }
ggml_tensor * llama_kv_cache_kvarn::get_tail(ggml_context *, int32_t, bool) const { return nullptr; }
ggml_tensor * llama_kv_cache_kvarn::store_tail(ggml_context *, ggml_tensor *, ggml_tensor *, int32_t, bool, ggml_tensor *) const { return nullptr; }
