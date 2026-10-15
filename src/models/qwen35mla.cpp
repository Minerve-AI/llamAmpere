#include "models.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

//
// Qwen3.8-27B-MLA: Training-free Multi-head Latent Attention retrofit
//
// Hybrid model: 48 Gated DeltaNet (linear attention) + 16 full-attention MLA layers.
// The MLA layers replace the standard GQA K/V projections with:
//   - kv_a_proj: shared low-rank latent + decoupled RoPE key
//   - kv_b_proj: up-projection from latent to per-head [k_nope | v] (with (1+w) folded)
//
// KV cache per token per layer: [latent (r) | rope (4*58=232)] where r varies per layer.
// The per-head RMS normalization is computed at runtime from the latent.
//

void llama_model_qwen35mla::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Load MLA-specific parameters
    ml.get_key_or_arr(LLM_KV_ATTENTION_KV_LORA_RANK_MLA, hparams.mla_kv_lora_rank_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_ATTENTION_QK_ROPE_HEAD_DIM_MLA,    hparams.mla_qk_rope_head_dim, false);
    ml.get_key_or_arr(LLM_KV_ROPE_KEEP_MLA,              hparams.mla_rope_keep, 32, false);
    if (!ml.get_key(LLM_KV_ROPE_KEEP_COUNT_MLA, hparams.mla_rope_keep_count, false)) {
        hparams.mla_rope_keep_count = 0;
        for (uint32_t i = 0; i < 32 && hparams.mla_rope_keep[i] != 0; ++i) {
            hparams.mla_rope_keep_count++;
        }
    }

    // Set MLA head dimensions for KV cache sizing
    uint32_t max_rank = 0;
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.mla_kv_lora_rank_arr[i] > max_rank) {
            max_rank = hparams.mla_kv_lora_rank_arr[i];
        }
    }
    const uint32_t rope_dim = hparams.mla_qk_rope_head_dim > 0 ? hparams.mla_qk_rope_head_dim : hparams.n_rot();
    hparams.n_embd_head_k_mla_impl = max_rank + rope_dim * hparams.n_head_kv();
    hparams.n_embd_head_v_mla_impl = max_rank;

    // Mark recurrent layers (linear attention layers)
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35mla::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Full-attention MLA layer
            const int64_t rank      = hparams.mla_kv_lora_rank_arr[il];
            const int64_t rope_dim  = hparams.mla_qk_rope_head_dim > 0 ? hparams.mla_qk_rope_head_dim : hparams.n_rot();
            const int64_t n_head    = hparams.n_head();
            const int64_t head_dim  = hparams.n_embd_head_v();
            const int64_t qk_nope   = head_dim - rope_dim;

            GGML_ASSERT(rank > 0 && "MLA layer must have a latent rank");

            // Q projection (same as base model: fused Q + gate)
            create_tensor_qkv(layer, il, n_embd, head_dim * n_head * 2, 0, 0, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { head_dim * n_head, n_embd }, flags);

            // Q/K normalization
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { head_dim }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { head_dim }, flags);

            // MLA: latent + rope projection (fused)
            const int64_t kv_a_out = rank + hparams.n_head_kv() * rope_dim;
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, kv_a_out }, flags);

            // MLA: absorbed K up-projection [rank, qk_nope, n_head]
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { rank, qk_nope, n_head }, flags);

            // MLA: absorbed V up-projection [rank, head_dim, n_head]
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { rank, head_dim, n_head }, flags);

            // MLA: unscaled K up-projection for RMS computation (optional)
            layer.wk_b_raw = create_tensor(tn(LLM_TENSOR_ATTN_K_B_RAW, "weight", il), { rank, qk_nope, n_head },
                                           TENSOR_NOT_REQUIRED | flags);
        } else {
            // Linear attention (gated delta net) - same as qwen35
            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;
            const int64_t conv_dim   = key_dim * 2 + value_dim;

            const ggml_tensor * qkv_meta = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_QKV, "weight", il).str().c_str());
            const bool fused_qkvz = qkv_meta != nullptr && qkv_meta->ne[1] == conv_dim + value_dim;
            layer.wqkv      = create_tensor(tn(LLM_TENSOR_ATTN_QKV,  "weight", il), { n_embd, fused_qkvz ? conv_dim + value_dim : conv_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            GGML_ASSERT(!fused_qkvz || layer.wqkv_gate == nullptr);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,    "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,        "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // FFN
        const ggml_tensor * up_meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_UP, "weight", il).str().c_str());
        const bool fused_gate_up = up_meta != nullptr && up_meta->ne[1] == 2 * n_ff;
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd, n_ff}, fused_gate_up ? (flags | TENSOR_NOT_REQUIRED) : flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, fused_gate_up ? 2 * n_ff : n_ff}, flags);
        GGML_ASSERT(!fused_gate_up || layer.ffn_gate == nullptr);

        // NextN/MTP tensors
        if (il >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
        }
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_trunk(i, mtp_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35mla::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35mla::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn_mla(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

//
// MLA attention layer
//

ggml_tensor * llama_model_qwen35mla::graph::build_layer_attn_mla(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t rank      = hparams.mla_kv_lora_rank_arr[il];
    const int64_t rope_dim  = hparams.mla_qk_rope_head_dim > 0 ? hparams.mla_qk_rope_head_dim : hparams.n_rot();
    const int64_t head_dim  = hparams.n_embd_head_v();
    const int64_t n_head    = hparams.n_head();
    const int64_t n_head_kv = hparams.n_head_kv();
    const int64_t qk_nope   = head_dim - rope_dim;
    const int64_t kv_a_out  = rank + n_head_kv * rope_dim;

    GGML_ASSERT(rank > 0);
    GGML_ASSERT(qk_nope > 0);

    //
    // 1. Q projection (fused Q + gate, same as base model)
    //
    ggml_tensor * Qcur_full = ggml_mul_mat(ctx0, layer.wqkv, cur);
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, head_dim, n_head, n_tokens,
        ggml_element_size(Qcur_full) * head_dim * 2,
        ggml_element_size(Qcur_full) * head_dim * 2 * n_head, 0);
    cb(Qcur, "Qcur", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, head_dim, n_head, n_tokens,
        ggml_element_size(Qcur_full) * head_dim * 2,
        ggml_element_size(Qcur_full) * head_dim * 2 * n_head,
        ggml_element_size(Qcur_full) * head_dim);
    cb(gate, "gate", il);

    // Q normalization
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    //
    // 2. Split Q into q_pe and q_nope (partial rope)
    //
    ggml_tensor * q_pe = ggml_view_3d(ctx0, Qcur, rope_dim, n_head, n_tokens,
        ggml_row_size(Qcur->type, head_dim),
        ggml_row_size(Qcur->type, head_dim) * n_head, 0);
    cb(q_pe, "q_pe", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, Qcur, qk_nope, n_head, n_tokens,
        ggml_row_size(Qcur->type, head_dim),
        ggml_row_size(Qcur->type, head_dim) * n_head,
        ggml_row_size(Qcur->type, rope_dim));
    cb(q_nope, "q_nope", il);

    //
    // 3. Latent + RoPE projection
    //
    ggml_tensor * kv_a = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_a, "kv_a", il);

    ggml_tensor * latent = ggml_view_2d(ctx0, kv_a, rank, n_tokens,
        ggml_row_size(kv_a->type, kv_a_out), 0);
    cb(latent, "latent", il);

    ggml_tensor * k_rope = ggml_view_3d(ctx0, kv_a, rope_dim, n_head_kv, n_tokens,
        ggml_row_size(kv_a->type, kv_a_out),
        ggml_row_size(kv_a->type, kv_a_out) * n_head_kv,
        ggml_row_size(kv_a->type, rank));
    cb(k_rope, "k_rope", il);

    //
    // 4. Apply RoPE (partial: rope_dim dims)
    //
    q_pe = ggml_rope_multi(
            ctx0, q_pe, inp_pos, nullptr,
            rope_dim, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe_rope", il);

    k_rope = ggml_rope_multi(
            ctx0, k_rope, inp_pos, nullptr,
            rope_dim, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_rope, "k_rope_rope", il);

    //
    // 5. Absorbed attention (like DeepSeek MLA)
    //
    // q_nope_absorbed = wk_b @ q_nope
    ggml_tensor * q_nope_perm = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    cb(q_nope_perm, "q_nope_perm", il);

    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope_perm);
    cb(q_nope_absorbed, "q_nope_absorbed_raw", il);

    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "q_nope_absorbed", il);

    //
    // 6. Assemble Q, K, V for build_attn
    //
    // Q_mla = [q_nope_absorbed (rank) | q_pe (rope_dim)] → [rank+rope_dim, n_head, n_tokens]
    ggml_tensor * Q_mla = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
    cb(Q_mla, "Q_mla", il);

    // K_mla = [latent (rank) | k_rope_packed (n_head_kv*rope_dim)] → [rank+n_head_kv*rope_dim, 1, n_tokens]
    ggml_tensor * latent_3d = ggml_reshape_3d(ctx0, latent, rank, 1, n_tokens);
    cb(latent_3d, "latent_3d", il);

    ggml_tensor * k_rope_packed = ggml_reshape_2d(ctx0, k_rope, n_head_kv * rope_dim, n_tokens);
    k_rope_packed = ggml_reshape_3d(ctx0, k_rope_packed, n_head_kv * rope_dim, 1, n_tokens);
    cb(k_rope_packed, "k_rope_packed", il);

    ggml_tensor * K_mla = ggml_concat(ctx0, latent_3d, k_rope_packed, 0);
    cb(K_mla, "K_mla", il);

    // V_mla = latent (shared)
    ggml_tensor * V_mla = latent_3d;
    cb(V_mla, "V_mla", il);

    //
    // 7. Attention computation with absorbed V
    //
    const float kq_scale = 1.0f / sqrtf(float(head_dim));

    ggml_tensor * attn_out = build_attn(inp,
                layer.wo, nullptr, layer.wo_s,
                Q_mla, K_mla, V_mla, nullptr, nullptr, layer.wv_b, kq_scale, il);
    cb(attn_out, "attn_out", il);

    //
    // 8. Apply gate (sigmoid)
    //
    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    ggml_tensor * result = ggml_mul(ctx0, attn_out, gate_sigmoid);
    cb(result, "attn_gated", il);

    return result;
}

//
// Linear attention layer (Gated Delta Net) - same as qwen35
//

ggml_tensor * llama_model_qwen35mla::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    // TODO: Full implementation should match qwen35.cpp build_layer_attn_linear.
    // The Gated Delta Net linear attention is identical to the base Qwen3.5 model.
    // For now, this is a placeholder that will be completed in a follow-up commit.
    GGML_ABORT("Qwen3.8-MLA linear attention not yet implemented - copy from qwen35.cpp");
    return cur;
}

//
// FFN layer - same as qwen35
//

ggml_tensor * llama_model_qwen35mla::graph::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];

    cur = build_ffn(cur,
            layer.ffn_up, nullptr, nullptr,
            layer.ffn_gate, nullptr, nullptr,
            layer.ffn_down, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);

    return cur;
}

ggml_tensor * llama_model_qwen35mla::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normed = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    normed = ggml_mul(ctx0, normed, gate);
    return normed;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35mla::graph::build_qkvz(
        ggml_tensor * input,
        int           il) {
    // TODO: Full implementation should match qwen35.cpp build_qkvz.
    GGML_ABORT("Qwen3.8-MLA build_qkvz not yet implemented - copy from qwen35.cpp");
    return {nullptr, nullptr};
}

//
// MTP graph - placeholder
//

llama_model_qwen35mla::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    throw std::runtime_error("Qwen3.8-MLA MTP graph not yet implemented");
}
