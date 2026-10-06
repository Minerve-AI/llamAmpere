#include "models.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

void llama_model_qwen35mla::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Load MLA parameters
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,    hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,  hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);

    // Load per-layer latent rank array (for non-uniform MLA ranks)
    ml.get_arr_n(LLM_KV_ATTENTION_LATENT_RANK_PER_LAYER, hparams.n_lora_kv_per_layer, 64);
    for (uint32_t i = 0; i < 64; ++i) {
        if (hparams.n_lora_kv_per_layer[i] > 0) {
            hparams.n_lora_kv_per_layer_count = i + 1;
        }
    }

    // Load total rope dim in fused wkv_a_mqa (for per_kv_head RoPE mode)
    ml.get_key(LLM_KV_ATTENTION_MLA_FUSED_ROPE_DIM, hparams.n_mla_fused_rope_dim, false);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
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

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // MLA dimensions
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    // Count full-attention layers for per-layer rank tracking
    int full_attn_count = 0;
    for (int i = 0; i < n_layer; ++i) {
        if (!hparams.is_recr(i)) full_attn_count++;
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters (for DeltaNet layers)
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // === MLA Attention layers ===
            // Get per-layer latent rank
            int fa_idx = 0;
            for (int i = 0; i < il; ++i) {
                if (!hparams.is_recr(i)) fa_idx++;
            }
            const int64_t lora_rank = hparams.n_lora_kv_at(fa_idx);

            // Q projection (with gate: outputs n_head * head_dim * 2)
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, n_head * n_embd_head_k_mla * 2 }, flags);

            // KV compression: fused latent + rope projection
            const int64_t fused_rope_dim = hparams.n_mla_fused_rope_dim > 0 ? hparams.n_mla_fused_rope_dim : n_embd_head_qk_rope;
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, lora_rank + fused_rope_dim }, flags);

            // K and V decompression (B matrices)
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { n_embd_head_qk_nope, lora_rank, n_head }, flags);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { lora_rank, n_embd_head_v_mla, n_head }, flags);

            // Output projection
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head * n_embd_head_v_mla, n_embd }, flags);

            // Q/K normalization
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k_mla }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k_mla }, flags);
        } else {
            // === Linear attention (gated delta net) layers ===
            const ggml_tensor * qkv_meta = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_QKV, "weight", il).str().c_str());
            const bool fused_qkvz = qkv_meta != nullptr && qkv_meta->ne[1] == conv_dim + value_dim;
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, fused_qkvz ? conv_dim + value_dim : conv_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            GGML_ASSERT(!fused_qkvz || layer.wqkv_gate == nullptr);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // FFN
        const ggml_tensor * up_meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_UP, "weight", il).str().c_str());
        const bool fused_gate_up = up_meta != nullptr && up_meta->ne[1] == 2 * n_ff;
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, fused_gate_up ? (flags | TENSOR_NOT_REQUIRED) : flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    for (int il = 0; il < n_layer; ++il) {
        load_block_trunk(il, trunk_flags);
    }

    // MTP layers
    if (hparams.n_layer_nextn > 0) {
        for (int il = n_layer; il < n_layer_all; ++il) {
            auto & layer = layers[il];

            layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
            layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

            // MTP uses standard GQA (not MLA)
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k_mla * n_head * 2, n_embd_head_k_mla * n_head_kv, n_embd_head_v_mla * n_head_kv, mtp_flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_v_mla * n_head, n_embd }, mtp_flags);
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k_mla }, mtp_flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k_mla }, mtp_flags);

            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd, n_ff}, mtp_flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd}, mtp_flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, n_ff}, mtp_flags);

            // MTP specific
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", il), { n_embd }, mtp_flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", il), { n_embd }, mtp_flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", il), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | mtp_flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | mtp_flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd }, TENSOR_NOT_REQUIRED | mtp_flags);
        }
    }
}

//
// graph
//

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

    // Use K-only hybrid memory input (MLA stores only K in KV cache)
    auto * inp = build_inp_mem_hybrid_k();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // MLA full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // FFN
        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "ffn_residual", il);

        inpL = cur;
    }

    // Final norm
    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "model.output_norm", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    res->t_h = cur;
    cb(cur, "model.h", -1);

    ggml_build_forward_expand_gf(gf);
}

ggml_tensor * llama_model_qwen35mla::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    // Get per-layer latent rank
    int fa_idx = 0;
    for (int i = 0; i < il; ++i) {
        if (!hparams.is_recr(i)) fa_idx++;
    }
    const int64_t lora_rank = hparams.n_lora_kv_at(fa_idx);

    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f / sqrtf(float(n_embd_head_k_mla))
        : hparams.f_attention_scale;

    // Step 1: Q projection (with gate: outputs n_head * head_dim * 2)
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    cb(Qcur_full, "Qcur_full", il);

    // Split Q and gate
    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head_k_mla, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2 * n_head, 0);
    cb(Qcur, "Qcur", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head_k_mla, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head_k_mla * n_head, n_tokens);
    cb(gate, "gate", il);

    // Step 2: Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Step 3: KV compression (latent + rope)
    ggml_tensor * kv_cmpr_pe = build_lora_mm(model.layers[il].wkv_a_mqa, cur, model.layers[il].wkv_a_mqa_s);
    cb(kv_cmpr_pe, "kv_cmpr_pe", il);

    // Split: latent and rope key
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, lora_rank, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, lora_rank + n_embd_head_qk_rope), 0);
    cb(kv_cmpr, "kv_cmpr", il);

    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, lora_rank));
    cb(k_pe, "k_pe", il);

    // Step 4: K normalization
    // k_norm (256,) targets the full decompressed K head (qk_nope + qk_rope = 198 + 58).
    // In the absorbed path, K is never explicitly decompressed — the nope part is
    // absorbed into Q via W_k_b. Applying k_norm here is not straightforward.
    // TODO: implement proper k_norm for absorbed MLA (may need to apply to Q post-absorption)
    cb(k_pe, "k_pe", il);

    // Step 5: Split Q into nope and pe
    ggml_tensor * q_nope = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens,
        ggml_row_size(Qcur->type, n_embd_head_k_mla),
        ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);
    cb(q_nope, "q_nope", il);

    ggml_tensor * q_pe = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens,
        ggml_row_size(Qcur->type, n_embd_head_k_mla),
        ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head,
        ggml_row_size(Qcur->type, n_embd_head_qk_nope));
    cb(q_pe, "q_pe", il);

    // Step 6: Apply RoPE to q_pe and k_pe
    q_pe = ggml_rope_multi(ctx0, q_pe, inp_pos, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe_rope", il);

    k_pe = ggml_rope_multi(ctx0, k_pe, inp_pos, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "k_pe_rope", il);

    // Step 7: K absorption into Q (the MLA trick)
    // q_nope: (n_embd_head_qk_nope, n_head, n_tokens)
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    cb(q_nope, "q_nope_perm", il);

    // wk_b: (n_embd_head_qk_nope, lora_rank, n_head)
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
    cb(q_nope_absorbed, "q_nope_absorbed", il);

    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

    // Step 8: Assemble final Q, K, V
    // Q: (lora_rank + n_embd_head_qk_rope, n_head, n_tokens)
    // Note: latent part first, rope part second (for in-place context shifting)
    ggml_tensor * Qfinal = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
    cb(Qfinal, "Qfinal", il);

    // K: (lora_rank + n_embd_head_qk_rope, 1, n_tokens) [MQA: single head]
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr_3d", il);

    ggml_tensor * Kfinal = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    cb(Kfinal, "Kfinal", il);

    // V: (lora_rank, 1, n_tokens) [decompressed via wv_b in attention kernel]
    ggml_tensor * Vcur = kv_cmpr;
    cb(Vcur, "Vcur", il);

    // Step 9: Attention (with V decompression via wv_b)
    cur = build_attn(inp_attn,
                model.layers[il].wo, nullptr, model.layers[il].wo_s,
                Qfinal, Kfinal, Vcur, nullptr, nullptr,
                model.layers[il].wv_b, kq_scale, il);
    cb(cur, "attn_out", il);

    // Step 10: Gate
    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    return cur;
}

ggml_tensor * llama_model_qwen35mla::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // Delta rule update
    ggml_tensor * v_new = ggml_sub(ctx0, v_conv, state);
    cb(v_new, "v_new", il);

    v_new = ggml_mul(ctx0, v_new, gate);
    cb(v_new, "v_new_gated", il);

    ggml_tensor * k_dot_state = ggml_mul_mat(ctx0, k_conv, state);
    cb(k_dot_state, "k_dot_state", il);

    ggml_tensor * delta = ggml_mul(ctx0, v_new, k_dot_state);
    cb(delta, "delta", il);

    // Update state
    ggml_tensor * state_new = ggml_add(ctx0, state, ggml_mul_mat(ctx0, k_conv, v_new));
    cb(state_new, "state_new", il);

    // Store state
    inp->set_s_l(il, state_new, n_seqs);

    // Output: q · state
    ggml_tensor * out = ggml_mul_mat(ctx0, q_conv, state);
    cb(out, "gdn_out", il);

    // Gate the output
    out = ggml_mul(ctx0, out, z);
    cb(out, "gdn_gated", il);

    // Reshape and project
    out = ggml_reshape_3d(ctx0, out, d_inner, 1, n_tokens);
    out = ggml_cont_2d(ctx0, out, d_inner, n_tokens);
    cur = build_lora_mm(model.layers[il].ssm_out, out, model.layers[il].ssm_out_s);
    cb(cur, "gdn_final", il);

    return cur;
}

ggml_tensor * llama_model_qwen35mla::graph::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    return build_ffn_swiglu(ctx0, model.layers[il].ffn_up, model.layers[il].ffn_gate, model.layers[il].ffn_down, cur);
}

ggml_tensor * llama_model_qwen35mla::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * result = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    if (gate) {
        result = ggml_mul(ctx0, result, gate);
    }
    return result;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35mla::graph::build_qkvz(
        ggml_tensor * input,
        int           il) {
    auto & layer = model.layers[il];

    if (layer.wqkv) {
        // Fused QKVZ
        ggml_tensor * qkvz = build_lora_mm(layer.wqkv, input, layer.wqkv_s);
        const int64_t conv_dim = hparams.ssm_d_state * hparams.ssm_n_group * 2 + hparams.ssm_d_inner;
        const int64_t value_dim = hparams.ssm_d_inner;

        ggml_tensor * qkv = ggml_view_2d(ctx0, qkvz, conv_dim, n_tokens,
            ggml_row_size(qkvz->type, conv_dim + value_dim), 0);
        ggml_tensor * z   = ggml_view_2d(ctx0, qkvz, value_dim, n_tokens,
            ggml_row_size(qkvz->type, conv_dim + value_dim),
            ggml_row_size(qkvz->type, conv_dim));
        return {qkv, z};
    } else {
        // Separate QKV and gate
        ggml_tensor * qkv = build_lora_mm(layer.wqkv_gate, input, layer.wqkv_gate_s);
        ggml_tensor * z   = qkv;
        return {qkv, z};
    }
}

//
// MTP graph
//

llama_model_qwen35mla::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(model, params) {
    // MTP graph implementation follows the same pattern as qwen35 graph_mtp
    // Using standard GQA attention (not MLA) for the MTP layer
    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * h_embd;
    ggml_tensor * tok_embd;

    h_embd   = build_inp_embd(model.tok_embd);
    tok_embd = build_inp_tok_embd();

    cb(h_embd, "model.h_embd", -1);

    auto * inp = build_inp_mem_hybrid();
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int il = n_layer; // MTP layer index

    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    // MTP: combine embedding and hidden state
    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // Standard GQA attention for MTP
    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head_k_mla * 2, n_head,
            n_embd_head_k_mla,     n_head_kv,
            n_embd_head_v_mla,     n_head_kv,
            il, false);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head_k_mla, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2 * n_head, 0);

    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k_mla, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head_k_mla, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head_k_mla);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head_k_mla * n_head, n_tokens);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v_mla, n_head_kv, n_tokens);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k_mla));

    cur = build_attn(inp->get_attn(), layer.wo, nullptr, layer.wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cur = ggml_mul(ctx0, cur, gate_sigmoid);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_out", il);

    ggml_tensor * ffn_inp = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_ffn_swiglu(ctx0, layer.ffn_up, layer.ffn_gate, layer.ffn_down, cur);
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_ffn_out", il);

    // MTP output norm
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
        ? layer.nextn.shared_head_norm
        : model.output_norm;
    GGML_ASSERT(head_norm_w);

    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    res->t_h_nextn = cur;
    cb(cur, "mtp_h_nextn", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    ggml_tensor * head_w = layer.nextn.shared_head_head
        ? layer.nextn.shared_head_head
        : model.output;
    GGML_ASSERT(head_w);

    cur = build_lora_mm(head_w, cur, nullptr);
    res->t_logits = cur;
    cb(cur, "mtp_logits", -1);

    ggml_build_forward_expand_gf(gf);
}
