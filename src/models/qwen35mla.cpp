#include "models.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

//
// Qwen3.8-27B-MLA: Training-free Multi-head Latent Attention retrofit
//
// Hybrid model: 48 Gated DeltaNet (linear attention) + 16 full-attention MLA layers.
// The MLA layers replace the standard GQA K/V projections with:
//   - wkv_a_mqa: shared low-rank latent + per-KV-head decoupled RoPE key
//   - wk_b: up-projection from latent to per-head K_nope (absorbed into Q)
//   - wv_b: up-projection from latent to per-head V
//
// Implementation: decompressed MHA (24 heads for K and V).
// The absorbed-attention optimization (MQA with latent cache) is a future TODO.
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

    // For the decompressed MHA implementation, we use n_head heads for K and V.
    // Override n_head_kv to n_head so the KV cache is sized correctly.
    // TODO: implement absorbed attention (MQA with latent cache) for memory efficiency.
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (!hparams.is_recr(i)) {
            hparams.n_head_kv_arr[i] = hparams.n_head();
        }
    }

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
            const int64_t rank     = hparams.mla_kv_lora_rank_arr[il];
            const int64_t rope_dim = hparams.n_rot(); // 64 (full rope dim for cache)
            const int64_t n_head   = hparams.n_head();
            const int64_t head_dim = hparams.n_embd_head_v();
            const int64_t qk_nope  = head_dim - hparams.n_rot();

            GGML_ASSERT(rank > 0 && "MLA layer must have a latent rank");

            // Q projection (fused Q + gate)
            create_tensor_qkv(layer, il, n_embd, head_dim * n_head * 2, 0, 0, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { head_dim * n_head, n_embd }, flags);

            // Q/K normalization
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { head_dim }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { head_dim }, flags);

            // MLA: latent projection [n_embd, rank]
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, rank }, flags);

            // MLA: per-KV-head rope key projection [n_embd, n_head_kv * rope_dim]
            const int64_t n_head_kv_base = 4; // base model KV heads
            layer.wk_rope = create_tensor(tn(LLM_TENSOR_ATTN_K_ROPE_PROJ, "weight", il), { n_embd, n_head_kv_base * rope_dim }, flags);

            // MLA: K up-projection [rank, qk_nope, n_head]
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { rank, qk_nope, n_head }, flags);

            // MLA: V up-projection [rank, head_dim, n_head]
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

        // Parallel FFN
        if (hparams.n_ff_par > 0) {
            const int64_t n_ff_par = hparams.n_ff_par;
            layer.ffn_gate_par = create_tensor(tn(LLM_TENSOR_FFN_GATE_PAR, "weight", il), {n_embd,   n_ff_par}, flags);
            layer.ffn_down_par = create_tensor(tn(LLM_TENSOR_FFN_DOWN_PAR, "weight", il), {n_ff_par, n_embd}, flags);
            layer.ffn_up_par   = create_tensor(tn(LLM_TENSOR_FFN_UP_PAR,   "weight", il), {n_embd,   n_ff_par}, flags);
        }

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
// MLA attention layer (decompressed MHA)
//
// Computes K and V explicitly from the latent:
//   K_nope[i,h,t] = sum_j wk_b[j,i,h] * latent[j,t]   (per Q-head)
//   V[i,h,t]      = sum_j wv_b[j,i,h] * latent[j,t]   (per Q-head)
//   K = [K_nope | k_rope_expanded]  (256, 24, n_tokens)
//   V = wv_b @ latent                (256, 24, n_tokens)
//
// The rope is computed per-KV-head (4 heads) and expanded to 24 Q heads via GQA repeat.
//

ggml_tensor * llama_model_qwen35mla::graph::build_layer_attn_mla(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t rank      = hparams.mla_kv_lora_rank_arr[il];
    const int64_t head_dim  = hparams.n_embd_head_v();   // 256
    const int64_t n_head    = hparams.n_head();           // 24
    const int64_t n_rot     = hparams.n_rot();            // 64
    const int64_t qk_nope   = head_dim - n_rot;           // 192
    const int64_t n_head_kv_base = 4;                     // base model KV heads

    GGML_ASSERT(rank > 0);

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
    // 2. Split Q into q_nope and q_pe, apply RoPE
    //
    ggml_tensor * q_pe = ggml_view_3d(ctx0, Qcur, n_rot, n_head, n_tokens,
        ggml_row_size(Qcur->type, head_dim),
        ggml_row_size(Qcur->type, head_dim) * n_head, 0);
    cb(q_pe, "q_pe", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, Qcur, qk_nope, n_head, n_tokens,
        ggml_row_size(Qcur->type, head_dim),
        ggml_row_size(Qcur->type, head_dim) * n_head,
        ggml_row_size(Qcur->type, n_rot));
    cb(q_nope, "q_nope", il);

    // Apply RoPE to q_pe (full 64 dims)
    q_pe = ggml_rope_multi(
            ctx0, q_pe, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe_rope", il);

    //
    // 3. Latent + RoPE projection (separate tensors)
    //
    ggml_tensor * latent = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(latent, "latent", il);

    ggml_tensor * k_rope_2d = ggml_mul_mat(ctx0, layer.wk_rope, cur);
    cb(k_rope_2d, "k_rope_2d", il);

    ggml_tensor * k_rope = ggml_reshape_3d(ctx0, k_rope_2d, n_rot, n_head_kv_base, n_tokens);
    cb(k_rope, "k_rope_raw", il);

    // Apply RoPE to k_rope (full 64 dims)
    k_rope = ggml_rope_multi(
            ctx0, k_rope, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_rope, "k_rope_rope", il);

    //
    // 4. Compute K_nope = wk_b^T @ latent  (per Q-head)
    //    wk_b: [rank, qk_nope, n_head]
    //    Result: [qk_nope, n_head, n_tokens]
    //
    {
        // Reshape wk_b to [rank, qk_nope*n_head], transpose to [qk_nope*n_head, rank]
        ggml_tensor * wk_b_2d = ggml_reshape_2d(ctx0, layer.wk_b, rank, qk_nope * n_head);
        ggml_tensor * K_nope_2d = ggml_mul_mat(ctx0, ggml_transpose(ctx0, wk_b_2d), latent);
        // [qk_nope*n_head, n_tokens] → [qk_nope, n_head, n_tokens]
        ggml_tensor * K_nope = ggml_reshape_3d(ctx0, K_nope_2d, qk_nope, n_head, n_tokens);
        cb(K_nope, "K_nope", il);

        //
        // 5. Expand k_rope from 4 KV heads to 24 Q heads (GQA repeat)
        //    k_rope: [64, 4, n_tokens] → [64, 24, n_tokens]
        //    Head i uses KV head i/6
        //
        ggml_tensor * k_rope_4d = ggml_reshape_4d(ctx0, k_rope, n_rot, n_head_kv_base, 1, n_tokens);
        ggml_tensor * k_rope_rep = ggml_repeat_4d(ctx0, k_rope_4d, n_rot, n_head_kv_base, n_head / n_head_kv_base, n_tokens);
        ggml_tensor * k_rope_24 = ggml_reshape_3d(ctx0, k_rope_rep, n_rot, n_head, n_tokens);
        cb(k_rope_24, "k_rope_24", il);

        //
        // 6. Assemble K = [K_nope | k_rope_24] → [head_dim, n_head, n_tokens]
        //
        ggml_tensor * K = ggml_concat(ctx0, K_nope, k_rope_24, 0);
        cb(K, "K", il);

        //
        // 7. Compute V = wv_b^T @ latent  (per Q-head)
        //    wv_b: [rank, head_dim, n_head]
        //    Result: [head_dim, n_head, n_tokens]
        //
        ggml_tensor * wv_b_2d = ggml_reshape_2d(ctx0, layer.wv_b, rank, head_dim * n_head);
        ggml_tensor * V_2d = ggml_mul_mat(ctx0, ggml_transpose(ctx0, wv_b_2d), latent);
        ggml_tensor * V = ggml_reshape_3d(ctx0, V_2d, head_dim, n_head, n_tokens);
        cb(V, "V", il);

        //
        // 8. Reassemble Q = [q_nope | q_pe_rope] → [head_dim, n_head, n_tokens]
        //
        ggml_tensor * Q = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(Q, "Q", il);

        //
        // 9. MHA attention
        //
        const float kq_scale = 1.0f / sqrtf(float(head_dim));

        ggml_tensor * attn_out = build_attn(inp,
                    layer.wo, nullptr, layer.wo_s,
                    Q, K, V, nullptr, nullptr, nullptr, kq_scale, il);
        cb(attn_out, "attn_out", il);

        //
        // 10. Apply gate (sigmoid)
        //
        ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
        cb(gate_sigmoid, "gate_sigmoid", il);

        ggml_tensor * result = ggml_mul(ctx0, attn_out, gate_sigmoid);
        cb(result, "attn_gated", il);

        return result;
    }
}

//
// Linear attention layer (Gated Delta Net) - identical to qwen35
//

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

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
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

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
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

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

//
// FFN layer - same as qwen35
//

ggml_tensor * llama_model_qwen35mla::graph::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    if (layer.ffn_gate == nullptr) {
        // EXL3 fused layout: ffn_up = [gate | up]
        GGML_ASSERT(layer.ffn_up_s == nullptr && layer.ffn_gate_s == nullptr);
        cur = build_ffn(cur,
            layer.ffn_up,   nullptr, nullptr,
            nullptr,        nullptr, nullptr,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
    } else {
        cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    // Run the parallel FFN on the same input as the main FFN.
    if (model.layers[il].ffn_up_par != nullptr) {
        ggml_tensor * par = build_ffn(cur,
            model.layers[il].ffn_up_par, NULL, NULL,
            model.layers[il].ffn_gate_par, NULL, NULL,
            model.layers[il].ffn_down_par, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(par, "ffn_par_out", il);
        cur = ggml_add(ctx0, cur, par);
    }

    cb(cur, "ffn_out", il);
    return cur;
}

ggml_tensor * llama_model_qwen35mla::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);
    return ggml_mul(ctx0, normalized, gated_silu);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35mla::graph::build_qkvz(
        ggml_tensor * input,
        int           il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv_gate == nullptr) {
        // EXL3 fused [in_proj_qkv | in_proj_z]: one matmul, split by views
        const int64_t n_tok     = n_seq_tokens * n_seqs;
        const int64_t value_dim = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t conv_dim  = model.layers[il].wqkv->ne[1] - value_dim;
        GGML_ASSERT(model.layers[il].wqkv_s == nullptr && model.layers[il].wqkv_gate_s == nullptr);

        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv, input, nullptr);
        cb(qkvz, "linear_attn_qkvz", il);
        const size_t es = ggml_element_size(qkvz);

        ggml_tensor * qkv_mixed = ggml_view_2d(ctx0, qkvz, conv_dim, n_tok, qkvz->nb[1], 0);
        qkv_mixed = ggml_cont_3d(ctx0, qkv_mixed, conv_dim, n_seq_tokens, n_seqs);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, value_dim, n_tok, qkvz->nb[1], conv_dim * es);
        z = ggml_cont(ctx0, z);
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

//
// MTP graph - placeholder
//

llama_model_qwen35mla::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    throw std::runtime_error("Qwen3.8-MLA MTP graph not yet implemented");
}
