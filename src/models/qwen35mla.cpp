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

    // Absorbed attention: KV cache stores latent + k_rope (MQA, single "head").
    // K cache dim = rank + rope_dim * n_head_kv_base (packed latent + rope key)
    // V is computed from the latent at attention time via wv_b (not stored).
    {
        const uint32_t n_kv_base = 4; // base model KV heads for rope
        const uint32_t rope_dim  = hparams.mla_qk_rope_head_dim > 0 ? hparams.mla_qk_rope_head_dim : 64;
        // Use the max rank across all MLA layers for uniform cache sizing
        uint32_t max_rank = 0;
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            if (!hparams.is_recr(i) && hparams.mla_kv_lora_rank_arr[i] > max_rank) {
                max_rank = hparams.mla_kv_lora_rank_arr[i];
            }
        }
        hparams.n_embd_head_k_mla_impl = max_rank + rope_dim * n_kv_base;
        hparams.n_embd_head_v_mla_impl = max_rank; // V dim (latent), not actually stored

        // KV cache uses single "head" (MQA): the latent is shared across all Q heads
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            if (!hparams.is_recr(i)) {
                hparams.n_head_kv_arr[i] = 1;
            }
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

        if (il >= n_layer) {
            // MTP layer: fused Q+gate stored as attn_q (not attn_qkv)
            const int64_t n_head   = hparams.n_head();
            const int64_t head_dim = hparams.n_embd_head_v();

            // Fused Q+gate: [n_embd, 2 * n_head * head_dim]
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, head_dim * n_head * 2 }, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head * head_dim, n_embd }, flags);

            // MTP: embedding hidden projection
            create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { n_embd, n_embd }, flags);

            // Q/K normalization
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { head_dim }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { head_dim }, flags);
        } else if (!hparams.is_recr(il)) {
            // Full-attention MLA layer
            const int64_t rank     = hparams.mla_kv_lora_rank_arr[il];
            const int64_t rope_dim = hparams.mla_qk_rope_head_dim > 0 ? hparams.mla_qk_rope_head_dim : hparams.n_rot();
            const int64_t n_head   = hparams.n_head();
            const int64_t head_dim = hparams.n_embd_head_v();
            const int64_t qk_nope  = head_dim - rope_dim;

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

            // MLA: K up-projection [rank, qk_nope * n_head] (stored flat, reshaped in compute)
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { rank, qk_nope * n_head }, flags);

            // MLA: V up-projection [rank, head_dim * n_head] (stored flat, reshaped in compute)
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { rank, head_dim * n_head }, flags);

            // MLA: unscaled K up-projection for RMS computation (optional)
            layer.wk_b_raw = create_tensor(tn(LLM_TENSOR_ATTN_K_B_RAW, "weight", il), { rank, qk_nope * n_head },
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
// MLA attention layer (absorbed attention with latent KV cache)
//
// Architecture (inspired by DeepSeek MLA, adapted for per-head wk_b/wv_b):
//
//   KV cache stores: [latent | k_rope] packed → [rank + rope_dim*n_kv_base, 1, n_cache]
//   V is NOT stored — computed from latent at attention time via wv_b
//
//   Q absorption: Q_nope is projected through wk_b^T (per-head) so that the
//   attention dot product Q·K operates on the latent directly.
//
//   scores = [Q_absorbed · K_latent^T  +  q_pe · k_rope^T] * scale
//   attn_out = scores · V  where V = wv_b^T @ latent (per-head)
//
//   Memory: (rank + rope_dim*4) per token per layer  vs  2*head_dim*n_head for MHA
//   Example: (512+256)=768  vs  2*256*24=12288  →  16x reduction
//

ggml_tensor * llama_model_qwen35mla::graph::build_layer_attn_mla(
        llm_graph_input_attn_k * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t rank      = hparams.mla_kv_lora_rank_arr[il];
    const int64_t head_dim  = hparams.n_embd_head_v();   // 256
    const int64_t n_head    = hparams.n_head();           // 24
    const int64_t n_rot     = hparams.mla_qk_rope_head_dim > 0 ? hparams.mla_qk_rope_head_dim : hparams.n_rot();
    const int64_t qk_nope   = head_dim - n_rot;
    const int64_t n_kv_base = 4;                          // base model KV heads for rope

    GGML_ASSERT(rank > 0);

    //
    // 1. Q projection (fused Q + gate)
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

    // Q normalization (per-head RMS over full head_dim)
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    //
    // 2. Split Q into q_nope and q_pe
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

    //
    // 3. Q absorption: Q_absorbed[d,h,t] = sum_p W_UK[h][d,p] * q_nope[p,h,t]
    //    wk_b GGUF: [rank, N*P] where W[d, h*P+p] = W_UK[h][d,p]
    //    Result: [rank, n_head, n_tokens]
    //
    {
        // Reshape wk_b: [r, N*P] → [r, P, N]  (W_reshaped[d,p,h] = W[d, h*P+p])
        ggml_tensor * wk_b_3d = ggml_reshape_3d(ctx0, layer.wk_b, rank, qk_nope, n_head);
        // Permute: [r, P, N] → [P, r, N]  (A[p,d,h] = W_UK[h][d,p])
        ggml_tensor * wk_b_t = ggml_permute(ctx0, wk_b_3d, 1, 0, 2, 3);
        // Permute q_nope: [P, N, n_tokens] → [P, n_tokens, N]  (B[p,t,h] = q_nope[p,h,t])
        ggml_tensor * q_nope_p = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        // Batched matmul: result[d,t,h] = sum_p A[p,d,h]*B[p,t,h] → [r, n_tokens, N]
        ggml_tensor * Q_abs_3d = ggml_mul_mat(ctx0, wk_b_t, q_nope_p);
        // Permute: [r, n_tokens, N] → [r, N, n_tokens]
        ggml_tensor * Q_absorbed = ggml_permute(ctx0, Q_abs_3d, 0, 2, 1, 3);
        cb(Q_absorbed, "Q_absorbed", il);

        //
        // 4. Latent + k_rope projection
        //
        ggml_tensor * latent = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        cb(latent, "latent", il);

        ggml_tensor * k_rope_2d = ggml_mul_mat(ctx0, layer.wk_rope, cur);
        cb(k_rope_2d, "k_rope_2d", il);

        ggml_tensor * k_rope = ggml_reshape_3d(ctx0, k_rope_2d, n_rot, n_kv_base, n_tokens);
        cb(k_rope, "k_rope_raw", il);

        // Apply RoPE to q_pe and k_rope
        q_pe = ggml_rope_multi(
                ctx0, q_pe, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "q_pe_rope", il);

        k_rope = ggml_rope_multi(
                ctx0, k_rope, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(k_rope, "k_rope_rope", il);

        //
        // 5. Pack K cache: [latent | k_rope] → [rank + n_rot*n_kv_base, 1, n_tokens]
        //    The KV cache stores this packed tensor (MQA, single "head")
        //
        // Reshape latent to 3D: [rank, 1, n_tokens]
        ggml_tensor * latent_3d = ggml_reshape_3d(ctx0, latent, rank, 1, n_tokens);
        // Reshape k_rope to 2D: [n_rot*n_kv_base, n_tokens]
        ggml_tensor * k_rope_2d_packed = ggml_reshape_2d(ctx0, k_rope, n_rot * n_kv_base, n_tokens);
        // Concatenate: [rank + n_rot*n_kv_base, n_tokens]
        ggml_tensor * K_packed_2d = ggml_concat(ctx0, latent_3d, k_rope_2d_packed, 0);
        // Reshape to [rank + n_rot*n_kv_base, 1, n_tokens] for KV cache
        ggml_tensor * K_packed = ggml_reshape_3d(ctx0, K_packed_2d, rank + n_rot * n_kv_base, 1, n_tokens);
        cb(K_packed, "K_packed", il);

        //
        // 6. Assemble Q = [Q_absorbed | q_pe_rope] → [rank + n_rot, n_head, n_tokens]
        //
        ggml_tensor * Q_full = ggml_concat(ctx0, Q_absorbed, q_pe, 0);
        cb(Q_full, "Q_full", il);

        //
        // 7. Attention with MLA (V computed from latent via wv_b)
        //    V is a view of K_packed (latent part), v_mla decompresses MQA→MHA
        //
        const float kq_scale = 1.0f / sqrtf(float(head_dim));

        // V = latent part of K_packed: [rank, 1, n_tokens]
        ggml_tensor * V_latent = ggml_view_3d(ctx0, K_packed, rank, 1, n_tokens,
            K_packed->nb[1], K_packed->nb[2], 0);
        cb(V_latent, "V_latent", il);

        // Reshape wv_b: [rank, head_dim*n_head] → [rank, head_dim, n_head] for v_mla
        ggml_tensor * wv_b_3d = ggml_reshape_3d(ctx0, layer.wv_b, rank, head_dim, n_head);

        ggml_tensor * attn_out = build_attn(inp,
                    layer.wo, nullptr, layer.wo_s,
                    Q_full, K_packed, V_latent, nullptr, nullptr, wv_b_3d, kq_scale, il);
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
// FFN layer - free function (shared between graph and graph_mtp)
//
static ggml_tensor * qwen35mla_build_ffn(llm_graph_context & g, const llama_layer & layer, ggml_tensor * cur, int il) {
    if (layer.ffn_gate == nullptr) {
        // EXL3 fused layout: ffn_up = [gate | up]
        GGML_ASSERT(layer.ffn_up_s == nullptr && layer.ffn_gate_s == nullptr);
        cur = g.build_ffn(cur,
            layer.ffn_up,   nullptr, nullptr,
            nullptr,        nullptr, nullptr,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
    } else {
        cur = g.build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    // Run the parallel FFN on the same input as the main FFN.
    if (layer.ffn_up_par != nullptr) {
        ggml_tensor * par = g.build_ffn(cur,
            layer.ffn_up_par, NULL, NULL,
            layer.ffn_gate_par, NULL, NULL,
            layer.ffn_down_par, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        g.cb(par, "ffn_par_out", il);

        cur = ggml_add(g.ctx0, cur, par);
    }

    g.cb(cur, "ffn_out", il);
    return cur;
}

ggml_tensor * llama_model_qwen35mla::graph::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    return qwen35mla_build_ffn(*this, model.layers[il], cur, il);
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
// MTP graph for Qwen3.8-MLA
// The MTP layer uses standard GQA attention (separate Q, K, V projections).
//

llama_model_qwen35mla::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35_MLA MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35_MLA MTP currently only supports a single MTP block");

    const int64_t head_dim = hparams.n_embd_head_v();
    const int64_t n_head   = hparams.n_head();
    const int64_t n_head_kv = hparams.n_head_kv();

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    // MTP-specific tensors are optional - if absent, use hidden state directly
    const bool has_mtp_proj = layer.nextn.eh_proj != nullptr;

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
        tok_embd = build_hadamard_inverse(tok_embd_w, tok_embd);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_k();

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(head_dim)) : hparams.f_attention_scale;

    // --- Chained drafting path ---
    if (params.cparams.mtp_chain && ubatch.n_seqs_unq == 1 && ubatch.token) {
        const auto * mctx_kv = static_cast<const llama_kv_cache_context *>(mctx);

        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        ggml_tensor * k_idxs = inp_attn->get_k_idxs();

        // If MTP projection is available, use it; otherwise just use the hidden state
        auto build_proj = [&](ggml_tensor * tok_in, ggml_tensor * h_in) -> ggml_tensor * {
            if (has_mtp_proj) {
                ggml_tensor * h_norm_b = build_norm(h_in, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
                ggml_tensor * e_norm_b = build_norm(tok_in, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
                return build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm_b, h_norm_b, 0), layer.nextn.eh_proj_s);
            }
            // No MTP projection: just return the hidden state
            return h_in;
        };

        // Standard GQA attention block for MTP chained drafting
        auto build_block = [&](ggml_tensor * cur_in, int64_t row0, int64_t width) -> ggml_tensor * {
            ggml_tensor * cur_b = cur_in;
            ggml_tensor * inpSA_b = cur_b;

            cur_b = build_norm(cur_b, layer.attn_norm, nullptr, LLM_NORM_RMS, il);

            // Q projection
            ggml_tensor * Q_b = ggml_mul_mat(ctx0, layer.wq, cur_b);
            Q_b = ggml_reshape_3d(ctx0, Q_b, head_dim, n_head, width);
            Q_b = build_norm(Q_b, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);

            // K projection
            ggml_tensor * K_b = ggml_mul_mat(ctx0, layer.wk, cur_b);
            K_b = ggml_reshape_3d(ctx0, K_b, head_dim, n_head_kv, width);
            K_b = build_norm(K_b, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

            // V projection
            ggml_tensor * V_b = ggml_mul_mat(ctx0, layer.wv, cur_b);
            V_b = ggml_reshape_3d(ctx0, V_b, head_dim, n_head_kv, width);

            // RoPE positions
            ggml_tensor * pos_b = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_pos, width, 4,
                    (size_t) n_tokens*inp_pos->nb[0], (size_t) row0*inp_pos->nb[0]));
            pos_b = ggml_reshape_1d(ctx0, pos_b, 4*width);

            Q_b = ggml_rope_multi(ctx0, Q_b, pos_b, nullptr,
                    hparams.n_rot(), sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            K_b = ggml_rope_multi(ctx0, K_b, pos_b, nullptr,
                    hparams.n_rot(), sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            // Store K and V in cache
            ggml_tensor * k_idx_b = ggml_view_1d(ctx0, k_idxs, width, row0*k_idxs->nb[0]);
            ggml_build_forward_expand(gf, mctx_kv->cpy_k(ctx0, K_b, k_idx_b, il));
            ggml_build_forward_expand(gf, mctx_kv->cpy_v(ctx0, V_b, k_idx_b, il));

            // Attention
            ggml_tensor * k_view = mctx_kv->get_k(ctx0, il);
            ggml_tensor * v_view = mctx_kv->get_v(ctx0, il);

            cur_b = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Q_b, k_view, v_view, nullptr, nullptr, nullptr, kq_scale, il);
            cur_b = build_lora_mm(layer.wo, cur_b, layer.wo_s);
            cur_b = ggml_add(ctx0, cur_b, inpSA_b);

            ggml_tensor * ffn_res_b = cur_b;
            cur_b = build_norm(cur_b, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cur_b = qwen35mla_build_ffn(*this, layer, cur_b, il);
            cur_b = ggml_add(ctx0, cur_b, ffn_res_b);

            return cur_b;
        };

        int64_t n_catchup = 0;
        while (n_catchup < n_tokens && ubatch.output && ubatch.output[n_catchup] == 0) {
            n_catchup++;
        }
        const int64_t n_chain = n_tokens - n_catchup;
        GGML_ASSERT(n_chain >= 1);

        ggml_tensor * proj_all = build_proj(tok_embd, h_embd);

        if (n_catchup > 0) {
            ggml_tensor * proj_c = ggml_view_2d(ctx0, proj_all, proj_all->ne[0], n_catchup, proj_all->nb[1], 0);
            build_block(proj_c, 0, n_catchup);
        }

        ggml_tensor * proj_cur = ggml_view_2d(ctx0, proj_all, proj_all->ne[0], 1, proj_all->nb[1], (size_t) n_catchup*proj_all->nb[1]);

        ggml_tensor * logits_all = nullptr;
        ggml_tensor * h_all      = nullptr;

        for (int64_t j = 0; j < n_chain; ++j) {
            ggml_tensor * cur_j = build_block(proj_cur, n_catchup + j, 1);

            ggml_tensor * head_norm_w2 = layer.nextn.shared_head_norm
                    ? layer.nextn.shared_head_norm
                    : model.output_norm;
            ggml_tensor * h_next_j = build_norm(cur_j, head_norm_w2, nullptr, LLM_NORM_RMS, -1);

            ggml_tensor * head_w2 = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
            ggml_tensor * head_s2 = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;

            static const int64_t n_sub_env = [] {
                const char * env = getenv("LLAMA_SPEC_CHAIN_SUB");
                return env != nullptr ? atoll(env) : 32768;
            }();

            ggml_tensor * id_map   = nullptr;
            ggml_tensor * logits_j = build_draft_vocab_logits_chain(head_w2, head_s2, h_next_j);
            if (logits_j != nullptr) {
                id_map = draft_vocab_ids;
            } else if (n_sub_env > 0 && n_sub_env < head_w2->ne[1] &&
                    !(hadamard_rotations && hadamard_rotations->count(head_w2))) {
                ggml_tensor * head_sub = ggml_view_2d(ctx0, head_w2,
                        head_w2->ne[0], n_sub_env, head_w2->nb[1], 0);
                logits_j = ggml_mul_mat(ctx0, head_sub, h_next_j);
                if (head_s2 != nullptr) {
                    logits_j = ggml_mul(ctx0, logits_j, head_s2);
                }
            } else {
                logits_j = build_lora_mm(head_w2, h_next_j, head_s2);
            }

            ggml_tensor * id_j;
            if (id_map) {
                ggml_tensor * pos_j = ggml_argmax(ctx0, logits_j);
                id_j = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, id_map, 1, id_map->ne[0]), pos_j), 1);
            } else {
                id_j = ggml_argmax(ctx0, logits_j);
            }
            ggml_tensor * probs_j = ggml_soft_max(ctx0, logits_j);
            ggml_tensor * p_j = ggml_get_rows(ctx0,
                    ggml_reshape_2d(ctx0, probs_j, 1, probs_j->ne[0]), id_j);
            p_j = ggml_reshape_2d(ctx0, p_j, 1, 1);
            ggml_tensor * id_f = ggml_cast(ctx0, ggml_reshape_2d(ctx0, id_j, 1, 1), GGML_TYPE_F32);
            ggml_tensor * out_j = ggml_concat(ctx0, id_f, p_j, 0);

            logits_all = logits_all == nullptr ? out_j : ggml_concat(ctx0, logits_all, out_j, 1);
            h_all      = h_all      == nullptr ? h_next_j : ggml_concat(ctx0, h_all, h_next_j, 1);

            if (j + 1 < n_chain) {
                ggml_tensor * tok_j = ggml_get_rows(ctx0, tok_embd_w, id_j);
                tok_j = build_hadamard_inverse(tok_embd_w, tok_j);
                proj_cur = build_proj(tok_j, h_next_j);
            }
        }

        cb(h_all, "h_nextn", -1);
        res->t_h_nextn = h_all;
        ggml_build_forward_expand(gf, h_all);

        ggml_build_forward_expand(gf, ggml_view_1d(ctx0, inp_out_ids, inp_out_ids->ne[0], 0));

        cb(logits_all, "result_output", -1);
        res->t_logits = logits_all;
        ggml_build_forward_expand(gf, logits_all);

        return;
    }

    // --- Single-step MTP path ---
    // If MTP projection is available, use it; otherwise just use the hidden state
    ggml_tensor * cur;
    if (has_mtp_proj) {
        ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
        cb(h_norm, "mtp_hnorm", il);

        ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
        cb(e_norm, "mtp_enorm", il);

        ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
        cb(concat, "mtp_concat", il);

        cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
        cb(cur, "mtp_eh_proj", il);
    } else {
        // No MTP projection: just use the hidden state
        cur = h_embd;
        cb(cur, "mtp_h_direct", il);
    }

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // Standard GQA attention for MTP
    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
    Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur", il);

    ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, cur);
    Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur", il);

    ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, cur);
    Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    // RoPE
    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            hparams.n_rot(), sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            hparams.n_rot(), sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    // Attention
    cur = build_attn(inp_attn,
            layer.wo, nullptr, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = qwen35mla_build_ffn(*this, layer, cur, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35_MLA MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35_MLA MTP: missing LM head (nextn.shared_head_head or model.output)");
    ggml_tensor * logits = build_draft_vocab_logits(head_w, head_s, cur);
    cur = logits ? logits : build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
