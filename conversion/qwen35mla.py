"""Converter for Qwen3.8-27B-MLA (TelperionAI) - hybrid Gated DeltaNet + MLA architecture.

MLA specifics:
  - Per-layer latent rank (256 / 768 / 1792)
  - Per-KV-head RoPE (4 KV heads x 58 dims = 232 total in fused tensor)
  - K decompressed per-head dim: 198 (nope) + 58 (rope) = 256
  - V decompressed per-head dim: 256
  - Absorbed attention path: K B-matrix absorbed into Q, V B-matrix in-kernel
"""
from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen import _Qwen35MRopeMixin, _LinearAttentionVReorderBase


@ModelBase.register("Qwen3_5MLAForConditionalGeneration", "Qwen3_5MLAForCausalLM")
@ModelBase.example("TelperionAI/Qwen3.8-27B-MLA")
class Qwen35MLATextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35_MLA

    # Per-layer latent ranks from config (only full-attention layers)
    _mla_ranks: dict[int, int] | None = None
    # Buffer for fusing kv_a_proj + k_rope_proj
    _kv_a_buf: dict[int, dict[str, Tensor]] | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        text_cfg = self.hparams
        n_head = text_cfg["num_attention_heads"]          # 24
        n_embd = text_cfg["hidden_size"]                   # 5120
        v_head_dim = text_cfg["v_head_dim"]                # 256
        mla_qk_rope = text_cfg.get("mla_qk_rope_head_dim", 58)  # 58
        n_kv_head = text_cfg["num_key_value_heads"]        # 4
        fused_rope_dim = n_kv_head * mla_qk_rope           # 232

        # Non-absorbed MLA: KV cache stores materialized K/V with n_head heads
        self.hparams["num_key_value_heads"] = n_head

        # Override n_rot: MLA uses mla_qk_rope_head_dim (58), not the standard
        # partial_rotary_factor * head_dim (64). This affects the C++ n_rot() value.
        self.gguf_writer.add_rope_dimension_count(mla_qk_rope)

        # Per-layer latent ranks (only full-attention layers have non-zero entries)
        mla_ranks_cfg = text_cfg.get("mla_ranks", {})
        self._mla_ranks = {int(k): v for k, v in mla_ranks_cfg.items()}

        # Determine the actual per-head nope dim from k_up_proj shape
        # k_up_proj output = n_head * qk_nope_actual
        # We'll verify this during tensor processing; for now use config value
        qk_nope_cfg = text_cfg.get("qk_nope_head_dim", 192)
        # The actual value may differ (198 vs 192); we'll detect from tensor shape
        self._qk_nope_actual: int | None = None
        self._v_head_dim_actual: int | None = None

        # Write MLA GGUF parameters
        # key_length_mla = decompressed K per-head dim (nope + rope)
        key_len_mla = v_head_dim  # 256 (same as v_head_dim in this model)
        self.gguf_writer.add_key_length_mla(key_len_mla)
        self.gguf_writer.add_value_length_mla(v_head_dim)

        # Fused rope dim (total, not per-head)
        self.gguf_writer.add_mla_fused_rope_dim(fused_rope_dim)

        # Per-layer latent ranks array (only full-attention layers, in order)
        layer_types = text_cfg.get("layer_types", [])
        fa_ranks = []
        for i, lt in enumerate(layer_types):
            if lt == "full_attention":
                rank = self._mla_ranks.get(i, 0)
                fa_ranks.append(rank)
        if fa_ranks:
            self.gguf_writer.add_latent_rank_per_layer(fa_ranks)
            self.gguf_writer.add_kv_lora_rank(max(fa_ranks))

        logger.info(
            f"Qwen35MLA: n_head={n_head}, v_head_dim={v_head_dim}, "
            f"fused_rope_dim={fused_rope_dim}, fa_layers={len(fa_ranks)}, "
            f"ranks={set(fa_ranks)}"
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if bid is None or ".self_attn." not in name:
            yield from super().modify_tensors(data_torch, name, bid)
            return

        # --- MLA tensor handling (full-attention layers only) ---

        # kv_a_proj: (lora_rank, n_embd) — latent projection
        if name.endswith(".kv_a_proj.weight"):
            if self._kv_a_buf is None:
                self._kv_a_buf = {}
            self._kv_a_buf.setdefault(bid, {})["kv_a"] = data_torch
            return

        # k_rope_proj: (fused_rope_dim, n_embd) — RoPE key projection
        if name.endswith(".k_rope_proj.weight"):
            if self._kv_a_buf is None:
                self._kv_a_buf = {}
            self._kv_a_buf.setdefault(bid, {})["k_rope"] = data_torch

            # Both parts available? Fuse and emit.
            buf = self._kv_a_buf.get(bid, {})
            if "kv_a" in buf and "k_rope" in buf:
                fused = torch.cat([buf["kv_a"], buf["k_rope"]], dim=0)
                del self._kv_a_buf[bid]
                new_name = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_KV_A_MQA, bid)
                logger.info(f"  fused wkv_a_mqa [{bid}]: {tuple(fused.shape)}")
                yield from super().modify_tensors(fused, new_name, bid)
            return

        # k_up_proj: (n_head * qk_nope, lora_rank) — K decompression B-matrix
        if name.endswith(".k_up_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            out_dim, lora_rank = data_torch.shape
            qk_nope = out_dim // n_head
            self._qk_nope_actual = qk_nope

            # (n_head*qk_nope, lora_rank) -> (n_head, qk_nope, lora_rank) -> (qk_nope, lora_rank, n_head)
            data_torch = data_torch.view(n_head, qk_nope, lora_rank).permute(1, 2, 0).contiguous()
            new_name = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid)
            logger.info(f"  wk_b [{bid}]: ({out_dim},{lora_rank}) -> {tuple(data_torch.shape)}")
            yield from super().modify_tensors(data_torch, new_name, bid)
            return

        # v_up_proj: (n_head * v_head_dim, lora_rank) — V decompression B-matrix
        if name.endswith(".v_up_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            out_dim, lora_rank = data_torch.shape
            v_head = out_dim // n_head
            self._v_head_dim_actual = v_head

            # (n_head*v_head, lora_rank) -> (n_head, v_head, lora_rank) -> (lora_rank, v_head, n_head)
            data_torch = data_torch.view(n_head, v_head, lora_rank).permute(2, 1, 0).contiguous()
            new_name = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid)
            logger.info(f"  wv_b [{bid}]: ({out_dim},{lora_rank}) -> {tuple(data_torch.shape)}")
            yield from super().modify_tensors(data_torch, new_name, bid)
            return

        # All other tensors: pass through to base (handles q_proj, o_proj, norms, DeltaNet, FFN, etc.)
        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        # Verify no unprocessed buffers remain
        if self._kv_a_buf:
            for bid, buf in self._kv_a_buf.items():
                if buf:
                    raise ValueError(f"Unprocessed MLA buffer for layer {bid}: {list(buf.keys())}")
            self._kv_a_buf = None

        if self._qk_nope_actual:
            logger.info(f"Qwen35MLA: detected qk_nope={self._qk_nope_actual}, v_head={self._v_head_dim_actual}")
