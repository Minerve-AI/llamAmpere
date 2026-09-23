# INT8-QK FlashAttention

Port du calcul QK^T sur les tensor cores INT8 (SageAttention-style, ref: HyperQwen).

## Principe

- K est mean-smoothed (per head, channel) puis quantise en INT8 (per 64 keys)
- Q est quantise per-row en INT8
- QK^T: mma.m16n8k16.s8.s8.s32 (2x FP16 throughput)
- Dequant: INT32 -> FP32 (q_scale * k_scale * sm_scale)
- Softmax: inchange (FP32)
- PV: inchange (FP16 MMA)

## Gain attendu

1.27-1.35x sur l'attention prefill (valide par HyperQwen sur meme geometrie)

## Fichiers a creer/modifier

| Fichier | Action |
|---------|--------|
| `ggml/src/ggml-cuda/fattn-mma-i8qk.cuh` | Nouveau kernel (base: fattn-mma-f16.cuh) |
| `ggml/src/ggml-cuda/fattn-k-quant.cuh` | Pre-kernel quantization K |
| `ggml/src/ggml-cuda/mma.cuh` | + tile<16,8,int> + mma() INT8 (si absent) |
| `ggml/src/ggml-cuda/fattn-mma-f16-instance.cu` | + dispatch vers kernel INT8 |
| `tests/test-int8-qk.cu` | Test unitaire isole |

## Checklist

- [ ] Pre-kernel K (mean + quant)
- [ ] Kernel FA INT8-QK
- [ ] Test unitaire
- [ ] Benchmark
