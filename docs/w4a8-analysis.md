# Analyse W4A8 — Port vers llama.cpp

## Résumé exécutif

**La bonne nouvelle** : llama.cpp utilise **déjà** les INT8 tensor cores (DP4A) pour le GEMM Q4_K.
La fonction `ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a` dans `mmq-vec-dot.cuh` applique exactement
le même principe que le Marlin W4A8 de vLLM : poids 4-bit × activations INT8 → DP4A.

**Le gap de performance (1200 vs 1850 tok/s) ne vient PAS du type de GEMM.**
Il vient d'autres facteurs, détaillés ci-dessous.

---

## 1. État actuel du GEMM Q4_K dans llama.cpp

### Pipeline existant

```
src1 (FP32 activations)
    │
    ▼ quantize_mmq_q8_1_cuda()
src1_q8_1 (Q8_1 = INT8 + per-32 scales)
    │
    ▼ ggml_cuda_mul_mat_q_switch_type()
    │
    ├──► vec_dot_q4_K_q8_1_dp4a()  ← DP4A (INT8 tensor cores) ✅
    │       └── vec_dot_q4_K_q8_1_impl_mmq()
    │
    └──► (fallback: vec_dot_q4_K_q8_1_mma() si DP4A non dispo)
```

### Ce que fait le kernel DP4A Q4_K

```cpp
// mmq-vec-dot.cuh:905
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a(
        const int *x, const int *y, float *sum, const int k00) {
    // x = tile de poids Q4_K (INT4 + scales half2 + scales int)
    // y = tile d'activations Q8_1 (INT8 + scales half2)
    
    for (k01 in K_tile) {
        for (j in J) {
            for (i in I) {
                sum[...] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[...],   // poids INT4
                    &y_qs[...],   // activations INT8
                    sc, sc+8,     // scales poids (6-bit → half)
                    x_dm[i],      // min/scale super-block
                    &y_ds[...]    // scales activations
                );
            }
        }
    }
}
```

**Le DP4A est déjà utilisé.** C'est la même chose que le Marlin W4A8.

---

## 2. Alors d'où vient le gap de 1200 vs 1850 tok/s ?

### 2.1 Comparaison des deux stacks

| Facteur | llama.cpp (ta config) | vLLM + HyperQwen |
|---------|----------------------|------------------|
| GEMM Q4_K | DP4A (INT8 TC) ✅ | Marlin W4A8 (INT8 TC) ✅ |
| Quantization activations | Q8_1 (per-32 scales) | INT8 per-token (1 scale/token) |
| Tile sizes | Génériques (I=128, J=64) | **Tunés pour sm86** (marlin-tune-table) |
| CUDA Graphs | ❌ (kernel launch par GEMM) | ✅ (graph capture) |
| Compilation | `-O3` | `-O3 --use_fast_math` + torch.compile |
| GDN state | **FP32** (par défaut) | **FP16** (`--mamba-ssm-cache-dtype float16`) |
| Embeddings (lm_head) | **BF16** (2.5 GB) | **INT8 requant** (0.6 GB) |
| Attention QK^T | INT8 (ta branche) ✅ | INT8 Triton ✅ |
| Batch prefill | Variable (chunked) | **2048 tokens/chunk** (optimisé) |
| Memory allocator | cudaMalloc + pool | **PagedAttention** (contiguous blocks) |

### 2.2 Impact estimé de chaque facteur

| Optimisation manquante | Gain estimé | Effort |
|------------------------|-------------|--------|
| **GDN state FP32 → FP16** | +3-5% (decode), +1-2% (prefill) | **Très faible** (1 flag) |
| **CUDA Graphs** | +5-10% (réduit kernel launch overhead) | Moyen |
| **Marlin tune table (tile sizes)** | +5-8% sur GEMM | Moyen |
| **INT8 per-token vs Q8_1 per-32** | +2-4% sur GEMM | Élevé |
| **Embeddings requant INT8** | 0% speed, -2 GB VRAM | Faible |
| **Compilation flags** | +2-3% | Très faible |
| **Chunk size prefill (2048)** | +3-5% | Faible |

**Total potentiel : +15-25%** → de 1200 à ~1400-1500 tok/s

Le reste du gap (1500 → 1850) vient probablement de :
- L'efficacité du kernel Marlin vs le kernel MMQ (différentes implémentations)
- Le paged attention de vLLM (meilleur memory pattern pour le KV cache)
- Le torch.compile qui fuse des opérations

### 2.3 Le chiffre clé

Du profil torch de HyperQwen :
> "79% Marlin GEMM time, 15 ms GPU idle out of 2.05 s"

Dans llama.cpp, le GPU idle est probablement **plus élevé** à cause de :
- Kernel launches non fusionnés (pas de CUDA Graphs)
- GDN state en FP32 (plus de bande passante mémoire)
- Chunk size non optimisé

---

## 3. Plan d'optimisation (par ordre de ROI)

### Phase 1 : Gains rapides (1-2 jours)

#### 1.1 GDN state en FP16

**Fichier** : `ggml/src/ggml-cuda/ggml-cuda.cu` ou le code GDN

**Problème** : Le state récurrent du Gated DeltaNet est stocké en FP32 par défaut.
C'est 48/64 layers qui utilisent ce state. Chaque accès au state double la bande
passante mémoire inutilement.

**Fix** :
```cpp
// Chercher où le GDN state est alloué et forcer FP16
// Ou ajouter un flag --gdn-state-dtype fp16
```

**Gain** : +3-5% decode, +1-2% prefill
**Effort** : 2-4 heures

#### 1.2 Chunk size prefill

**Problème** : Le chunk size par défaut de llama.cpp pour le prefill chunked
n'est pas optimisé pour les shapes de Qwen3.8-27B.

**Fix** :
```bash
# Tester différents chunk sizes
./llama-bench -m model.gguf -n 0 -b 2048  # 2048 tokens par chunk
./llama-bench -m model.gguf -n 0 -b 4096
./llama-bench -m model.gguf -n 0 -b 1024
```

HyperQwen utilise **2048 tokens/chunk** pour le prefill.

**Gain** : +3-5%
**Effort** : 30 min (test)

#### 1.3 Compilation flags

**Problème** : Le build par défaut n'utilise pas tous les flags d'optimisation.

**Fix** :
```cmake
# CMakeLists.txt ou flags de build
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
# Vérifier que -O3 est bien actif
```

**Gain** : +2-3%
**Effort** : 1 heure

### Phase 2 : Gains moyens (1-2 semaines)

#### 2.1 CUDA Graphs pour le GEMM

**Problème** : Chaque GEMM est un kernel launch séparé. Sur 64 layers × 3 GEMM FFN
+ projections, c'est ~200+ kernel launches par forward pass. L'overhead de launch
(~5-10 µs each) s'accumule.

**Solution** : Capturer les GEMM en CUDA Graph et les rejouer.

**Fichiers à modifier** :
- `ggml/src/ggml-cuda/ggml-cuda.cu` : ajouter graph capture
- `ggml/src/ggml-cuda/mmq.cu` : faire les GEMM graph-capturable

**Gain** : +5-10%
**Effort** : 1-2 semaines

#### 2.2 Tile sizes optimisés pour sm86 + Qwen3.8-27B

**Problème** : Le MMQ framework utilise des tile sizes génériques (I=128, J=64).
Pour les shapes de Qwen3.8-27B (K=5120, N=17408, M=128-2048), des tile sizes
différents seraient plus efficaces.

**Shapes critiques** :
- FFN gate/up: [17408, 5120] × [5120, batch]
- FFN down: [5120, 17408] × [17408, batch]
- Attention proj: [5120, 5120] × [5120, batch]

**Solution** : Ajouter une table de tile sizes par (type, K, N, M, cc)
similaire à la `marlin-tune-table.patch` de HyperQwen.

**Fichiers à modifier** :
- `ggml/src/ggml-cuda/mmq.cuh` : ajouter table de configs
- `ggml/src/ggml-cuda/mmq.cu` : lookup de la config optimale

**Gain** : +5-8% sur GEMM
**Effort** : 1 semaine (benchmark + table)

### Phase 3 : Gains avancés (2-4 semaines)

#### 3.1 INT8 per-token activations (vs Q8_1 per-32)

**Problème** : La quantization Q8_1 utilise des scales per-32 éléments.
Le Marlin W4A8 utilise des scales per-token (1 scale pour tout le vecteur).
C'est plus rapide car :
- 1 lookup de scale vs 16 lookups (pour K=5120)
- Moins de bande passante pour les scales
- Le scale peut être hoisted hors de la boucle interne

**Solution** : Ajouter un mode de quantization "per-token" pour les activations.

**Fichiers à modifier** :
- `ggml/src/ggml-cuda/mmq-quantize.cu` : nouveau mode de quantization
- `ggml/src/ggml-cuda/mmq-vec-dot.cuh` : nouveau vec_dot avec scale per-token
- `ggml/src/ggml-cuda/mmq.cu` : dispatch vers le nouveau mode

**Gain** : +2-4% sur GEMM
**Effort** : 2-3 semaines

#### 3.2 Fusion GEMM + Activation (SiLU, etc.)

**Problème** : Le SiLU (ou GLU) entre les GEMM FFN est un kernel séparé.
Le fusionner avec le GEMM éviterait un round-trip mémoire.

**Solution** : Ajouter un epilogue SiLU dans le kernel GEMM.

**Gain** : +2-3%
**Effort** : 1-2 semaines

---

## 4. Ce qu'il NE faut PAS faire

### ❌ Port du Marlin kernel de vLLM

Le Marlin kernel de vLLM est écrit en CUDA C++ avec un layout de poids spécifique
(Marlin layout). Le porter dans llama.cpp serait :
- 4-6 semaines de travail
- Un conflit avec le framework MMQ existant
- Pas nécessaire : le MMQ déjà utilise DP4A

### ❌ Changer de quantization (Q4_0, Q8_0)

- Q4_0 : perte de qualité, pas de gain de speed (déjà DP4A)
- Q8_0 : double la VRAM, pas de gain de speed (déjà DP4A)

### ❌ Utiliser vLLM à la place

C'est une option valide si l'objectif est la performance pure,
mais ça quitte l'écosystème llama.cpp (GGUF, quantization, etc.)

---

## 5. Roadmap recommandée

```
Semaine 1 :
  ├── [1.1] GDN state FP16          → +3-5%
  ├── [1.2] Chunk size tuning       → +3-5%
  └── [1.3] Compilation flags       → +2-3%
  Total semaine 1 : +8-13% (1200 → ~1300-1350 tok/s)

Semaine 2-3 :
  ├── [2.1] CUDA Graphs             → +5-10%
  └── [2.2] Tile sizes sm86         → +5-8%
  Total semaine 2-3 : +10-18% (1350 → ~1500-1580 tok/s)

Semaine 4-6 :
  ├── [3.1] INT8 per-token          → +2-4%
  └── [3.2] GEMM+SiLU fusion        → +2-3%
  Total semaine 4-6 : +4-7% (1580 → ~1650-1700 tok/s)
```

**Target réaliste : ~1650-1700 tok/s prefill** (vs 1850-1940 de HyperQwen)

Le gap résiduel (~10%) vient de différences architecturales entre vLLM et llama.cpp
(paged attention, torch.compile, memory allocator) qui sont difficiles à combler.

---

## 6. Vérification immédiate

Pour confirmer que le DP4A est bien actif sur ta 3090, lance :

```bash
# Vérifier que le kernel DP4A est compilé
grep -c "dp4a" build/CMakeFiles/ggml-cuda.dir/ggml/src/ggml-cuda/mmq-vec-dot.cu.o

# Ou vérifier via Nsight Compute
ncu --kernel-name regex:mul_mat_q --metrics sm__inst_executed_pipe_tensor_op_int8 \
    ./llama-bench -m model.gguf -n 0 -b 128
```

Si le métrique `tensor_op_int8` est > 0, le DP4A est actif.
