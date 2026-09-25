# GDN State FP32 → FP16 — Analyse détaillée

## 1. Ce qu'est le GDN state

Le **Gated DeltaNet (GDN)** est le mécanisme d'attention linéaire utilisé dans
Qwen3.8-27B (48/64 layers). Contrairement à l'attention standard (KV cache),
le GDN maintient un **état récurrent** `S` de shape `[S_v, S_v, H_v, n_seqs]`
par layer, mis à jour à chaque token :

```
S[i][col] = g * S[i][col] + k[i] * delta[col]    (pour chaque token)
```

Ce state est le **seul** état persistant du GDN — il remplace le KV cache.
Sa taille par layer est `S_v × S_v × H_v × n_seqs × sizeof(dtype)`.

### Taille pour Qwen3.8-27B (S_v=128, H_v≈64, n_seqs=1)

| Dtype | Par layer | 48 layers |
|-------|-----------|-----------|
| FP32 | 128×128×64×4 = **4 MB** | **192 MB** |
| FP16 | 128×128×64×2 = **2 MB** | **96 MB** |

**Économie VRAM : 96 MB** (modeste, mais le gain principal est la bande passante).

---

## 2. Pourquoi FP16 est bénéficiaire

Le state est lu **une fois** et écrit **une fois** par token (decode) ou
par chunk (prefill) **par layer**. C'est le plus gros flux mémoire du GDN kernel.

### Decode (n_tokens=1, K=1)

Par layer, par token :
- Read state: 4 MB (FP32) / 2 MB (FP16)
- Write state: 4 MB (FP32) / 2 MB (FP16)
- Autres I/O (q,k,v,g,beta,out): ~0.1 MB

| | FP32 | FP16 | Gain |
|---|---|---|---|
| I/O total/layer | 8.1 MB | 4.1 MB | **-50%** |
| 48 layers | 389 MB | 197 MB | **-192 MB/token** |
| @ 936 GB/s (3090) | 416 µs | 210 µs | **-206 µs/token** |

### Prefill (n_tokens=128, K=128, keep_rs=true)

Le state est écrit **K fois** (une fois par token retenu) :
- Write state: K × 4 MB = 512 MB (FP32) / K × 2 MB = 256 MB (FP16) par layer

| | FP32 | FP16 | Gain |
|---|---|---|---|
| I/O state/layer | 516 MB | 260 MB | **-256 MB** |
| 48 layers | 24.8 GB | 12.5 GB | **-12.3 GB/prefill** |
| @ 936 GB/s | 26.5 ms | 13.4 ms | **-13.1 ms/prefill** |

**Le gain prefill est massif** : ~13 ms économisés par chunk de 128 tokens.

---

## 3. Architecture actuelle du state dans llama.cpp

### 3.1 Allocation

```
src/llama-model.cpp:3102-3111
┌─────────────────────────────────────────────────────────────────┐
│ res = new llama_memory_hybrid(                                  │
│     ...                                                         │
│     /* recurrent_type_k */ GGML_TYPE_F32,   ← HARDCODED        │
│     /* recurrent_type_v */ GGML_TYPE_F32,   ← HARDCODED        │
│     ...                                                         │
│ );                                                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
src/llama-memory-recurrent.cpp:129-130
┌─────────────────────────────────────────────────────────────────┐
│ ggml_tensor * s = ggml_new_tensor_2d(                           │
│     ctx, type_s,              ← GGML_TYPE_F32                   │
│     hparams.n_embd_s(),       ← S_v × S_v × H_v                 │
│     n_rows_s                  ← mem_size × (1 + n_rs_seq)       │
│ );                                                              │
│ s_l[i] = s;                                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Flow de données par token

```
                    ┌──────────────────────────────────────────────────┐
                    │           GDN Kernel (CUDA)                      │
                    │                                                  │
  s_l[i] (FP32) ──►│  Load state → registers (FP32)                   │
  [input state]     │  Compute recurrence in FP32 registers           │
                    │  Store state → output tensor tail (FP32)        │
                    │  Store attn scores → output (FP32)              │
                    └──────────────────────────────────────────────────┘
                    │
                    ▼
  dst tensor (FP32)
  ┌──────────────────────────────────────────────────────────────────┐
  │ [attn scores: S_v × H × T × B] [state: S_v × S_v × H × B]     │
  └──────────────────────────────────────────────────────────────────┘
                    │
                    ▼  (memory management: copy state back)
  s_l[i] (FP32)  ← updated for next token
```

### 3.3 Constrainte actuelle

```c
// ggml/src/ggml.c:6540
GGML_ASSERT(state->type == GGML_TYPE_F32);  ← BLOQUE FP16
```

---

## 4. Ce qu'il faut changer — Plan détaillé

### Changement 1 : Allocation du state en FP16

**Fichier** : `src/llama-model.cpp` (~ligne 3110)

```cpp
// AVANT
/* recurrent_type_k  */ GGML_TYPE_F32,
/* recurrent_type_v  */ GGML_TYPE_F32,

// APRÈS (option A: flag CLI)
/* recurrent_type_k  */ cparams.gdn_state_dtype,  // GGML_TYPE_F32 ou GGML_TYPE_F16
/* recurrent_type_v  */ cparams.gdn_state_dtype,

// APRÈS (option B: env var, plus rapide)
/* recurrent_type_k  */ getenv("LLAMA_GDN_STATE_F16") ? GGML_TYPE_F16 : GGML_TYPE_F32,
/* recurrent_type_v  */ getenv("LLAMA_GDN_STATE_F16") ? GGML_TYPE_F16 : GGML_TYPE_F32,
```

**Fichier** : `common/common.h` ou `src/llama.h` (si option A)
```cpp
// Ajouter au struct params:
ggml_type gdn_state_dtype = GGML_TYPE_F32;
```

**Fichier** : `common/arg.cpp` (si option A)
```cpp
// Ajouter le flag:
{ "--gdn-state-dtype", {""}, "dtype for GDN recurrent state (f32, f16)",
  [](common_params & params, const char * value) {
      if (strcmp(value, "f16") == 0) params.gdn_state_dtype = GGML_TYPE_F16;
      else if (strcmp(value, "f32") == 0) params.gdn_state_dtype = GGML_TYPE_F32;
      else throw std::invalid_argument("invalid dtype");
  }},
```

### Changement 2 : Relâcher l'assertion dans ggml_gated_delta_net

**Fichier** : `ggml/src/ggml.c` (~ligne 6540)

```c
// AVANT
GGML_ASSERT(state->type == GGML_TYPE_F32);

// APRÈS
GGML_ASSERT(state->type == GGML_TYPE_F32 || state->type == GGML_TYPE_F16);
```

### Changement 3 : Kernel CUDA — supporter FP16 state

**Fichier** : `ggml/src/ggml-cuda/gated_delta_net.cu`

C'est le changement principal. Le kernel actuel :

```cpp
// Ligne ~25-30 : signature du kernel
template <int S_v, bool KDA, bool keep_rs_t, bool emit_ingredients_t>
__global__ void gated_delta_net_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * curr_state,    ← TOUJOURS FP32
    float *       dst,
    float *       state,         ← TOUJOURS FP32
    ...
```

**Approche** : Ajouter un template parameter `bool STATE_F16` (ou utiliser
`if constexpr` sur le type du state).

```cpp
template <int S_v, bool KDA, bool keep_rs_t, bool emit_ingredients_t, bool STATE_F16>
__global__ void gated_delta_net_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * curr_state,    // reinterpret as half* if STATE_F16
    float *       dst,
    float *       state,         // reinterpret as half* if STATE_F16
    ...
{
    // ...

    // LOAD STATE (ligne ~75)
    // AVANT:
    //   s_shard[r] = curr_state[i];
    // APRÈS:
    if constexpr (STATE_F16) {
        const half * curr_state_h = (const half *) curr_state;
        s_shard[r] = __half2float(curr_state_h[i]);
    } else {
        s_shard[r] = curr_state[i];
    }

    // ... COMPUTE (inchangé, toujours en FP32 dans les registres) ...

    // STORE STATE (ligne ~230)
    // AVANT:
    //   state_out[col * S_v + i] = s_shard[r];
    // APRÈS:
    if constexpr (STATE_F16) {
        half * state_out_h = (half *) state_out;
        state_out_h[col * S_v + i] = __float2half(s_shard[r]);
    } else {
        state_out[col * S_v + i] = s_shard[r];
    }
}
```

**Même changement** pour le kernel `gated_delta_net_cuda_ilp` (ligne ~250).

### Changement 4 : Dispatch dans le launch

**Fichier** : `ggml/src/ggml-cuda/gated_delta_net.cu` (~ligne 570)

Dans `ggml_cuda_op_gated_delta_net_impl` :

```cpp
// AVANT:
const float * s_d = (const float *) src_state->data;
float *       state_d = dst_d + S_v * H * n_tokens * n_seqs;

// APRÈS:
const bool state_f16 = (src_state->type == GGML_TYPE_F16);

// Les pointers restent les mêmes (on fait le cast dans le kernel)
const float * s_d = (const float *) src_state->data;
float *       state_d = dst_d + S_v * H * n_tokens * n_seqs;

// Dispatch:
if (state_f16) {
    launch_gated_delta_net<..., /*STATE_F16=*/true>(...);
} else {
    launch_gated_delta_net<..., /*STATE_F16=*/false>(...);
}
```

### Changement 5 : Le state output dans le dst tensor

**Problème** : Le state est écrit dans le tail du tensor `dst` qui est alloué
en FP32 (`ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)`). Si on veut écrire le
state en FP16, on ne peut pas l'écrire directement dans un buffer FP32.

**Solution** : Le state output reste écrit en FP32 dans le dst tensor
(comme aujourd'hui). Le cast FP32→FP16 se fait lors de la copie back vers
`s_l[i]` par le memory management.

**Mais** : ce cast ajoute un kernel supplémentaire. Pour l'éviter, on peut :

**Option A (simple, +1 kernel)** :
- Le GDN kernel écrit le state en FP32 dans le dst tensor (inchangé)
- Le memory management fait un `ggml_cast` du state (FP32→FP16) en copiant vers `s_l[i]`
- Overhead: 1 petit kernel de cast par layer (~3 MB I/O)

**Option B (optimal, 0 kernel extra)** :
- Le GDN kernel écrit le state **directement** dans `s_l[i]` (FP16)
- Il faut passer `s_l[i]` comme buffer de sortie du state au kernel
- Le dst tensor ne contient que les attention scores (pas le state)
- Changement plus invasif : modifier l'interface du GDN op

**Recommandation** : Commencer avec Option A (simple, 2h de travail),
passer à Option B si le cast est mesurable en profil.

### Changement 6 : Memory management — copy back avec cast

**Fichier** : `src/llama-memory-recurrent.cpp`

Là où le state est copié du dst tensor vers `s_l[i]` :

```cpp
// Si types différent (dst=FP32, s_l=FP16):
if (s_l[il]->type != dst_state->type) {
    // ggml_cast + ggml_backend_copy
    // ou cudaMemcopy si même device
}
```

En pratique, le copy back est probablement fait par `ggml_backend_copy` qui
gère déjà les casts. Vérifier que le backend CUDA supporte le copy FP32→FP16.

---

## 5. Résumé des fichiers à modifier

| # | Fichier | Changement | Lignes |
|---|---------|-----------|--------|
| 1 | `src/llama-model.cpp` | Changer `GGML_TYPE_F32` → configurable | ~3110 |
| 2 | `common/arg.cpp` | Ajouter flag `--gdn-state-dtype` | ~50 lignes |
| 3 | `common/common.h` | Ajouter field `gdn_state_dtype` | ~5 lignes |
| 4 | `ggml/src/ggml.c` | Relâcher assertion | 1 ligne |
| 5 | `ggml/src/ggml-cuda/gated_delta_net.cu` | Template `STATE_F16`, load/store half | ~40 lignes |
| 6 | `ggml/src/ggml-cuda/gated_delta_net.cu` | Dispatch dans launch | ~20 lignes |
| 7 | `ggml/src/ggml-cpu/ggml-cpu.c` | Même changement pour CPU (optionnel) | ~20 lignes |

**Total estimé : ~150 lignes de code, 4-8 heures de travail.**

---

## 6. Risques et vérifications

### Risque 1 : Precision loss

Le state GDN contient des valeurs qui peuvent être petites (decay exponentiel).
FP16 a un range de ±65504 et une précision de ~3 chiffres significatifs.

**Mitigation** : Le compute reste en FP32 dans les registres. Seule la
stockage en mémoire est en FP16. La perte est comparable au KV cache en FP16
(qui est déjà standard dans llama.cpp).

**Vérification** : Comparer les logits en FP32 vs FP16 state sur un prompt
court. Le perplexity devrait être identique à ±0.01.

### Risque 2 : gdn_replay

Le mode `gdn_replay` (DRC) utilise des "ingredient slots" (k, v, g, beta)
au lieu de snapshots complets du state. Ces ingredients sont aussi stockés
dans `s_l[i]` avec le même `type_s`.

**Impact** : Les ingredients (k, v, g, beta) en FP16 pourraient perdre de la
précision pour le replay. Mais c'est le même trade-off que le KV cache FP16.

**Vérification** : Tester `--gdn-replay` avec state FP16.

### Risque 3 : Save/load du state

Le state est sauvegardé/chargé dans les session files. Le code utilise
`ggml_type_size(s_l[il]->type)` pour calculer les offsets, ce qui est
déjà dtype-aware.

**Impact** : Aucun. Le code est déjà paramétré par type.

---

## 7. Plan d'implémentation

```
Étape 1 (1h) : Allocation FP16 + assertion
  ├── src/llama-model.cpp : changer type_s
  ├── ggml/src/ggml.c : relâcher assert
  └── Vérifier que le build passe

Étape 2 (2h) : Kernel CUDA FP16
  ├── gated_delta_net.cu : template STATE_F16
  ├── Load: __half2float
  ├── Store: __float2half
  └── Dispatch dans ggml_cuda_op_gated_delta_net_impl

Étape 3 (1h) : Copy back avec cast
  ├── Vérifier que ggml_backend_copy gère FP32→FP16
  ├── Si non : ajouter un kernel de cast
  └── Tester le round-trip state

Étape 4 (1h) : Tests
  ├── llama-perplexity : comparer FP32 vs FP16
  ├── llama-bench : mesurer le gain decode + prefill
  └── Vérifier la qualité (logits, perplexity)
```

**Total : 5-6 heures**

---

## 8. Gain attendu

| Metric | FP32 (actuel) | FP16 (proposé) | Gain |
|--------|--------------|----------------|------|
| VRAM state | 192 MB | 96 MB | -96 MB |
| Decode I/O state | 389 MB/token | 197 MB/token | -50% |
| Prefill I/O state | 24.8 GB/chunk | 12.5 GB/chunk | -50% |
| Decode speedup | — | — | +2-4% |
| Prefill speedup | — | — | +3-5% |

Le gain est **modeste en %** car le GDN n'est qu'une partie du forward pass
(les GEMM FFN dominent). Mais c'est un gain **gratuit** (pas de perte de
qualité mesurable) et il s'additionne aux autres optimisations.
