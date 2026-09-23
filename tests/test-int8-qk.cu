// test-int8-qk.cu
// Standalone unit test for INT8-QK FlashAttention (Q8_0 KV cache)
// Compile: nvcc -O3 -arch=sm_86 test-int8-qk.cu -o test-int8-qk -I../ggml/src/ggml-cuda/
// Run: ./test-int8-qk

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "fattn-k-quant.cuh"
#include "fattn-mma-i8.cuh"

// =====================================================================
// CPU reference
// =====================================================================

static void cpu_q8_to_float(const ggml_cuda_fattn_i8qk::block_q8_0_cuda * K_q8,
                            float * K_f32, int seq_len, int n_heads, int head_dim) {
    const int nblocks = head_dim / 32;
    for (int h = 0; h < n_heads; ++h) {
        for (int s = 0; s < seq_len; ++s) {
            for (int c = 0; c < head_dim; ++c) {
                const int idx = (h * seq_len + s) * nblocks + c / 32;
                K_f32[(h * seq_len + s) * head_dim + c] =
                    (float)K_q8[idx].qs[c % 32] * K_q8[idx].d;
            }
        }
    }
}

static void cpu_attention_ref(
        const float * Q, const float * K, const float * V, float * O,
        int nq, int nk, int dk, int dv, float sm_scale) {
    for (int i = 0; i < nq; ++i) {
        float * S = (float *)malloc(nk * sizeof(float));
        float max_s = -1e30f;
        for (int j = 0; j < nk; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < dk; ++k) dot += Q[i*dk+k] * K[j*dk+k];
            S[j] = dot * sm_scale;
            if (j <= i) max_s = fmaxf(max_s, S[j]); // causal
        }
        float sum = 0.0f;
        for (int j = 0; j < nk; ++j) {
            if (j > i) { S[j] = 0.0f; continue; }
            S[j] = expf(S[j] - max_s);
            sum += S[j];
        }
        for (int j = 0; j < nk; ++j) S[j] /= sum;
        for (int d = 0; d < dv; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < nk; ++j) acc += S[j] * V[j*dv+d];
            O[i*dv+d] = acc;
        }
        free(S);
    }
}

// =====================================================================
// Test 1: K quantization (Q8_0 → INT8)
// =====================================================================
static int test_k_quant_q8() {
    const int seq_len = 256;
    const int n_heads = 4;
    const int head_dim = 128;
    const int nblocks = head_dim / 32;
    const int total_blocks = n_heads * seq_len * nblocks;

    // Generate random Q8_0 data
    ggml_cuda_fattn_i8qk::block_q8_0_cuda * h_K_q8 =
        (ggml_cuda_fattn_i8qk::block_q8_0_cuda *)malloc(total_blocks * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda));
    for (int i = 0; i < total_blocks; ++i) {
        h_K_q8[i].d = 0.1f + 0.01f * (rand() / (float)RAND_MAX);
        for (int j = 0; j < 32; ++j) {
            h_K_q8[i].qs[j] = (int8_t)(rand() % 200 - 100);
        }
    }

    // CPU reference: dequant → mean → quant
    float * h_K_f32 = (float *)malloc(n_heads * seq_len * head_dim * sizeof(float));
    cpu_q8_to_float(h_K_q8, h_K_f32, seq_len, n_heads, head_dim);

    float * h_k_mean_cpu = (float *)malloc(n_heads * head_dim * sizeof(float));
    for (int h = 0; h < n_heads; ++h)
        for (int c = 0; c < head_dim; ++c) {
            float sum = 0;
            for (int s = 0; s < seq_len; ++s)
                sum += h_K_f32[(h*seq_len+s)*head_dim + c];
            h_k_mean_cpu[h*head_dim+c] = sum / seq_len;
        }

    int8_t * h_k_int8_cpu = (int8_t *)malloc(seq_len * n_heads * head_dim * sizeof(int8_t));
    float * h_k_scale_cpu = (float *)malloc(((seq_len+63)/64) * n_heads * sizeof(float));
    const int n_tiles = (seq_len + 63) / 64;
    for (int t = 0; t < n_tiles; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float amax = 0;
            for (int s = t*64; s < min((t+1)*64, seq_len); ++s)
                for (int c = 0; c < head_dim; ++c)
                    amax = fmaxf(amax, fabsf(h_K_f32[(h*seq_len+s)*head_dim+c] - h_k_mean_cpu[h*head_dim+c]));
            float scale = amax / 127.0f;
            float inv = (scale > 1e-10f) ? 1.0f/scale : 0.0f;
            h_k_scale_cpu[t*n_heads+h] = scale;
            for (int s = t*64; s < min((t+1)*64, seq_len); ++s)
                for (int c = 0; c < head_dim; ++c)
                    h_k_int8_cpu[(s*n_heads+h)*head_dim+c] =
                        (int8_t)roundf((h_K_f32[(h*seq_len+s)*head_dim+c] - h_k_mean_cpu[h*head_dim+c]) * inv);
        }
    }

    // GPU
    ggml_cuda_fattn_i8qk::block_q8_0_cuda * d_K_q8;
    float * d_k_mean; int8_t * d_k_int8; float * d_k_scale;
    cudaMalloc(&d_K_q8, total_blocks * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda));
    cudaMalloc(&d_k_mean, n_heads * head_dim * sizeof(float));
    cudaMalloc(&d_k_int8, seq_len * n_heads * head_dim * sizeof(int8_t));
    cudaMalloc(&d_k_scale, n_tiles * n_heads * sizeof(float));
    cudaMemcpy(d_K_q8, h_K_q8, total_blocks * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda), cudaMemcpyHostToDevice);

    ggml_cuda_fattn_i8qk::k_quant_buffers kbuf = {d_k_mean, d_k_int8, d_k_scale};
    ggml_cuda_fattn_i8qk::launch_k_quant<true>(d_K_q8, kbuf, seq_len, n_heads, head_dim);
    cudaDeviceSynchronize();

    float * h_k_mean_gpu = (float *)malloc(n_heads * head_dim * sizeof(float));
    int8_t * h_k_int8_gpu = (int8_t *)malloc(seq_len * n_heads * head_dim * sizeof(int8_t));
    float * h_k_scale_gpu = (float *)malloc(n_tiles * n_heads * sizeof(float));
    cudaMemcpy(h_k_mean_gpu, d_k_mean, n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_int8_gpu, d_k_int8, seq_len * n_heads * head_dim * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_scale_gpu, d_k_scale, n_tiles * n_heads * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    int mismatches = 0;
    for (int i = 0; i < seq_len * n_heads * head_dim; ++i)
        if (h_k_int8_cpu[i] != h_k_int8_gpu[i]) mismatches++;

    float max_scale_err = 0;
    for (int i = 0; i < n_tiles * n_heads; ++i)
        max_scale_err = fmaxf(max_scale_err, fabsf(h_k_scale_cpu[i] - h_k_scale_gpu[i]));

    // Round-trip error
    float max_rt = 0;
    for (int t = 0; t < n_tiles; ++t)
        for (int h = 0; h < n_heads; ++h)
            for (int s = t*64; s < min((t+1)*64, seq_len); ++s)
                for (int c = 0; c < head_dim; ++c) {
                    float recon = (float)h_k_int8_gpu[(s*n_heads+h)*head_dim+c] * h_k_scale_gpu[t*n_heads+h]
                                + h_k_mean_gpu[h*head_dim+c];
                    max_rt = fmaxf(max_rt, fabsf(recon - h_K_f32[(h*seq_len+s)*head_dim+c]));
                }

    printf("TEST 1: K quantization (Q8_0 → INT8)\n");
    printf("  INT8 mismatches: %d / %d\n", mismatches, seq_len*n_heads*head_dim);
    printf("  Max scale error: %.8f\n", max_scale_err);
    printf("  Max round-trip error: %.6f\n", max_rt);
    printf("  %s\n\n", (mismatches < 100 && max_rt < 0.1f) ? "PASS" : "FAIL");

    cudaFree(d_K_q8); cudaFree(d_k_mean); cudaFree(d_k_int8); cudaFree(d_k_scale);
    free(h_K_q8); free(h_K_f32); free(h_k_mean_cpu); free(h_k_int8_cpu); free(h_k_scale_cpu);
    free(h_k_mean_gpu); free(h_k_int8_gpu); free(h_k_scale_gpu);

    return (mismatches < 100 && max_rt < 0.1f) ? 0 : 1;
}

// =====================================================================
// Test 2: Full attention (Q8_0 KV) vs CPU reference
// =====================================================================
static int test_full_attention_q8() {
    const int seq_q = 128;
    const int seq_k = 128;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 128;
    const float sm_scale = 1.0f / sqrtf(head_dim);

    // Generate random data
    half2 * h_Q = (half2 *)malloc(n_heads * seq_q * (head_dim/2) * sizeof(half2));
    for (int i = 0; i < n_heads * seq_q * head_dim/2; ++i)
        h_Q[i] = __float2half2_rn(make_float2(
            (rand()/(float)RAND_MAX - 0.5f) * 2.0f,
            (rand()/(float)RAND_MAX - 0.5f) * 2.0f));

    ggml_cuda_fattn_i8qk::block_q8_0_cuda * h_K_q8 =
        (ggml_cuda_fattn_i8qk::block_q8_0_cuda *)malloc(n_kv_heads * seq_k * (head_dim/32) * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda));
    ggml_cuda_fattn_i8qk::block_q8_0_cuda * h_V_q8 =
        (ggml_cuda_fattn_i8qk::block_q8_0_cuda *)malloc(n_kv_heads * seq_k * (head_dim/32) * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda));
    for (int i = 0; i < n_kv_heads * seq_k * head_dim/32; ++i) {
        h_K_q8[i].d = 0.1f;
        h_V_q8[i].d = 0.1f;
        for (int j = 0; j < 32; ++j) {
            h_K_q8[i].qs[j] = (int8_t)(rand() % 200 - 100);
            h_V_q8[i].qs[j] = (int8_t)(rand() % 200 - 100);
        }
    }

    // CPU reference
    float * h_K_f32 = (float *)malloc(n_kv_heads * seq_k * head_dim * sizeof(float));
    float * h_V_f32 = (float *)malloc(n_kv_heads * seq_k * head_dim * sizeof(float));
    cpu_q8_to_float(h_K_q8, h_K_f32, seq_k, n_kv_heads, head_dim);
    cpu_q8_to_float(h_V_q8, h_V_f32, seq_k, n_kv_heads, head_dim);

    float * h_Q_f32 = (float *)malloc(n_heads * seq_q * head_dim * sizeof(float));
    for (int i = 0; i < n_heads * seq_q * head_dim/2; ++i) {
        float2 f = __half22float2(h_Q[i]);
        h_Q_f32[i*2] = f.x; h_Q_f32[i*2+1] = f.y;
    }

    float * h_O_cpu = (float *)malloc(n_heads * seq_q * head_dim * sizeof(float));
    for (int h = 0; h < n_heads; ++h) {
        cpu_attention_ref(
            h_Q_f32 + h*seq_q*head_dim,
            h_K_f32 + h*seq_k*head_dim,
            h_V_f32 + h*seq_k*head_dim,
            h_O_cpu + h*seq_q*head_dim,
            seq_q, seq_k, head_dim, head_dim, sm_scale);
    }

    // GPU
    half2 * d_Q;
    ggml_cuda_fattn_i8qk::block_q8_0_cuda * d_K_q8, * d_V_q8;
    half2 * d_O;
    cudaMalloc(&d_Q, n_heads * seq_q * (head_dim/2) * sizeof(half2));
    cudaMalloc(&d_K_q8, n_kv_heads * seq_k * (head_dim/32) * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda));
    cudaMalloc(&d_V_q8, n_kv_heads * seq_k * (head_dim/32) * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda));
    cudaMalloc(&d_O, n_heads * seq_q * (head_dim/2) * sizeof(half2));
    cudaMemcpy(d_Q, h_Q, n_heads * seq_q * (head_dim/2) * sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_q8, h_K_q8, n_kv_heads * seq_k * (head_dim/32) * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_q8, h_V_q8, n_kv_heads * seq_k * (head_dim/32) * sizeof(ggml_cuda_fattn_i8qk::block_q8_0_cuda), cudaMemcpyHostToDevice);

    // Allocate workspace and run
    ggml_cuda_fattn_i8qk::i8qk_workspace ws = ggml_cuda_fattn_i8qk::i8qk_alloc(seq_k, n_kv_heads, head_dim);
    ggml_cuda_fattn_i8qk::flash_attn_i8qk_q8(
        d_Q, d_K_q8, d_V_q8, d_O,
        seq_q, seq_k, n_heads, n_kv_heads, head_dim, sm_scale, ws);
    cudaDeviceSynchronize();

    half2 * h_O_gpu = (half2 *)malloc(n_heads * seq_q * (head_dim/2) * sizeof(half2));
    cudaMemcpy(h_O_gpu, d_O, n_heads * seq_q * (head_dim/2) * sizeof(half2), cudaMemcpyDeviceToHost);

    // Compare
    float max_err = 0, mean_err = 0;
    int total = n_heads * seq_q * head_dim;
    for (int i = 0; i < n_heads * seq_q * head_dim/2; ++i) {
        float2 gpu = __half22float2(h_O_gpu[i]);
        mean_err += fabsf(gpu.x - h_O_cpu[i*2]) + fabsf(gpu.y - h_O_cpu[i*2+1]);
        max_err = fmaxf(max_err, fmaxf(fabsf(gpu.x - h_O_cpu[i*2]), fabsf(gpu.y - h_O_cpu[i*2+1])));
    }
    mean_err /= total;

    printf("TEST 2: Full attention (Q8_0 KV) vs CPU\n");
    printf("  Max error: %.6f\n", max_err);
    printf("  Mean error: %.8f\n", mean_err);
    printf("  %s\n\n", mean_err < 0.05f ? "PASS" : "FAIL");

    ggml_cuda_fattn_i8qk::i8qk_free(ws);
    cudaFree(d_Q); cudaFree(d_K_q8); cudaFree(d_V_q8); cudaFree(d_O);
    free(h_Q); free(h_K_q8); free(h_V_q8); free(h_K_f32); free(h_V_f32); free(h_Q_f32); free(h_O_cpu); free(h_O_gpu);

    return mean_err < 0.05f ? 0 : 1;
}

// =====================================================================
// Main
// =====================================================================
int main() {
    printf("=== INT8-QK FlashAttention Tests (Q8_0 KV) ===\n\n");
    int failures = 0;
    failures += test_k_quant_q8();
    failures += test_full_attention_q8();
    printf("=== %s (%d failures) ===\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures;
}
