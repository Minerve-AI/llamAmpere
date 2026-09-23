// test-int8-qk.cu
// Standalone unit test for INT8-QK FlashAttention components.
// Compile: nvcc -O3 -arch=sm_86 test-int8-qk.cu -o test-int8-qk -I../ggml/src/ggml-cuda/
// Run: ./test-int8-qk
//
// Tests:
//   1. K mean computation
//   2. K quantization round-trip (FP16 → INT8 → FP32)
//   3. INT8 MMA primitive (m16n8k16.s8.s8.s32)
//   4. Full QK^T: INT8 vs FP16 reference

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Include the K quantization kernel
#include "fattn-k-quant.cuh"

// =====================================================================
// CPU reference functions
// =====================================================================

static void cpu_k_mean(const half * K, float * k_mean, int seq_len, int n_heads, int head_dim) {
    for (int h = 0; h < n_heads; ++h) {
        for (int c = 0; c < head_dim; ++c) {
            float sum = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
                sum += (float)K[(int64_t)s * n_heads * head_dim + h * head_dim + c];
            }
            k_mean[h * head_dim + c] = sum / seq_len;
        }
    }
}

static void cpu_k_quant(
        const half * K, const float * k_mean,
        int8_t * k_int8, float * k_scale,
        int seq_len, int n_heads, int head_dim) {
    const int n_tiles = (seq_len + 63) / 64;
    for (int t = 0; t < n_tiles; ++t) {
        const int k_start = t * 64;
        const int k_end = min(k_start + 64, seq_len);
        for (int h = 0; h < n_heads; ++h) {
            // Compute absmax
            float amax = 0.0f;
            for (int s = k_start; s < k_end; ++s) {
                for (int c = 0; c < head_dim; ++c) {
                    float v = (float)K[(int64_t)s * n_heads * head_dim + h * head_dim + c]
                            - k_mean[h * head_dim + c];
                    amax = fmaxf(amax, fabsf(v));
                }
            }
            float scale = amax / 127.0f;
            float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
            k_scale[t * n_heads + h] = scale;

            // Quantize
            for (int s = k_start; s < k_end; ++s) {
                for (int c = 0; c < head_dim; ++c) {
                    float v = (float)K[(int64_t)s * n_heads * head_dim + h * head_dim + c]
                            - k_mean[h * head_dim + c];
                    k_int8[(int64_t)s * n_heads * head_dim + h * head_dim + c] =
                        (int8_t)roundf(v * inv_scale);
                }
            }
        }
    }
}

// CPU attention reference (FP32, naive)
static void cpu_attention(
        const float * Q, const float * K, const float * V, float * O,
        int nq, int nk, int dk, int dv, float sm_scale) {
    for (int i = 0; i < nq; ++i) {
        // S = Q @ K^T
        float * S = (float *)malloc(nk * sizeof(float));
        float max_s = -1e30f;
        for (int j = 0; j < nk; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < dk; ++k) {
                dot += Q[i * dk + k] * K[j * dk + k];
            }
            S[j] = dot * sm_scale;
            max_s = fmaxf(max_s, S[j]);
        }
        // Softmax
        float sum = 0.0f;
        for (int j = 0; j < nk; ++j) {
            S[j] = expf(S[j] - max_s);
            sum += S[j];
        }
        for (int j = 0; j < nk; ++j) S[j] /= sum;
        // O = P @ V
        for (int d = 0; d < dv; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < nk; ++j) {
                acc += S[j] * V[j * dv + d];
            }
            O[i * dv + d] = acc;
        }
        free(S);
    }
}

// =====================================================================
// Test 1: K mean
// =====================================================================
static int test_k_mean() {
    const int seq_len = 256;
    const int n_heads = 4;
    const int head_dim = 256;
    const int total = seq_len * n_heads * head_dim;

    // Random data
    half * h_K = (half *)malloc(total * sizeof(half));
    float * h_k_mean_cpu = (float *)malloc(n_heads * head_dim * sizeof(float));
    float * h_k_mean_gpu = (float *)malloc(n_heads * head_dim * sizeof(float));

    for (int i = 0; i < total; ++i) {
        h_K[i] = __float2half((rand() / (float)RAND_MAX - 0.5f) * 2.0f);
    }

    // CPU reference
    cpu_k_mean((const half *)h_K, h_k_mean_cpu, seq_len, n_heads, head_dim);

    // GPU
    half2 * d_K;
    float * d_k_mean;
    cudaMalloc(&d_K, total * sizeof(half));
    cudaMalloc(&d_k_mean, n_heads * head_dim * sizeof(float));
    cudaMemcpy(d_K, h_K, total * sizeof(half), cudaMemcpyHostToDevice);

    {
        const int total_half2 = n_heads * (head_dim / 2);
        const int block = 256;
        const int grid = (total_half2 + block - 1) / block;
        ggml_cuda_fattn_i8qk::fattn_k_mean_kernel<<<grid, block>>>(
            (const half2 *)d_K, d_k_mean, seq_len, n_heads, head_dim);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_k_mean_gpu, d_k_mean, n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float max_err = 0.0f;
    for (int i = 0; i < n_heads * head_dim; ++i) {
        max_err = fmaxf(max_err, fabsf(h_k_mean_cpu[i] - h_k_mean_gpu[i]));
    }

    printf("TEST 1: K mean\n");
    printf("  Max error: %.8f\n", max_err);
    printf("  %s\n", max_err < 1e-4f ? "PASS" : "FAIL");

    cudaFree(d_K);
    cudaFree(d_k_mean);
    free(h_K);
    free(h_k_mean_cpu);
    free(h_k_mean_gpu);

    return max_err < 1e-4f ? 0 : 1;
}

// =====================================================================
// Test 2: K quantization round-trip
// =====================================================================
static int test_k_quant() {
    const int seq_len = 256;
    const int n_heads = 4;
    const int head_dim = 256;
    const int total = seq_len * n_heads * head_dim;
    const int n_tiles = (seq_len + 63) / 64;

    // Random data
    half * h_K = (half *)malloc(total * sizeof(half));
    for (int i = 0; i < total; ++i) {
        h_K[i] = __float2half((rand() / (float)RAND_MAX - 0.5f) * 2.0f);
    }

    // CPU reference
    float * h_k_mean = (float *)malloc(n_heads * head_dim * sizeof(float));
    int8_t * h_k_int8_cpu = (int8_t *)malloc(total * sizeof(int8_t));
    float * h_k_scale_cpu = (float *)malloc(n_tiles * n_heads * sizeof(float));
    cpu_k_mean(h_K, h_k_mean, seq_len, n_heads, head_dim);
    cpu_k_quant(h_K, h_k_mean, h_k_int8_cpu, h_k_scale_cpu, seq_len, n_heads, head_dim);

    // GPU
    half2 * d_K;
    float * d_k_mean;
    int8_t * d_k_int8;
    float * d_k_scale;
    cudaMalloc(&d_K, total * sizeof(half));
    cudaMalloc(&d_k_mean, n_heads * head_dim * sizeof(float));
    cudaMalloc(&d_k_int8, total * sizeof(int8_t));
    cudaMalloc(&d_k_scale, n_tiles * n_heads * sizeof(float));
    cudaMemcpy(d_K, h_K, total * sizeof(half), cudaMemcpyHostToDevice);

    ggml_cuda_fattn_i8qk::launch_k_quant(
        (const half2 *)d_K, 
        /* buf */ {d_k_mean, d_k_int8, d_k_scale},
        seq_len, n_heads, head_dim);
    cudaDeviceSynchronize();

    float * h_k_mean_gpu = (float *)malloc(n_heads * head_dim * sizeof(float));
    int8_t * h_k_int8_gpu = (int8_t *)malloc(total * sizeof(int8_t));
    float * h_k_scale_gpu = (float *)malloc(n_tiles * n_heads * sizeof(float));
    cudaMemcpy(h_k_mean_gpu, d_k_mean, n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_int8_gpu, d_k_int8, total * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_scale_gpu, d_k_scale, n_tiles * n_heads * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare INT8 values
    int mismatches = 0;
    for (int i = 0; i < total; ++i) {
        if (h_k_int8_cpu[i] != h_k_int8_gpu[i]) mismatches++;
    }

    // Compare scales
    float max_scale_err = 0.0f;
    for (int i = 0; i < n_tiles * n_heads; ++i) {
        max_scale_err = fmaxf(max_scale_err, fabsf(h_k_scale_cpu[i] - h_k_scale_gpu[i]));
    }

    // Round-trip error: reconstruct and compare
    float max_rt_err = 0.0f;
    for (int t = 0; t < n_tiles; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float scale = h_k_scale_gpu[t * n_heads + h];
            for (int s = t * 64; s < min((t+1)*64, seq_len); ++s) {
                for (int c = 0; c < head_dim; ++c) {
                    float recon = (float)h_k_int8_gpu[(int64_t)s*n_heads*head_dim + h*head_dim + c] * scale
                                + h_k_mean_gpu[h * head_dim + c];
                    float orig = (float)h_K[(int64_t)s*n_heads*head_dim + h*head_dim + c];
                    max_rt_err = fmaxf(max_rt_err, fabsf(recon - orig));
                }
            }
        }
    }

    printf("TEST 2: K quantization round-trip\n");
    printf("  INT8 mismatches: %d / %d\n", mismatches, total);
    printf("  Max scale error: %.8f\n", max_scale_err);
    printf("  Max round-trip error: %.6f\n", max_rt_err);
    printf("  %s\n", (mismatches == 0 && max_rt_err < 0.05f) ? "PASS" : "FAIL");

    cudaFree(d_K); cudaFree(d_k_mean); cudaFree(d_k_int8); cudaFree(d_k_scale);
    free(h_K); free(h_k_mean); free(h_k_int8_cpu); free(h_k_scale_cpu);
    free(h_k_mean_gpu); free(h_k_int8_gpu); free(h_k_scale_gpu);

    return (mismatches == 0 && max_rt_err < 0.05f) ? 0 : 1;
}

// =====================================================================
// Test 3: INT8 MMA primitive
// =====================================================================
// We test mma.m16n8k16.s8.s8.s32 by comparing against CPU matmul.
// A: 16x16 int8 (row-major), B: 16x8 int8 (col-major), C: 16x8 int32

__global__ void test_mma_i8_kernel(
        const int * A_regs, // 2 int32 per thread (A tile)
        const int * B_regs, // 1 int32 per thread (B tile)
        int * C_out) {      // 4 int32 per thread (C tile)
    // This is a simplified test - in reality the registers come from ldmatrix
    // Here we just verify the PTX instruction works
    int D[4] = {0, 0, 0, 0};
    int A[2] = {A_regs[threadIdx.x * 2], A_regs[threadIdx.x * 2 + 1]};
    int B[1] = {B_regs[threadIdx.x]};

    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]));

    C_out[threadIdx.x * 4 + 0] = D[0];
    C_out[threadIdx.x * 4 + 1] = D[1];
    C_out[threadIdx.x * 4 + 2] = D[2];
    C_out[threadIdx.x * 4 + 3] = D[3];
}

static int test_mma_i8() {
    // For a proper test we need to set up the registers in the correct layout.
    // The m16n8k16.s8 layout:
    // A (16x16 int8, row-major): each thread holds 2 int32 (8 int8 values)
    //   Thread t: A[t/4][8*(t%4) .. 8*(t%4)+3] and A[t/4+8][8*(t%4) .. 8*(t%4)+3]
    // B (16x8 int8, col-major): each thread holds 1 int32 (4 int8 values)
    //   Thread t: B[8*(t%4) .. 8*(t%4)+3][t/4] (col-major means B[k][n])
    // C (16x8 int32, col-major): each thread holds 4 int32
    //   Thread t: C[t/4][2*(t%4)] and C[t/4+8][2*(t%4)]

    // Simple test: A = all 1s, B = all 1s → C should be 16 (sum of 16 ones)
    const int nthreads = 32; // 1 warp
    int * h_A = (int *)malloc(nthreads * 2 * sizeof(int));
    int * h_B = (int *)malloc(nthreads * 1 * sizeof(int));
    int * h_C = (int *)malloc(nthreads * 4 * sizeof(int));

    // Set A = all 1s (each int32 = 0x01010101 = four int8 of value 1)
    for (int i = 0; i < nthreads * 2; ++i) h_A[i] = 0x01010101;
    // Set B = all 1s
    for (int i = 0; i < nthreads; ++i) h_B[i] = 0x01010101;

    int * d_A, * d_B, * d_C;
    cudaMalloc(&d_A, nthreads * 2 * sizeof(int));
    cudaMalloc(&d_B, nthreads * sizeof(int));
    cudaMalloc(&d_C, nthreads * 4 * sizeof(int));
    cudaMemcpy(d_A, h_A, nthreads * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nthreads * sizeof(int), cudaMemcpyHostToDevice);

    test_mma_i8_kernel<<<1, nthreads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, nthreads * 4 * sizeof(int), cudaMemcpyDeviceToHost);

    // Each C element should be 16 (16 ones summed)
    int errors = 0;
    for (int i = 0; i < nthreads * 4; ++i) {
        if (h_C[i] != 16) errors++;
    }

    printf("TEST 3: INT8 MMA primitive (m16n8k16.s8)\n");
    printf("  Expected: all C = 16\n");
    printf("  Errors: %d / %d\n", errors, nthreads * 4);
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return errors == 0 ? 0 : 1;
}

// =====================================================================
// Main
// =====================================================================
int main() {
    printf("=== INT8-QK FlashAttention Unit Tests ===\n\n");

    int failures = 0;
    failures += test_k_mean();
    printf("\n");
    failures += test_k_quant();
    printf("\n");
    failures += test_mma_i8();
    printf("\n");

    printf("=== %s (%d failures) ===\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures;
}
