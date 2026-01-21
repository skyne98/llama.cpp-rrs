/**
 * RRS (Rotated Runtime Smooth) CUDA Implementation
 *
 * Clean implementation supporting TCQ4 RRS W4A4 path only.
 * Based on validated kernels from rrs_validation/
 *
 * Key algorithms (from reference.py):
 * - Activation: FWHT -> scale = max(|x|) -> q = round(x * 7 / scale)
 * - Dequant: x_approx = q * (scale / 7)
 * - GEMM: C += int_dot * (a_scale/7) * b_scale + sum_a * (a_scale/7) * b_zero
 */

#include "rrs.cuh"
#include "tcq4_k32.cuh"
#include "common.cuh"
#include <cuda_fp16.h>
#include <unordered_map>
#include <string>
#include <cstdio>

// Debug kernel to dump first tile's values
__global__ void debug_dump_tile_kernel(const block_tcq4_tile* tiles, float* out_S, int8_t* out_sc, uint8_t* out_bytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const block_tcq4_tile* tile = &tiles[0];
        // Copy S values (8 half -> 8 float)
        for (int i = 0; i < 8; i++) {
            out_S[i] = __half2float(tile->S[i]);
        }
        // Copy sc[0][*] (first channel's scale codes)
        for (int i = 0; i < 8; i++) {
            out_sc[i] = tile->sc[0][i];
        }
        // Copy first 16 bytes of tiles[0]
        for (int i = 0; i < 16; i++) {
            out_bytes[i] = tile->tiles[0][i];
        }
    }
}

void debug_dump_tile(const void* d_tiles, cudaStream_t stream) {
    float* d_S;
    int8_t* d_sc;
    uint8_t* d_bytes;
    cudaMalloc(&d_S, 8 * sizeof(float));
    cudaMalloc(&d_sc, 8 * sizeof(int8_t));
    cudaMalloc(&d_bytes, 16 * sizeof(uint8_t));
    
    debug_dump_tile_kernel<<<1, 1, 0, stream>>>((const block_tcq4_tile*)d_tiles, d_S, d_sc, d_bytes);
    cudaStreamSynchronize(stream);
    
    float h_S[8];
    int8_t h_sc[8];
    uint8_t h_bytes[16];
    cudaMemcpy(h_S, d_S, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sc, d_sc, 8 * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bytes, d_bytes, 16 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "  [DEBUG] Tile[0].S: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            h_S[0], h_S[1], h_S[2], h_S[3], h_S[4], h_S[5], h_S[6], h_S[7]);
    fprintf(stderr, "  [DEBUG] Tile[0].sc[0]: %d %d %d %d %d %d %d %d\n",
            h_sc[0], h_sc[1], h_sc[2], h_sc[3], h_sc[4], h_sc[5], h_sc[6], h_sc[7]);
    fprintf(stderr, "  [DEBUG] Tile[0].tiles[0][0:16]: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
            h_bytes[0], h_bytes[1], h_bytes[2], h_bytes[3], h_bytes[4], h_bytes[5], h_bytes[6], h_bytes[7],
            h_bytes[8], h_bytes[9], h_bytes[10], h_bytes[11], h_bytes[12], h_bytes[13], h_bytes[14], h_bytes[15]);
    
    cudaFree(d_S);
    cudaFree(d_sc);
    cudaFree(d_bytes);
}

// ============================================================================
// Configuration
// ============================================================================

#define TCQ4_USE_RRS_W4A4 1
#define TCQ4_USE_TENSOR_CORES 1

// ============================================================================
// Channel Permutation Registry
// ============================================================================

static std::unordered_map<std::string, int32_t*> g_perm_registry;
static bool g_reorder_enabled = true;

void ggml_cuda_rrs_register_perm(const char* tensor_name, const int32_t* h_perm, int K) {
    std::string key(tensor_name);
    
    // Free existing if present
    if (g_perm_registry.count(key)) {
        cudaFree(g_perm_registry[key]);
    }
    
    // Allocate and copy to device
    int32_t* d_perm = nullptr;
    cudaMalloc(&d_perm, K * sizeof(int32_t));
    cudaMemcpy(d_perm, h_perm, K * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    g_perm_registry[key] = d_perm;
}

const int32_t* ggml_cuda_rrs_get_perm(const char* tensor_name) {
    if (!g_reorder_enabled) return nullptr;
    
    std::string key(tensor_name);
    auto it = g_perm_registry.find(key);
    return (it != g_perm_registry.end()) ? it->second : nullptr;
}

bool ggml_cuda_rrs_has_perm(const char* tensor_name) {
    if (!g_reorder_enabled) return false;
    return g_perm_registry.count(std::string(tensor_name)) > 0;
}

void ggml_cuda_rrs_clear_perms() {
    for (auto& kv : g_perm_registry) {
        cudaFree(kv.second);
    }
    g_perm_registry.clear();
}

void ggml_cuda_rrs_set_reorder_enabled(bool enabled) {
    g_reorder_enabled = enabled;
}

bool ggml_cuda_rrs_get_reorder_enabled() {
    return g_reorder_enabled;
}

// ============================================================================
// FWHT Kernels
// ============================================================================

__global__ void fwht_kernel(const float* __restrict__ x, float* __restrict__ y, int n, int batch_size) {
    extern __shared__ float smem[];
    
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int tid = threadIdx.x;
    const float* x_row = x + row * n;
    float* y_row = y + row * n;
    
    // Load to shared memory
    for (int i = tid; i < n; i += blockDim.x) {
        smem[i] = x_row[i];
    }
    __syncthreads();
    
    // FWHT butterfly stages
    for (int h = 1; h < n; h *= 2) {
        for (int i = tid; i < n / 2; i += blockDim.x) {
            int grp = i / h;
            int pos = i % h;
            int j = grp * h * 2 + pos;
            int k = j + h;
            
            float a = smem[j];
            float b = smem[k];
            smem[j] = a + b;
            smem[k] = a - b;
        }
        __syncthreads();
    }
    
    // Normalize and store
    float norm = 1.0f / sqrtf((float)n);
    for (int i = tid; i < n; i += blockDim.x) {
        y_row[i] = smem[i] * norm;
    }
}

void ggml_cuda_rrs_fwht(const float* x, float* y, int n, int batch_size, cudaStream_t stream) {
    int threads = min(256, n / 2);
    size_t smem = n * sizeof(float);
    fwht_kernel<<<batch_size, threads, smem, stream>>>(x, y, n, batch_size);
}

void ggml_cuda_tcq4_fwht_step256(const float* x, float* y, int n, int batch_size, cudaStream_t stream) {
    // For TCQ4, FWHT is done in 256-element chunks
    int num_chunks = n / 256;
    int total_rows = batch_size * num_chunks;
    
    // Reshape: treat each 256-element chunk as a separate row
    int threads = 128;
    size_t smem = 256 * sizeof(float);
    
    // Launch kernel that processes chunks
    fwht_kernel<<<total_rows, threads, smem, stream>>>(x, y, 256, total_rows);
}

// ============================================================================
// RRS Quantization (stub - actual implementation in tcq4_k32.cu)
// ============================================================================

void ggml_cuda_rrs_quantize_act(const float* x, void* y, int n, int batch_size, cudaStream_t stream) {
    // Delegate to TCQ4 RRS quantization
    tcq4_rrs_fwht_quantize(x, y, n, batch_size, stream);
}

void ggml_cuda_rrs_fwht_quantize(const float* x, void* y, int n, int batch_size, cudaStream_t stream) {
    // Delegate to TCQ4 RRS quantization (includes FWHT)
    tcq4_rrs_fwht_quantize(x, y, n, batch_size, stream);
}

// ============================================================================
// GEMM Kernels (stubs - not used in RRS W4A4 path)
// ============================================================================

void ggml_cuda_rrs_gemm_q4q4(
    const void* A, const void* B, float* C,
    int M, int N, int K,
    const half* scales_A, const half* mins_A,
    const half* scales_B, const half* mins_B,
    cudaStream_t stream
) {
    // Not used - TCQ4 RRS uses tcq4_rrs_gemm_imma directly
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    (void)scales_A; (void)mins_A; (void)scales_B; (void)mins_B; (void)stream;
}

void ggml_cuda_rrs_gemm_q4_via_q8(
    const void* A_q4, const void* B_q4, float* C,
    int M, int N, int K,
    const half* scales_A, const half* mins_A,
    const half* scales_B, const half* mins_B,
    cudaStream_t stream
) {
    // Not used - TCQ4 RRS uses tcq4_rrs_gemm_imma directly
    (void)A_q4; (void)B_q4; (void)C; (void)M; (void)N; (void)K;
    (void)scales_A; (void)mins_A; (void)scales_B; (void)mins_B; (void)stream;
}

// ============================================================================
// Main Entry Point: RRS Mul Mat
// ============================================================================

bool ggml_cuda_supports_rrs(const ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_TCQ4_K32;
}

void ggml_cuda_rrs_mul_mat(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,  // weights (TCQ4_K32)
    const ggml_tensor* src1,  // activations (F32)
    ggml_tensor* dst
) {
    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // N
    const int64_t ne10 = src1->ne[0];  // K
    const int64_t ne11 = src1->ne[1];  // M
    
    const int M = ne11;
    const int N = ne01;
    const int K = ne00;
    
    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(src0->type == GGML_TYPE_TCQ4_K32);
    GGML_ASSERT(K % TCQ4_TILE_K == 0);
    GGML_ASSERT(N % TCQ4_TILE_CHANNELS == 0);  // N must be multiple of 8 for tile format
    
    const int num_k_tiles = K / TCQ4_TILE_K;
    const int n_tiles = N / TCQ4_TILE_CHANNELS;  // Number of N-tiles (8 channels each)
    
    cudaStream_t stream = ctx.stream();
    
    // Get channel permutation if available
    const int32_t* d_perm = nullptr;
    if (src0->name[0] != '\0') {
        d_perm = ggml_cuda_rrs_get_perm(src0->name);
    }
    
    // Call counting (set TCQ4_COUNT=1 env var to enable)
    static bool count_enabled = (getenv("TCQ4_COUNT") != nullptr);
    static int call_count = 0;
    static int token_count = 0;
    static int last_M = -1;
    
    if (count_enabled) {
        call_count++;
        // Detect new token: M changes or we see M=1 after M>1
        if (M != last_M && (last_M > 1 || last_M == -1)) {
            if (token_count > 0) {
                fprintf(stderr, "[TCQ4-COUNT] Token %d: %d mul_mat calls\n", token_count, call_count - 1);
            }
            token_count++;
            call_count = 1;
        }
        last_M = M;
    }
    
    // Debug output (disabled for production - set to 1 to enable)
    static bool first_call = true;
    if (first_call) {
        fprintf(stderr, "[TCQ4-RRS-W4A4] mul_mat M=%d N=%d K=%d perm=%s count=%s\n",
                M, N, K, d_perm ? "yes" : "no", count_enabled ? "ON" : "off");
        first_call = false;
    }
    (void)n_tiles;
    
#if TCQ4_USE_TENSOR_CORES
    // Tensor core path
    if (M == 1) {
        // M=1: Use fused v2d kernel (2.8-3.6x faster than separate quant + IMMA)
        // Fuses: permutation + FWHT + quantize + GEMV in single kernel
        tcq4_rrs_fused_gemv((const float*)src1->data, d_perm, src0->data, 
                           (float*)dst->data, N, K, stream);
    } else {
        // M>1: Use separate quantize + IMMA GEMM (tensor cores efficient for batched)
        size_t rrs_size = M * num_k_tiles * sizeof(block_rrs_int4_tc);
        size_t rrs_actual;
        void* d_act_rrs = ctx.pool().alloc(rrs_size, &rrs_actual);
        
        // Fused (permutation +) FWHT + Runtime Smooth + INT4 quantize
        tcq4_rrs_perm_fwht_quantize_tc((const float*)src1->data, d_act_rrs, d_perm, K, M, stream);
        
        // Check for CUDA errors after activation quantization
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[TCQ4-RRS] CUDA error after activation quant: %s\n", cudaGetErrorString(err));
        }
        
        // Dispatch GEMM with tensor cores
        tcq4_rrs_gemm_imma(d_act_rrs, src0->data, (float*)dst->data, M, N, K, stream);
    }
#else /* !TCQ4_USE_TENSOR_CORES */
    // Scalar fallback path: uses block_rrs_int4 (simpler, no group sums)
    size_t rrs_size = M * num_k_tiles * sizeof(block_rrs_int4);
    size_t rrs_actual;
    void* d_act_rrs = ctx.pool().alloc(rrs_size, &rrs_actual);
    
    // Fused FWHT + Runtime Smooth + INT4 quantize
    if (d_perm) {
        tcq4_rrs_perm_fwht_quantize((const float*)src1->data, d_act_rrs, d_perm, K, M, stream);
    } else {
        tcq4_rrs_fwht_quantize((const float*)src1->data, d_act_rrs, K, M, stream);
    }
    
    // Dispatch GEMM/GEMV with scalar kernels
    if (M == 1) {
        tcq4_rrs_gemv(d_act_rrs, src0->data, (float*)dst->data, N, K, stream);
    } else {
        tcq4_rrs_gemm(d_act_rrs, src0->data, (float*)dst->data, M, N, K, stream);
    }
#endif
}

// ============================================================================
// Benchmarking
// ============================================================================

void ggml_cuda_rrs_benchmark(int M, int N, int K, int iterations, RRSBenchmarkResult* result) {
    result->M = M;
    result->N = N;
    result->K = K;
    result->int4_wmma_time_ms = 0.0f;
    result->q8_repack_time_ms = 0.0f;
    result->fwht_time_ms = 0.0f;
    result->quantize_time_ms = 0.0f;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    const int num_k_tiles = K / TCQ4_TILE_K;
    const int n_tiles = (N + TCQ4_TILE_CHANNELS - 1) / TCQ4_TILE_CHANNELS;
    
    // Allocate test data
    float* d_act_fp32;
    void* d_act_rrs;
    void* d_weights;
    float* d_output;
    
    cudaMalloc(&d_act_fp32, M * K * sizeof(float));
    cudaMalloc(&d_act_rrs, M * num_k_tiles * sizeof(block_rrs_int4_tc));
    // Tile format: [n_tile][k_tile] where each tile is 1184 bytes (8 channels × 256 K)
    cudaMalloc(&d_weights, n_tiles * num_k_tiles * sizeof(block_tcq4_tile));
    cudaMalloc(&d_output, M * N * sizeof(float));
    
    // Initialize with random data
    std::vector<float> h_act(M * K);
    for (int i = 0; i < M * K; i++) h_act[i] = (float)(i % 100) * 0.01f - 0.5f;
    cudaMemcpy(d_act_fp32, h_act.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warmup
    tcq4_rrs_fwht_quantize_tc(d_act_fp32, d_act_rrs, K, M, stream);
    tcq4_rrs_gemm_imma(d_act_rrs, d_weights, d_output, M, N, K, stream);
    cudaStreamSynchronize(stream);
    
    // Benchmark FWHT + quantize
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        tcq4_rrs_fwht_quantize_tc(d_act_fp32, d_act_rrs, K, M, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result->quantize_time_ms, start, stop);
    result->quantize_time_ms /= iterations;
    
    // Benchmark GEMM
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        tcq4_rrs_gemm_imma(d_act_rrs, d_weights, d_output, M, N, K, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result->int4_wmma_time_ms, start, stop);
    result->int4_wmma_time_ms /= iterations;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_act_fp32);
    cudaFree(d_act_rrs);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

void ggml_cuda_rrs_print_benchmark(const RRSBenchmarkResult* result) {
    printf("RRS Benchmark Results (M=%d, N=%d, K=%d):\n", result->M, result->N, result->K);
    printf("  FWHT + Quantize: %.3f ms\n", result->quantize_time_ms);
    printf("  INT4 GEMM:       %.3f ms\n", result->int4_wmma_time_ms);
    
    // Compute TFLOPS
    double ops = 2.0 * result->M * result->N * result->K;
    double tflops = ops / (result->int4_wmma_time_ms * 1e-3) / 1e12;
    printf("  Throughput:      %.2f TFLOPS\n", tflops);
}

// ============================================================================
// Test Function
// ============================================================================

extern "C" void ggml_cuda_rrs_test(void) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("RRS CUDA Test on: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);
    
    RRSBenchmarkResult res;
    ggml_cuda_rrs_benchmark(128, 2048, 2048, 50, &res);
    ggml_cuda_rrs_print_benchmark(&res);
}