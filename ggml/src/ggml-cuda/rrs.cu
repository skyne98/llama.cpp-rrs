#include "rrs.cuh"
#include "common.cuh"

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace nvcuda;

// ============================================================================ 
// Shared Helpers
// ============================================================================ 

__device__ __forceinline__ void fwht_butterfly(float& a, float& b) {
    float t = a;
    a = t + b;
    b = t - b;
}

__device__ __forceinline__ void unpack_scales_mins_k4_cuda(const uint8_t* scales, uint8_t* sc, uint8_t* mn) {
    #pragma unroll
    for (int j = 0; j < 8; j++) get_scale_min_k4_cuda(j, scales, &sc[j], &mn[j]);
}

// ============================================================================ 
// FWHT (Fast Walsh-Hadamard Transform) CUDA Kernels
// ============================================================================ 

template<int N>
__global__ void fwht_kernel_pow2(const float* __restrict__ x, float* __restrict__ y, int batch_size) {
    extern __shared__ float smem_fwht[];
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    const float* x_row = x + batch_idx * N;
    float* y_row = y + batch_idx * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) smem_fwht[i] = x_row[i];
    __syncthreads();
    const float scale = rsqrtf((float)N);
    #pragma unroll
    for (int h = 1; h < N; h <<= 1) {
        const int stride = h << 1;
        for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
            const int block = i / h;
            const int offset = i % h;
            const int idx1 = block * stride + offset;
            const int idx2 = idx1 + h;
            fwht_butterfly(smem_fwht[idx1], smem_fwht[idx2]);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < N; i += blockDim.x) y_row[i] = smem_fwht[i] * scale;
}

__global__ void fwht_kernel_chunked(const float* __restrict__ x, float* __restrict__ y, int n, int batch_size) {
    extern __shared__ float smem_chunk[];
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    const float* x_row = x + batch_idx * n;
    float* y_row = y + batch_idx * n;
    const int step = n & -n;
    const int num_chunks = n / step;
    const float scale = rsqrtf((float)step);
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int base = chunk * step;
        for (int i = threadIdx.x; i < step; i += blockDim.x) smem_chunk[i] = x_row[base + i];
        __syncthreads();
        for (int h = 1; h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = threadIdx.x; i < step / 2; i += blockDim.x) {
                const int block = i / h;
                const int offset = i % h;
                const int idx1 = block * stride + offset;
                const int idx2 = idx1 + h;
                fwht_butterfly(smem_chunk[idx1], smem_chunk[idx2]);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < step; i += blockDim.x) y_row[base + i] = smem_chunk[i] * scale;
        __syncthreads();
    }
}

void ggml_cuda_rrs_fwht(const float* x, float* y, int n, int batch_size, cudaStream_t stream) {
    const int threads = min(256, (n + 1) / 2);
    const size_t smem_size = n * sizeof(float);
    if ((n & (n - 1)) == 0) {
        switch (n) {
            case 256:  fwht_kernel_pow2<256><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size); break;
            case 512:  fwht_kernel_pow2<512><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size); break;
            case 1024: fwht_kernel_pow2<1024><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size); break;
            case 2048: fwht_kernel_pow2<2048><<<batch_size, 512, smem_size, stream>>>(x, y, batch_size); break;
            case 4096: fwht_kernel_pow2<4096><<<batch_size, 512, smem_size, stream>>>(x, y, batch_size); break;
            case 8192: fwht_kernel_pow2<8192><<<batch_size, 512, smem_size, stream>>>(x, y, batch_size); break;
            default:   fwht_kernel_chunked<<<batch_size, threads, smem_size, stream>>>(x, y, n, batch_size); break;
        }
    } else {
        const int step = n & -n;
        fwht_kernel_chunked<<<batch_size, min(256, step/2), step * sizeof(float), stream>>>(x, y, n, batch_size);
    }
}

// ============================================================================ 
// Fused GPU RRS Transform + Quantization
// ============================================================================ 

// Quantize a 256-element block that has ALREADY been FWHT-transformed
// smem points to the 256 floats, y_row is output Q4_K blocks, block_idx is which block
__device__ __forceinline__ void quantize_block_256(float * smem, block_q4_K * y_row, int block_idx) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    __shared__ float s_scales[8];
    __shared__ float s_mins[8];
    __shared__ float s_max_scale;
    __shared__ float s_max_min;

    if (warp_id < 8) {
        const int base = warp_id * 32;
        float val = smem[base + lane];
        float vmin = val, vmax = val;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            vmin = fminf(vmin, __shfl_xor_sync(0xFFFFFFFF, vmin, mask));
            vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, mask));
        }
        if (vmin > 0) vmin = 0;
        if (lane == 0) { s_scales[warp_id] = (vmax - vmin) / 15.0f; s_mins[warp_id] = -vmin; }
    }
    __syncthreads();

    if (tid == 0) {
        float max_s = s_scales[0], max_m = s_mins[0];
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            if (s_scales[i] > max_s) max_s = s_scales[i];
            if (s_mins[i] > max_m) max_m = s_mins[i];
        }
        s_max_scale = max_s; s_max_min = max_m;
        y_row[block_idx].dm.x = __float2half(max_s / 63.0f);
        y_row[block_idx].dm.y = __float2half(max_m / 63.0f);
    }
    __syncthreads();

    if (tid == 0) {
        const float inv_s = (s_max_scale > 0.0f) ? (63.0f / s_max_scale) : 0.0f;
        const float inv_m = (s_max_min > 0.0f) ? (63.0f / s_max_min) : 0.0f;
        uint8_t ls[8], lm[8];
        for (int j = 0; j < 8; j++) {
            ls[j] = (uint8_t)(s_scales[j] * inv_s + 0.5f); if (ls[j] > 63) ls[j] = 63;
            lm[j] = (uint8_t)(s_mins[j] * inv_m + 0.5f);   if (lm[j] > 63) lm[j] = 63;
        }
        for (int j = 0; j < 4; j++) {
            y_row[block_idx].scales[j] = ls[j] | ((ls[j+4] & 0x30) << 2);
            y_row[block_idx].scales[j+4] = lm[j] | ((lm[j+4] & 0x30) << 2);
        }
        for (int j = 0; j < 4; j++) y_row[block_idx].scales[j+8] = (ls[j+4] & 0x0F) | ((lm[j+4] & 0x0F) << 4);
    }
    __syncthreads();

    if (warp_id < 4) {
        const int g_lo = warp_id * 2;
        const int g_hi = g_lo + 1;
        uint8_t sc_lo, mn_lo, sc_hi, mn_hi;
        get_scale_min_k4_cuda(g_lo, y_row[block_idx].scales, &sc_lo, &mn_lo);
        get_scale_min_k4_cuda(g_hi, y_row[block_idx].scales, &sc_hi, &mn_hi);
        const float d_lo = __half2float(y_row[block_idx].dm.x) * (float)sc_lo;
        const float dm_lo = __half2float(y_row[block_idx].dm.y) * (float)mn_lo;
        const float id_lo = (d_lo > 1e-10f) ? (1.0f / d_lo) : 0.0f;
        const float d_hi = __half2float(y_row[block_idx].dm.x) * (float)sc_hi;
        const float dm_hi = __half2float(y_row[block_idx].dm.y) * (float)mn_hi;
        const float id_hi = (d_hi > 1e-10f) ? (1.0f / d_hi) : 0.0f;
        const float val_lo = smem[g_lo * 32 + lane];
        const float val_hi = smem[g_hi * 32 + lane];
        int q_lo = (int)((val_lo + dm_lo) * id_lo + 0.5f); if (q_lo < 0) q_lo = 0; if (q_lo > 15) q_lo = 15;
        int q_hi = (int)((val_hi + dm_hi) * id_hi + 0.5f); if (q_hi < 0) q_hi = 0; if (q_hi > 15) q_hi = 15;
        y_row[block_idx].qs[warp_id * 32 + lane] = (uint8_t)(q_lo | (q_hi << 4));
    }
    __syncthreads();
}

__global__ void fwht_quantize_kernel_any(const float* __restrict__ x, void* __restrict__ vy, int n, int batch_size) {
    extern __shared__ float smem_any[];
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    const float* x_row = x + batch_idx * n;
    block_q4_K* y_row = (block_q4_K*)vy + batch_idx * (n / 256);
    
    // We assume n is a multiple of 256 for Q4_K
    const int step = n & -n;
    if (step < 256) return;
    
    for (int s = 0; s < n; s += step) {
        for (int i = threadIdx.x; i < step; i += blockDim.x) smem_any[i] = x_row[s + i];
        __syncthreads();
        // FWHT for the current power-of-2 step
        const float scale_fwht = rsqrtf((float)step);
        #pragma unroll
        for (int h = 1; h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = threadIdx.x; i < step / 2; i += blockDim.x) {
                const int b = i / h; const int offset = i % h;
                const int idx1 = b * stride + offset; const int idx2 = idx1 + h;
                fwht_butterfly(smem_any[idx1], smem_any[idx2]);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < step; i += blockDim.x) smem_any[i] *= scale_fwht;
        __syncthreads();
        
        // Quantize each 256-block within the step (FWHT already done above)
        for (int b = 0; b < step / 256; b++) {
            quantize_block_256(smem_any + b * 256, y_row, (s / 256) + b);
        }
        __syncthreads();
    }
}

void ggml_cuda_rrs_fwht_quantize(const float* x, void* y, int n, int batch_size, cudaStream_t stream) {
    if (n % 256 == 0) {
        const int step = n & -n;
        const size_t smem = max(step, 256) * sizeof(float);
        fwht_quantize_kernel_any<<<batch_size, 256, smem, stream>>>(x, y, n, batch_size);
    } else {
        fprintf(stderr, "[RRS CUDA] Error: Dimension N=%d not a multiple of 256\n", n);
    }
}

// ============================================================================ 
// GEMM Kernel (A=Q4_K, B=Q4_K)
// ============================================================================ 

__global__ void rrs_gemm_q4k_q4k_kernel(const void* __restrict__ A, const void* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
    const int col = blockIdx.x; const int row = blockIdx.y;
    if (col >= N || row >= M) return;
    const int tid = threadIdx.x; const int n_threads = blockDim.x;
    const block_q4_K* blocks_a = (const block_q4_K*)A + row * (K / 256);
    const block_q4_K* blocks_w = (const block_q4_K*)B + col * (K / 256);
    float total_sum = 0.0f;
    for (int b = tid; b < K / 256; b += n_threads) {
        const block_q4_K& ab = blocks_a[b]; const block_q4_K& wb = blocks_w[b];
        const float d_a = __half2float(ab.dm.x); const float dmin_a = __half2float(ab.dm.y);
        const float d_w = __half2float(wb.dm.x); const float dmin_w = __half2float(wb.dm.y);
        uint8_t sc_a[8], mn_a[8], sc_w[8], mn_w[8];
        unpack_scales_mins_k4_cuda(ab.scales, sc_a, mn_a); unpack_scales_mins_k4_cuda(wb.scales, sc_w, mn_w);
        for (int g = 0; g < 8; g++) {
            const int p = g / 2; const bool is_hi = (g % 2 != 0);
            const uint8_t* qs_a_ptr = ab.qs + p * 32; const uint8_t* qs_w_ptr = wb.qs + p * 32;
            int dot_qq = 0, sum_qa = 0, sum_qw = 0;
            #pragma unroll
            for (int l = 0; l < 16; l++) {
                const uint8_t pa_0 = qs_a_ptr[l], pa_1 = qs_a_ptr[l+16];
                const uint8_t pw_0 = qs_w_ptr[l], pw_1 = qs_w_ptr[l+16];
                int qa0 = (int)(is_hi ? (pa_0 >> 4) : (pa_0 & 0x0F)), qa1 = (int)(is_hi ? (pa_1 >> 4) : (pa_1 & 0x0F));
                int qw0 = (int)(is_hi ? (pw_0 >> 4) : (pw_0 & 0x0F)), qw1 = (int)(is_hi ? (pw_1 >> 4) : (pw_1 & 0x0F));
                dot_qq += qa0 * qw0 + qa1 * qw1; sum_qa += qa0 + qa1; sum_qw += qw0 + qw1;
            }
            total_sum += (d_a * d_w) * (float)(dot_qq * (int)sc_a[g] * sc_w[g]);
            total_sum -= (d_a * dmin_w) * (float)(sum_qa * (int)sc_a[g] * mn_w[g]);
            total_sum -= (d_w * dmin_a) * (float)(sum_qw * (int)sc_w[g] * mn_a[g]);
            total_sum += (dmin_a * dmin_w) * 32.0f * (float)((int)mn_a[g] * mn_w[g]);
        }
    }
    __shared__ float s_part[32];
    float warp_sum = total_sum;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, mask);
    if ((tid % 32) == 0) s_part[tid / 32] = warp_sum;
    __syncthreads();
    if (tid < 32) {
        float final_sum = (tid < (n_threads / 32)) ? s_part[tid] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) final_sum += __shfl_xor_sync(0xffffffff, final_sum, mask);
        if (tid == 0) C[row * N + col] = final_sum;
    }
}

// ============================================================================ 
// Dispatch Integration
// ============================================================================ 

void ggml_cuda_rrs_mul_mat(ggml_backend_cuda_context& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    const int M = src1->ne[1], N = src0->ne[1], K = src0->ne[0];
    cudaStream_t stream = ctx.stream();
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS_ACT, K);
    size_t total_size = M * row_size, actual_size;
    void* d_act_q4k = ctx.pool().alloc(total_size, &actual_size);
    ggml_cuda_rrs_fwht_quantize((const float*)src1->data, d_act_q4k, K, M, stream);
    dim3 grid(N, M); rrs_gemm_q4k_q4k_kernel<<<grid, 256, 0, stream>>>(d_act_q4k, src0->data, (float*)dst->data, M, N, K);
    ctx.pool().free(d_act_q4k, actual_size);
}

bool ggml_cuda_supports_rrs(const ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_Q4_K_RRS) return false;
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    return (prop.major > 7) || (prop.major == 7 && prop.minor >= 5);
}

// ============================================================================ 
// Benchmarking
// ============================================================================ 

void ggml_cuda_rrs_benchmark(int M, int N, int K, int iterations, RRSBenchmarkResult* result) {
    cudaStream_t stream; cudaStreamCreate(&stream);
    float *h_A = (float*)malloc(M * K * sizeof(float)), *h_B = (float*)malloc(N * K * sizeof(float));
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * K; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;
    float *d_A, *d_B, *d_C; void *d_A_q4k, *d_B_q4k;
    cudaMalloc(&d_A, M * K * sizeof(float)); cudaMalloc(&d_B, N * K * sizeof(float)); cudaMalloc(&d_C, M * N * sizeof(float));
    size_t rs = ggml_row_size(GGML_TYPE_Q4_K_RRS_ACT, K);
    cudaMalloc(&d_A_q4k, M * rs); cudaMalloc(&d_B_q4k, N * rs);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) ggml_cuda_rrs_fwht(d_A, d_A, K, M, stream); 
    cudaEventRecord(stop, stream); cudaEventSynchronize(stop); cudaEventElapsedTime(&result->fwht_time_ms, start, stop);
    result->fwht_time_ms /= iterations;
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) ggml_cuda_rrs_fwht_quantize(d_A, d_A_q4k, K, M, stream);
    cudaEventRecord(stop, stream); cudaEventSynchronize(stop); cudaEventElapsedTime(&result->quantize_time_ms, start, stop);
    result->quantize_time_ms /= iterations;
    ggml_cuda_rrs_fwht_quantize(d_B, d_B_q4k, K, N, stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) rrs_gemm_q4k_q4k_kernel<<<dim3(N, M), 256, 0, stream>>>(d_A_q4k, d_B_q4k, d_C, M, N, K);
    cudaEventRecord(stop, stream); cudaEventSynchronize(stop); cudaEventElapsedTime(&result->int4_wmma_time_ms, start, stop);
    result->int4_wmma_time_ms /= iterations;
    result->q8_repack_time_ms = 0.0f; result->M = M; result->N = N; result->K = K;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_A_q4k); cudaFree(d_B_q4k);
    free(h_A); free(h_B); cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
}

void ggml_cuda_rrs_print_benchmark(const RRSBenchmarkResult* result) {
    printf("RRS CUDA Benchmark Results (M=%d, N=%d, K=%d):\n", result->M, result->N, result->K);
    printf("  FWHT:           %.3f ms\n", result->fwht_time_ms);
    printf("  FWHT+Quantize:  %.3f ms\n", result->quantize_time_ms);
    printf("  GPU GEMM:       %.3f ms\n", result->int4_wmma_time_ms);
    double ops = 2.0 * result->M * result->N * result->K;
    printf("  GPU Perform.:   %.2f TOPS\n", ops / (result->int4_wmma_time_ms * 1e-3) / 1e12);
}

extern "C" void ggml_cuda_rrs_test(void) {
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("RRS CUDA Test on: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);
    RRSBenchmarkResult res; ggml_cuda_rrs_benchmark(128, 2048, 2048, 50, &res); ggml_cuda_rrs_print_benchmark(&res);
}
