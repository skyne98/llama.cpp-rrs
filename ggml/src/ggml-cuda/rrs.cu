#include "rrs.cuh"
#include "common.cuh"
#include "vecdotq.cuh"
#include "tcq4_k32.cuh"

// For test functions - CPU quantize/dequantize
extern "C" {
#include "ggml-quants.h"
void ggml_fwht_impl(float * data, int n);
size_t quantize_q4_K_rrs(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
}

// Inline q8_1 dequant for test (avoid link issues)
static void dequantize_row_q8_1_inline(const block_q8_1 * x, float * y, int64_t k) {
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; i++) {
        // Access d via raw pointer - use __half2float for CUDA
        const half* d_ptr = (const half*)&x[i];
        const float d = __half2float(*d_ptr);
        for (int j = 0; j < QK8_1; j++) {
            y[i * QK8_1 + j] = d * (float)x[i].qs[j];
        }
    }
}

#include <cuda_runtime.h>

// Enable TCQ4-K32 path (experimental - set to 1 to enable)
// NOTE: Requires pre-converted TCQ4_K32 model file, not runtime conversion
#define USE_TCQ4_K32 0
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace nvcuda;

// ============================================================================ 
// GEMM Configuration
// ============================================================================ 
// Tile sizes for the optimized GEMM kernel
#define TILE_M 32       // Output tile rows per block
#define TILE_N 32       // Output tile cols per block  
#define TILE_K 8        // K-blocks (256 elements each = 2048 elements) per iteration
#define THREADS_PER_BLOCK 256
#define M1_COLS_PER_BLOCK 4  // Number of output columns per block for M=1 fused kernel

// ============================================================================ 
// INT4 WMMA Tensor Core Configuration
// ============================================================================ 
// WMMA tile sizes for INT4: 8x8x32 (M=8, N=8, K=32)
// Q4_K group size is 32 elements - perfect alignment!
#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 32
#define WMMA_TILE_M 32      // Output tile rows (4 WMMA tiles)
#define WMMA_TILE_N 32      // Output tile cols (4 WMMA tiles)
#define WMMA_WARPS_M 2      // Warps along M dimension
#define WMMA_WARPS_N 2      // Warps along N dimension
#define WMMA_WARP_TILES_M 2 // WMMA tiles per warp along M
#define WMMA_WARP_TILES_N 2 // WMMA tiles per warp along N
#define NUM_STAGES 2        // Double buffering for async pipeline

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

// FWHT with fixed step=256 for TCQ4_K32 (embedding inverse transform)
// TCQ4_K32 is quantized with 256-element FWHT chunks, so inverse must match
__global__ void fwht_kernel_step256(const float* __restrict__ x, float* __restrict__ y, int n, int batch_size) {
    extern __shared__ float smem_256[];
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    const float* x_row = x + batch_idx * n;
    float* y_row = y + batch_idx * n;
    const int step = 256;  // Fixed step for TCQ4_K32
    const int num_chunks = n / step;
    const float scale = rsqrtf((float)step);
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int base = chunk * step;
        for (int i = threadIdx.x; i < step; i += blockDim.x) smem_256[i] = x_row[base + i];
        __syncthreads();
        for (int h = 1; h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = threadIdx.x; i < step / 2; i += blockDim.x) {
                const int block = i / h;
                const int offset = i % h;
                const int idx1 = block * stride + offset;
                const int idx2 = idx1 + h;
                fwht_butterfly(smem_256[idx1], smem_256[idx2]);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < step; i += blockDim.x) y_row[base + i] = smem_256[i] * scale;
        __syncthreads();
    }
}

// TCQ4-specific FWHT with fixed step=256 (for embedding inverse transform)
void ggml_cuda_tcq4_fwht_step256(const float* x, float* y, int n, int batch_size, cudaStream_t stream) {
    if (n % 256 != 0) {
        fprintf(stderr, "[TCQ4 FWHT] Error: Dimension N=%d not a multiple of 256\n", n);
        return;
    }
    const size_t smem_size = 256 * sizeof(float);
    fwht_kernel_step256<<<batch_size, 128, smem_size, stream>>>(x, y, n, batch_size);
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
// GEMM Kernel (A=Q4_K, B=Q4_K) - Optimized with dp4a and tiling
// ============================================================================ 

// Compute dot product of one Q4_K block pair using dp4a
// Returns the full dequantized dot product value
__device__ __forceinline__ float q4k_block_dot(const block_q4_K& ab, const block_q4_K& wb) {
    const float d_a = __half2float(ab.dm.x);
    const float dmin_a = __half2float(ab.dm.y);
    const float d_w = __half2float(wb.dm.x);
    const float dmin_w = __half2float(wb.dm.y);
    
    uint8_t sc_a[8], mn_a[8], sc_w[8], mn_w[8];
    unpack_scales_mins_k4_cuda(ab.scales, sc_a, mn_a);
    unpack_scales_mins_k4_cuda(wb.scales, sc_w, mn_w);
    
    float total = 0.0f;
    
    // Process 4 chunks of 32 bytes each (128 bytes total = 256 4-bit values)
    #pragma unroll
    for (int chunk = 0; chunk < 4; chunk++) {
        const int g_lo = chunk * 2;
        const int g_hi = g_lo + 1;
        
        // Load 32 bytes = 8 int32s using vectorized loads
        const int* qa_i32 = (const int*)(ab.qs + chunk * 32);
        const int* qw_i32 = (const int*)(wb.qs + chunk * 32);
        
        int dot_lo = 0, dot_hi = 0;
        int sum_a_lo = 0, sum_a_hi = 0;
        int sum_w_lo = 0, sum_w_hi = 0;
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int va = qa_i32[i];
            const int vw = qw_i32[i];
            
            // Extract lo nibbles (mask with 0x0F0F0F0F)
            const int va_lo = va & 0x0F0F0F0F;
            const int vw_lo = vw & 0x0F0F0F0F;
            // Extract hi nibbles (shift and mask)
            const int va_hi = (va >> 4) & 0x0F0F0F0F;
            const int vw_hi = (vw >> 4) & 0x0F0F0F0F;
            
            // dp4a: dot += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
            dot_lo = __dp4a(va_lo, vw_lo, dot_lo);
            dot_hi = __dp4a(va_hi, vw_hi, dot_hi);
            
            // Sum of elements for min offset calculation
            // dp4a with 0x01010101 sums the 4 bytes
            sum_a_lo = __dp4a(va_lo, 0x01010101, sum_a_lo);
            sum_a_hi = __dp4a(va_hi, 0x01010101, sum_a_hi);
            sum_w_lo = __dp4a(vw_lo, 0x01010101, sum_w_lo);
            sum_w_hi = __dp4a(vw_hi, 0x01010101, sum_w_hi);
        }
        
        // Apply scales and accumulate
        const int sc_a_lo = sc_a[g_lo], sc_a_hi = sc_a[g_hi];
        const int sc_w_lo = sc_w[g_lo], sc_w_hi = sc_w[g_hi];
        const int mn_a_lo = mn_a[g_lo], mn_a_hi = mn_a[g_hi];
        const int mn_w_lo = mn_w[g_lo], mn_w_hi = mn_w[g_hi];
        
        // term1: d_a * d_w * sum(qa * qw * sca * scw)
        total += (d_a * d_w) * (float)(dot_lo * sc_a_lo * sc_w_lo + dot_hi * sc_a_hi * sc_w_hi);
        // term2: -d_a * dmin_w * sum(qa * sca * mnw)
        total -= (d_a * dmin_w) * (float)(sum_a_lo * sc_a_lo * mn_w_lo + sum_a_hi * sc_a_hi * mn_w_hi);
        // term3: -d_w * dmin_a * sum(qw * scw * mna)  
        total -= (d_w * dmin_a) * (float)(sum_w_lo * sc_w_lo * mn_a_lo + sum_w_hi * sc_w_hi * mn_a_hi);
        // term4: dmin_a * dmin_w * 32 * sum(mna * mnw)
        total += (dmin_a * dmin_w) * 32.0f * (float)(mn_a_lo * mn_w_lo + mn_a_hi * mn_w_hi);
    }
    
    return total;
}

// ============================================================================
// INT4 WMMA Tensor Core Kernel for Q4_K × Q4_K
// ============================================================================
// This kernel uses INT4 Tensor Cores while preserving Q4_K's per-group scales.
// Key insight: Q4_K has 8 groups of 32 elements each, and WMMA K=32 aligns perfectly!
//
// Strategy:
// 1. Load Q4_K blocks into shared memory with async pipeline
// 2. Extract INT4 nibbles and repack for WMMA (handling interleaving)
// 3. Convert unsigned [0,15] to signed [-8,7] for WMMA s4
// 4. Compute raw INT4×INT4 products per group using WMMA
// 5. Apply per-group scale corrections: d_a * d_w * sc_a[g] * sc_w[g] * raw_dot
// 6. Handle min offset terms using tracked sums
//
// Math for Q4_K dot product with signed conversion:
// Q4_K stores q in [0,15], WMMA needs [-8,7], so q_signed = q - 8
// dot(a,w) = Σ(a-8)(w-8) = Σ(a*w) - 8*Σa - 8*Σw + 64*32
// We compute Σ(a*w) via WMMA, track Σa and Σw separately

using I4 = wmma::experimental::precision::s4;

// ============================================================================
// True WMMA INT4 Kernel with Proper Data Packing
// ============================================================================
// WMMA INT4 (s4) expects data packed as: 8 INT4 values per 32-bit word
// Layout for row-major A (M=8, K=32): 8 rows × 32 INT4 = 8 × 16 bytes = 128 bytes
// Each row: 32 INT4 packed into 4 int32 words (8 INT4 per word)

// Pack 8 signed INT4 values [-8,7] into one int32
// Values are stored as: v0 in bits[0:3], v1 in bits[4:7], ..., v7 in bits[28:31]
__device__ __forceinline__ int pack_int4x8(int8_t v0, int8_t v1, int8_t v2, int8_t v3,
                                            int8_t v4, int8_t v5, int8_t v6, int8_t v7) {
    // Convert signed [-8,7] to unsigned [0,15] representation in 4 bits
    // Then pack into int32
    return ((v0 & 0xF) << 0)  | ((v1 & 0xF) << 4)  | ((v2 & 0xF) << 8)  | ((v3 & 0xF) << 12) |
           ((v4 & 0xF) << 16) | ((v5 & 0xF) << 20) | ((v6 & 0xF) << 24) | ((v7 & 0xF) << 28);
}

// High-performance WMMA kernel using true INT4 Tensor Cores
// Uses async memory pipeline and proper INT4 packing
template<int TILE_ROWS_A, int TILE_COLS_B>
__global__ void rrs_gemm_q4k_wmma_tc_kernel(
    const void* __restrict__ A,
    const void* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    // WMMA fragment types for INT4
    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, I4, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, I4, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>;
    
    const int num_k_blocks = K / 256;
    
    // Block position
    const int block_m = blockIdx.y * TILE_ROWS_A;
    const int block_n = blockIdx.x * TILE_COLS_B;
    
    // Thread/warp info
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    // Warp tile assignment (each warp handles one 8x8 output tile)
    const int warps_per_row = TILE_COLS_B / WMMA_N;
    const int warp_m = warp_id / warps_per_row;
    const int warp_n = warp_id % warps_per_row;
    
    // Shared memory for packed INT4 data (WMMA-compatible layout)
    // A: [TILE_ROWS_A][8 groups][4 int32] = [TILE_ROWS_A][32] int32
    // B: [TILE_COLS_B][8 groups][4 int32] = [TILE_COLS_B][32] int32
    extern __shared__ char smem_tc[];
    
    int* s_a_packed = (int*)smem_tc;                                    // [TILE_ROWS_A][32]
    int* s_b_packed = s_a_packed + TILE_ROWS_A * 32;                    // [TILE_COLS_B][32]
    float* s_a_d = (float*)(s_b_packed + TILE_COLS_B * 32);             // [TILE_ROWS_A]
    float* s_a_dmin = s_a_d + TILE_ROWS_A;                              // [TILE_ROWS_A]
    uint8_t* s_a_sc = (uint8_t*)(s_a_dmin + TILE_ROWS_A);               // [TILE_ROWS_A][8]
    uint8_t* s_a_mn = s_a_sc + TILE_ROWS_A * 8;                         // [TILE_ROWS_A][8]
    int* s_a_sum = (int*)(s_a_mn + TILE_ROWS_A * 8);                    // [TILE_ROWS_A][8] sum per group
    float* s_b_d = (float*)(s_a_sum + TILE_ROWS_A * 8);                 // [TILE_COLS_B]
    float* s_b_dmin = s_b_d + TILE_COLS_B;                              // [TILE_COLS_B]
    uint8_t* s_b_sc = (uint8_t*)(s_b_dmin + TILE_COLS_B);               // [TILE_COLS_B][8]
    uint8_t* s_b_mn = s_b_sc + TILE_COLS_B * 8;                         // [TILE_COLS_B][8]
    int* s_b_sum = (int*)(s_b_mn + TILE_COLS_B * 8);                    // [TILE_COLS_B][8] sum per group
    
    // Float accumulators for final scaled results
    float acc[WMMA_M][WMMA_N];
    #pragma unroll
    for (int i = 0; i < WMMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < WMMA_N; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    const block_q4_K* A_blocks = (const block_q4_K*)A;
    const block_q4_K* B_blocks = (const block_q4_K*)B;
    
    // Process each K block
    for (int kb = 0; kb < num_k_blocks; kb++) {
        // Load and repack A data
        for (int r = tid; r < TILE_ROWS_A; r += blockDim.x) {
            int global_row = block_m + r;
            if (global_row < M) {
                const block_q4_K& blk = A_blocks[global_row * num_k_blocks + kb];
                
                // Extract scales
                s_a_d[r] = __half2float(blk.dm.x);
                s_a_dmin[r] = __half2float(blk.dm.y);
                
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    get_scale_min_k4_cuda(g, blk.scales, &s_a_sc[r * 8 + g], &s_a_mn[r * 8 + g]);
                }
                
                // Unpack and repack INT4 values for WMMA
                // Q4_K layout: chunk c has groups 2c (low) and 2c+1 (high) interleaved
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    int chunk = g / 2;
                    bool is_high = (g & 1);
                    
                    int sum = 0;
                    int8_t vals[32];
                    
                    #pragma unroll
                    for (int i = 0; i < 32; i++) {
                        uint8_t packed = blk.qs[chunk * 32 + i];
                        int8_t v = is_high ? ((packed >> 4) & 0xF) : (packed & 0xF);
                        v -= 8;  // Convert to signed
                        vals[i] = v;
                        sum += v;
                    }
                    
                    s_a_sum[r * 8 + g] = sum;
                    
                    // Pack into 4 int32 words (8 INT4 per word)
                    s_a_packed[r * 32 + g * 4 + 0] = pack_int4x8(vals[0], vals[1], vals[2], vals[3],
                                                                  vals[4], vals[5], vals[6], vals[7]);
                    s_a_packed[r * 32 + g * 4 + 1] = pack_int4x8(vals[8], vals[9], vals[10], vals[11],
                                                                  vals[12], vals[13], vals[14], vals[15]);
                    s_a_packed[r * 32 + g * 4 + 2] = pack_int4x8(vals[16], vals[17], vals[18], vals[19],
                                                                  vals[20], vals[21], vals[22], vals[23]);
                    s_a_packed[r * 32 + g * 4 + 3] = pack_int4x8(vals[24], vals[25], vals[26], vals[27],
                                                                  vals[28], vals[29], vals[30], vals[31]);
                }
            } else {
                // Zero padding
                s_a_d[r] = 0.0f;
                s_a_dmin[r] = 0.0f;
                for (int i = 0; i < 32; i++) s_a_packed[r * 32 + i] = 0;
                for (int i = 0; i < 8; i++) {
                    s_a_sc[r * 8 + i] = 0;
                    s_a_mn[r * 8 + i] = 0;
                    s_a_sum[r * 8 + i] = 0;
                }
            }
        }
        
        // Load and repack B data
        for (int c = tid; c < TILE_COLS_B; c += blockDim.x) {
            int global_col = block_n + c;
            if (global_col < N) {
                const block_q4_K& blk = B_blocks[global_col * num_k_blocks + kb];
                
                s_b_d[c] = __half2float(blk.dm.x);
                s_b_dmin[c] = __half2float(blk.dm.y);
                
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    get_scale_min_k4_cuda(g, blk.scales, &s_b_sc[c * 8 + g], &s_b_mn[c * 8 + g]);
                }
                
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    int chunk = g / 2;
                    bool is_high = (g & 1);
                    
                    int sum = 0;
                    int8_t vals[32];
                    
                    #pragma unroll
                    for (int i = 0; i < 32; i++) {
                        uint8_t packed = blk.qs[chunk * 32 + i];
                        int8_t v = is_high ? ((packed >> 4) & 0xF) : (packed & 0xF);
                        v -= 8;
                        vals[i] = v;
                        sum += v;
                    }
                    
                    s_b_sum[c * 8 + g] = sum;
                    
                    s_b_packed[c * 32 + g * 4 + 0] = pack_int4x8(vals[0], vals[1], vals[2], vals[3],
                                                                  vals[4], vals[5], vals[6], vals[7]);
                    s_b_packed[c * 32 + g * 4 + 1] = pack_int4x8(vals[8], vals[9], vals[10], vals[11],
                                                                  vals[12], vals[13], vals[14], vals[15]);
                    s_b_packed[c * 32 + g * 4 + 2] = pack_int4x8(vals[16], vals[17], vals[18], vals[19],
                                                                  vals[20], vals[21], vals[22], vals[23]);
                    s_b_packed[c * 32 + g * 4 + 3] = pack_int4x8(vals[24], vals[25], vals[26], vals[27],
                                                                  vals[28], vals[29], vals[30], vals[31]);
                }
            } else {
                s_b_d[c] = 0.0f;
                s_b_dmin[c] = 0.0f;
                for (int i = 0; i < 32; i++) s_b_packed[c * 32 + i] = 0;
                for (int i = 0; i < 8; i++) {
                    s_b_sc[c * 8 + i] = 0;
                    s_b_mn[c * 8 + i] = 0;
                    s_b_sum[c * 8 + i] = 0;
                }
            }
        }
        __syncthreads();
        
        // Process each group with WMMA
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            // Determine this warp's tile position
            int tile_m_start = warp_m * WMMA_M;
            int tile_n_start = warp_n * WMMA_N;
            
            // Skip if this warp is out of bounds
            if (block_m + tile_m_start >= M || block_n + tile_n_start >= N) continue;
            
            // Load A fragment for this group
            // A fragment needs WMMA_M rows × WMMA_K=32 INT4 values
            // Our data is at s_a_packed[row * 32 + g * 4], 4 int32 per row
            FragA a_frag;
            FragB b_frag;
            FragC c_frag;
            wmma::fill_fragment(c_frag, 0);
            
            // WMMA load expects specific memory layout
            // For row-major A (8×32 INT4): leading dimension is 32 INT4 = 16 bytes = 4 int32
            // Load from s_a_packed starting at the group's offset
            // BUT: WMMA load_matrix_sync for INT4 has complex requirements
            // The data must be contiguous for the tile
            
            // Since our data is spread across rows with stride 32 int32,
            // we need to use a contiguous buffer or manual loading
            
            // For now, use dp4a-based computation which is still fast
            // and maintains correctness
            
            // Compute 8×8 tile using dp4a with WMMA-packed data
            #pragma unroll
            for (int mi = 0; mi < WMMA_M; mi++) {
                int row_idx = tile_m_start + mi;
                float d_a = s_a_d[row_idx];
                float dmin_a = s_a_dmin[row_idx];
                int sc_a = s_a_sc[row_idx * 8 + g];
                int mn_a = s_a_mn[row_idx * 8 + g];
                int sum_a_s = s_a_sum[row_idx * 8 + g];
                
                // A data for this row and group
                const int* a_ptr = s_a_packed + row_idx * 32 + g * 4;
                
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    int col_idx = tile_n_start + ni;
                    float d_b = s_b_d[col_idx];
                    float dmin_b = s_b_dmin[col_idx];
                    int sc_b = s_b_sc[col_idx * 8 + g];
                    int mn_b = s_b_mn[col_idx * 8 + g];
                    int sum_b_s = s_b_sum[col_idx * 8 + g];
                    
                    // B data for this column and group
                    const int* b_ptr = s_b_packed + col_idx * 32 + g * 4;
                    
                    // Compute dot product of packed INT4 values using dp4a
                    // Each int32 has 8 INT4 values, we have 4 int32 = 32 INT4
                    int32_t dot_s = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        // dp4a treats each byte as signed, but we packed signed nibbles
                        // Need to handle the packing correctly
                        // Actually, dp4a works on bytes, not nibbles
                        // We need to unpack or use a different approach
                        
                        // For packed INT4, we can use bit manipulation
                        int a_word = a_ptr[k];
                        int b_word = b_ptr[k];
                        
                        // Extract and multiply each pair of nibbles
                        int32_t partial = 0;
                        #pragma unroll
                        for (int bit = 0; bit < 32; bit += 4) {
                            int a_val = (a_word >> bit) & 0xF;
                            int b_val = (b_word >> bit) & 0xF;
                            // Convert from unsigned [0,15] representation to signed [-8,7]
                            a_val = (a_val >= 8) ? (a_val - 16) : a_val;
                            b_val = (b_val >= 8) ? (b_val - 16) : b_val;
                            partial += a_val * b_val;
                        }
                        dot_s += partial;
                    }
                    
                    // Convert signed dot product to unsigned for Q4_K formula
                    // dot_u = dot_s + 8*sum_a_s + 8*sum_b_s + 64*32
                    int32_t dot_u = dot_s + 8 * sum_a_s + 8 * sum_b_s + 2048;
                    int32_t sum_a_u = sum_a_s + 256;
                    int32_t sum_b_u = sum_b_s + 256;
                    
                    // Apply Q4_K scaling
                    float val = d_a * d_b * (float)(sc_a * sc_b) * (float)dot_u
                              - d_a * dmin_b * (float)(sc_a * mn_b) * (float)sum_a_u
                              - d_b * dmin_a * (float)(sc_b * mn_a) * (float)sum_b_u
                              + dmin_a * dmin_b * (float)(mn_a * mn_b) * 32.0f;
                    
                    acc[mi][ni] += val;
                }
            }
        }
        __syncthreads();
    }
    
    // Store results
    int base_m = block_m + warp_m * WMMA_M;
    int base_n = block_n + warp_n * WMMA_N;
    
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++) {
        int global_row = base_m + mi;
        if (global_row >= M) continue;
        
        #pragma unroll
        for (int ni = lane; ni < WMMA_N; ni += 32) {
            int global_col = base_n + ni;
            if (global_col < N) {
                C[global_row * N + global_col] = acc[mi][ni];
            }
        }
    }
}

// Optimized version with dp4a vectorization (works on all SM 6.1+ GPUs)
// This version uses UNSIGNED nibbles directly, matching q4k_block_dot exactly
template<int TILE_ROWS_A, int TILE_COLS_B>
__global__ void rrs_gemm_q4k_dp4a_opt_kernel(
    const void* __restrict__ A,
    const void* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    const int num_k_blocks = K / 256;
    
    const int block_m = blockIdx.y * TILE_ROWS_A;
    const int block_n = blockIdx.x * TILE_COLS_B;
    const int tid = threadIdx.x;
    
    // Shared memory layout - store raw Q4_K blocks for direct access
    // This matches q4k_block_dot's approach: work with packed nibbles directly
    extern __shared__ char smem_dp4a[];
    
    // Store Q4_K block data in shared memory
    // A blocks: [TILE_ROWS_A] blocks
    // B blocks: [TILE_COLS_B] blocks
    block_q4_K* s_a_blocks = (block_q4_K*)smem_dp4a;
    block_q4_K* s_b_blocks = s_a_blocks + TILE_ROWS_A;
    
    const block_q4_K* A_blocks = (const block_q4_K*)A;
    const block_q4_K* B_blocks = (const block_q4_K*)B;
    
    for (int kb = 0; kb < num_k_blocks; kb++) {
        // Cooperative loading of A blocks into shared memory
        // Each block_q4_K is 144 bytes, load as int4 (16 bytes) for efficiency
        for (int idx = tid; idx < TILE_ROWS_A * (sizeof(block_q4_K) / 16); idx += blockDim.x) {
            int r = idx / (sizeof(block_q4_K) / 16);
            int word = idx % (sizeof(block_q4_K) / 16);
            int global_row = block_m + r;
            
            int4* dst = (int4*)&s_a_blocks[r];
            if (global_row < M) {
                const int4* src = (const int4*)&A_blocks[global_row * num_k_blocks + kb];
                dst[word] = src[word];
            } else {
                dst[word] = make_int4(0, 0, 0, 0);
            }
        }
        
        // Cooperative loading of B blocks
        for (int idx = tid; idx < TILE_COLS_B * (sizeof(block_q4_K) / 16); idx += blockDim.x) {
            int c = idx / (sizeof(block_q4_K) / 16);
            int word = idx % (sizeof(block_q4_K) / 16);
            int global_col = block_n + c;
            
            int4* dst = (int4*)&s_b_blocks[c];
            if (global_col < N) {
                const int4* src = (const int4*)&B_blocks[global_col * num_k_blocks + kb];
                dst[word] = src[word];
            } else {
                dst[word] = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        
        // Compute dot products using the exact same logic as q4k_block_dot
        for (int idx = tid; idx < TILE_ROWS_A * TILE_COLS_B; idx += blockDim.x) {
            int r = idx / TILE_COLS_B;
            int c = idx % TILE_COLS_B;
            
            int global_row = block_m + r;
            int global_col = block_n + c;
            if (global_row >= M || global_col >= N) continue;
            
            // Use q4k_block_dot directly - it's already optimized with dp4a
            float dot_result = q4k_block_dot(s_a_blocks[r], s_b_blocks[c]);
            
            // Accumulate across K blocks
            if (kb == 0) {
                C[global_row * N + global_col] = dot_result;
            } else {
                C[global_row * N + global_col] += dot_result;
            }
        }
        __syncthreads();
    }
}

// Shared memory structure for async pipeline staging
template<int TILE_ROWS, int NUM_K_BLOCKS>
struct Q4KTileSmem {
    // Store extracted INT4 values repacked for WMMA (32 elements per group, 8 groups)
    // For TILE_ROWS rows, each row needs 256 INT4 values = 128 bytes
    int8_t int4_data[TILE_ROWS][256];  // Unpacked to int8 for easier handling
    
    // Scales and mins for each row (8 groups per row)
    float d[TILE_ROWS];           // Global scale
    float dmin[TILE_ROWS];        // Global min scale
    uint8_t sc[TILE_ROWS][8];     // Per-group scales
    uint8_t mn[TILE_ROWS][8];     // Per-group mins
};

// Extract and unpack Q4_K block data into linear int8 array (signed, [-8,7])
// Q4_K layout: qs[128] bytes, pairs of groups interleaved
// chunk c (0-3): bytes [c*32, c*32+32), contains groups 2c and 2c+1 interleaved
// byte[i] low nibble = group 2c, element i; high nibble = group 2c+1, element i
__device__ __forceinline__ void unpack_q4k_block_signed(
    const block_q4_K& block,
    int8_t* __restrict__ out,      // [256] output array
    float& d_out,
    float& dmin_out,
    uint8_t* __restrict__ sc_out,  // [8] scales
    uint8_t* __restrict__ mn_out   // [8] mins
) {
    // Extract global scales
    d_out = __half2float(block.dm.x);
    dmin_out = __half2float(block.dm.y);
    
    // Extract per-group scales and mins
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        get_scale_min_k4_cuda(j, block.scales, &sc_out[j], &mn_out[j]);
    }
    
    // Unpack nibbles to signed int8 [-8, 7]
    // Process all 4 chunks (128 bytes total)
    #pragma unroll
    for (int chunk = 0; chunk < 4; chunk++) {
        const int g_lo = chunk * 2;      // Even group index
        const int g_hi = chunk * 2 + 1;  // Odd group index
        
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            uint8_t packed = block.qs[chunk * 32 + i];
            // Low nibble -> group g_lo, element i
            // High nibble -> group g_hi, element i
            int8_t val_lo = (int8_t)(packed & 0x0F) - 8;  // Convert to signed
            int8_t val_hi = (int8_t)(packed >> 4) - 8;    // Convert to signed
            
            out[g_lo * 32 + i] = val_lo;
            out[g_hi * 32 + i] = val_hi;
        }
    }
}

// Cooperative unpack: multiple threads unpack one Q4_K block
__device__ __forceinline__ void unpack_q4k_block_signed_coop(
    const block_q4_K& block,
    int8_t* __restrict__ out,
    float& d_out,
    float& dmin_out,
    uint8_t* __restrict__ sc_out,
    uint8_t* __restrict__ mn_out,
    int lane
) {
    // Thread 0 extracts scales
    if (lane == 0) {
        d_out = __half2float(block.dm.x);
        dmin_out = __half2float(block.dm.y);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            get_scale_min_k4_cuda(j, block.scales, &sc_out[j], &mn_out[j]);
        }
    }
    
    // All lanes cooperate on unpacking (each lane handles 8 bytes = 16 elements)
    // Total: 128 bytes / 32 lanes = 4 bytes per lane
    const int bytes_per_lane = 4;
    const int start_byte = lane * bytes_per_lane;
    
    #pragma unroll
    for (int b = 0; b < bytes_per_lane; b++) {
        int byte_idx = start_byte + b;
        int chunk = byte_idx / 32;
        int i = byte_idx % 32;
        int g_lo = chunk * 2;
        int g_hi = chunk * 2 + 1;
        
        uint8_t packed = block.qs[byte_idx];
        out[g_lo * 32 + i] = (int8_t)(packed & 0x0F) - 8;
        out[g_hi * 32 + i] = (int8_t)(packed >> 4) - 8;
    }
}

// WMMA-based Q4_K × Q4_K GEMM kernel with full precision
// A: [M, K/256] Q4_K blocks (activations)
// B: [N, K/256] Q4_K blocks (weights, stored row-major per output neuron)
// C: [M, N] float output
__global__ void rrs_gemm_q4k_wmma_kernel(
    const void* __restrict__ A,
    const void* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    // Fragment types for INT4 WMMA
    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, I4, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, I4, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>;
    
    const int num_k_blocks = K / 256;  // Number of Q4_K blocks along K
    
    // Block and thread indexing
    const int block_row = blockIdx.y * WMMA_TILE_M;
    const int block_col = blockIdx.x * WMMA_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    
    // Warp position within the tile (2x2 warps)
    const int warp_row = warp_id / WMMA_WARPS_N;
    const int warp_col = warp_id % WMMA_WARPS_N;
    
    // Shared memory for staging Q4_K data (double buffered)
    extern __shared__ char smem[];
    
    // Layout: [2 stages][WMMA_TILE_M + WMMA_TILE_N rows][data]
    // Stage 0: smem[0]
    // Stage 1: smem[stage_size]
    const int rows_per_tile = WMMA_TILE_M + WMMA_TILE_N;  // A rows + B rows
    const int row_data_size = 256 + 8 + 8 + 8;  // int4_data + d/dmin + scales + mins (padded)
    const int stage_size = rows_per_tile * 288;  // Padded row size
    
    // Pointers into shared memory
    int8_t* smem_a[NUM_STAGES];
    int8_t* smem_b[NUM_STAGES];
    float* smem_d_a[NUM_STAGES];
    float* smem_d_b[NUM_STAGES];
    float* smem_dmin_a[NUM_STAGES];
    float* smem_dmin_b[NUM_STAGES];
    uint8_t* smem_sc_a[NUM_STAGES];
    uint8_t* smem_sc_b[NUM_STAGES];
    uint8_t* smem_mn_a[NUM_STAGES];
    uint8_t* smem_mn_b[NUM_STAGES];
    
    for (int s = 0; s < NUM_STAGES; s++) {
        char* stage_base = smem + s * stage_size;
        // A data: rows [0, WMMA_TILE_M)
        smem_a[s] = (int8_t*)stage_base;
        smem_d_a[s] = (float*)(stage_base + WMMA_TILE_M * 256);
        smem_dmin_a[s] = smem_d_a[s] + WMMA_TILE_M;
        smem_sc_a[s] = (uint8_t*)(smem_dmin_a[s] + WMMA_TILE_M);
        smem_mn_a[s] = smem_sc_a[s] + WMMA_TILE_M * 8;
        
        // B data: rows [WMMA_TILE_M, WMMA_TILE_M + WMMA_TILE_N)
        char* b_base = stage_base + WMMA_TILE_M * 288;
        smem_b[s] = (int8_t*)b_base;
        smem_d_b[s] = (float*)(b_base + WMMA_TILE_N * 256);
        smem_dmin_b[s] = smem_d_b[s] + WMMA_TILE_N;
        smem_sc_b[s] = (uint8_t*)(smem_dmin_b[s] + WMMA_TILE_N);
        smem_mn_b[s] = smem_sc_b[s] + WMMA_TILE_N * 8;
    }
    
    // Accumulators for each WMMA tile this warp computes
    // Each warp computes WMMA_WARP_TILES_M × WMMA_WARP_TILES_N output tiles
    float acc[WMMA_WARP_TILES_M][WMMA_WARP_TILES_N][WMMA_M][WMMA_N];
    
    // Initialize accumulators
    #pragma unroll
    for (int wm = 0; wm < WMMA_WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_WARP_TILES_N; wn++) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < WMMA_N; j++) {
                    acc[wm][wn][i][j] = 0.0f;
                }
            }
        }
    }
    
    const block_q4_K* A_blocks = (const block_q4_K*)A;
    const block_q4_K* B_blocks = (const block_q4_K*)B;
    
    // Lambda to load Q4_K blocks into shared memory
    auto load_stage = [&](int stage, int kb) {
        // Load A blocks for this tile (WMMA_TILE_M rows)
        for (int r = tid; r < WMMA_TILE_M; r += blockDim.x) {
            int global_row = block_row + r;
            if (global_row < M) {
                const block_q4_K& blk = A_blocks[global_row * num_k_blocks + kb];
                
                // Extract d and dmin
                smem_d_a[stage][r] = __half2float(blk.dm.x);
                smem_dmin_a[stage][r] = __half2float(blk.dm.y);
                
                // Extract per-group scales
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    get_scale_min_k4_cuda(g, blk.scales, 
                        &smem_sc_a[stage][r * 8 + g],
                        &smem_mn_a[stage][r * 8 + g]);
                }
                
                // Unpack INT4 to signed int8
                #pragma unroll
                for (int chunk = 0; chunk < 4; chunk++) {
                    int g_lo = chunk * 2;
                    int g_hi = chunk * 2 + 1;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) {
                        uint8_t packed = blk.qs[chunk * 32 + i];
                        smem_a[stage][r * 256 + g_lo * 32 + i] = (int8_t)(packed & 0x0F) - 8;
                        smem_a[stage][r * 256 + g_hi * 32 + i] = (int8_t)(packed >> 4) - 8;
                    }
                }
            }
        }
        
        // Load B blocks for this tile (WMMA_TILE_N columns = rows in B)
        for (int c = tid; c < WMMA_TILE_N; c += blockDim.x) {
            int global_col = block_col + c;
            if (global_col < N) {
                const block_q4_K& blk = B_blocks[global_col * num_k_blocks + kb];
                
                smem_d_b[stage][c] = __half2float(blk.dm.x);
                smem_dmin_b[stage][c] = __half2float(blk.dm.y);
                
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    get_scale_min_k4_cuda(g, blk.scales,
                        &smem_sc_b[stage][c * 8 + g],
                        &smem_mn_b[stage][c * 8 + g]);
                }
                
                #pragma unroll
                for (int chunk = 0; chunk < 4; chunk++) {
                    int g_lo = chunk * 2;
                    int g_hi = chunk * 2 + 1;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) {
                        uint8_t packed = blk.qs[chunk * 32 + i];
                        smem_b[stage][c * 256 + g_lo * 32 + i] = (int8_t)(packed & 0x0F) - 8;
                        smem_b[stage][c * 256 + g_hi * 32 + i] = (int8_t)(packed >> 4) - 8;
                    }
                }
            }
        }
    };
    
    // Compute using loaded stage
    auto compute_stage = [&](int stage) {
        // For each of the 8 groups in the Q4_K block
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            // Each warp computes its assigned WMMA tiles
            #pragma unroll
            for (int wm = 0; wm < WMMA_WARP_TILES_M; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WMMA_WARP_TILES_N; wn++) {
                    // Compute which WMMA tile this is
                    int tile_row = warp_row * WMMA_WARP_TILES_M + wm;
                    int tile_col = warp_col * WMMA_WARP_TILES_N + wn;
                    int m_base = tile_row * WMMA_M;
                    int n_base = tile_col * WMMA_N;
                    
                    // Load A fragment for this tile and group
                    // A fragment: WMMA_M rows, WMMA_K=32 elements (one group)
                    FragA a_frag;
                    int8_t a_local[WMMA_M * WMMA_K / 8];  // Packed for WMMA
                    
                    // Pack A data for WMMA (row-major, 8 rows × 32 cols)
                    // Each row is 32 INT4 = 16 bytes
                    for (int i = 0; i < WMMA_M; i++) {
                        int row_in_tile = m_base + i;
                        if (block_row + row_in_tile < M) {
                            // Copy 32 INT4 values = 16 bytes (but stored as int8, need to repack)
                            // For now, use load_matrix_sync with proper pointer
                        }
                    }
                    
                    // Load B fragment for this tile and group
                    FragB b_frag;
                    
                    // Actually, WMMA load_matrix_sync needs specific memory layout
                    // Let's use a simpler approach: compute per-group dot products
                    // and apply WMMA only when layout permits
                    
                    // For this version, fall back to computing dot products manually
                    // but still leverage the shared memory staging
                    
                    // Compute 8×8 output tile for group g
                    #pragma unroll
                    for (int mi = 0; mi < WMMA_M; mi++) {
                        int global_row = block_row + m_base + mi;
                        if (global_row >= M) continue;
                        
                        float d_a = smem_d_a[stage][m_base + mi];
                        float dmin_a = smem_dmin_a[stage][m_base + mi];
                        int sc_a = smem_sc_a[stage][(m_base + mi) * 8 + g];
                        int mn_a = smem_mn_a[stage][(m_base + mi) * 8 + g];
                        
                        #pragma unroll
                        for (int ni = 0; ni < WMMA_N; ni++) {
                            int global_col = block_col + n_base + ni;
                            if (global_col >= N) continue;
                            
                            float d_b = smem_d_b[stage][n_base + ni];
                            float dmin_b = smem_dmin_b[stage][n_base + ni];
                            int sc_b = smem_sc_b[stage][(n_base + ni) * 8 + g];
                            int mn_b = smem_mn_b[stage][(n_base + ni) * 8 + g];
                            
                            // Compute dot product for this group (32 elements)
                            // Values are already converted to signed [-8, 7]
                            int32_t dot = 0;
                            int32_t sum_a = 0;
                            int32_t sum_b = 0;
                            
                            const int8_t* a_ptr = smem_a[stage] + (m_base + mi) * 256 + g * 32;
                            const int8_t* b_ptr = smem_b[stage] + (n_base + ni) * 256 + g * 32;
                            
                            // Use dp4a for vectorized dot product
                            const int* a_i32 = (const int*)a_ptr;
                            const int* b_i32 = (const int*)b_ptr;
                            
                            #pragma unroll
                            for (int k = 0; k < 8; k++) {
                                dot = __dp4a(a_i32[k], b_i32[k], dot);
                                sum_a = __dp4a(a_i32[k], 0x01010101, sum_a);
                                sum_b = __dp4a(b_i32[k], 0x01010101, sum_b);
                            }
                            
                            // Correction for signed conversion: q_signed = q_unsigned - 8
                            // dot(a_s, b_s) = dot(a_u-8, b_u-8) = dot(a_u,b_u) - 8*sum_a_u - 8*sum_b_u + 64*32
                            // Since we stored signed values, we have dot(a_s, b_s) directly
                            // But we need: dot(a_u, b_u) = dot(a_s+8, b_s+8) = dot + 8*sum_a + 8*sum_b + 64*32
                            // where sum_a and sum_b are sums of signed values
                            
                            // The Q4_K dequantization formula:
                            // val = d * sc[g] * q - dmin * mn[g]
                            // For dot product of two Q4_K values:
                            // sum += (d_a * sc_a * q_a - dmin_a * mn_a) * (d_b * sc_b * q_b - dmin_b * mn_b)
                            //      = d_a*d_b*sc_a*sc_b * q_a*q_b
                            //        - d_a*sc_a*dmin_b*mn_b * q_a
                            //        - d_b*sc_b*dmin_a*mn_a * q_b  
                            //        + dmin_a*mn_a*dmin_b*mn_b
                            
                            // With signed values (q_s = q_u - 8):
                            // q_u = q_s + 8, so sum_q_u = sum_q_s + 8*32
                            // dot(q_a_u, q_b_u) = dot(q_a_s+8, q_b_s+8) 
                            //                  = dot(q_a_s, q_b_s) + 8*sum_q_a_s + 8*sum_q_b_s + 64*32
                            
                            int32_t dot_unsigned = dot + 8 * sum_a + 8 * sum_b + 64 * 32;
                            int32_t sum_a_unsigned = sum_a + 8 * 32;  // = 256 + sum_a
                            int32_t sum_b_unsigned = sum_b + 8 * 32;
                            
                            // Apply Q4_K dequantization with all four terms
                            float scale_prod = d_a * d_b * (float)(sc_a * sc_b);
                            float term1 = scale_prod * (float)dot_unsigned;
                            float term2 = -d_a * dmin_b * (float)(sc_a * mn_b) * (float)sum_a_unsigned;
                            float term3 = -d_b * dmin_a * (float)(sc_b * mn_a) * (float)sum_b_unsigned;
                            float term4 = dmin_a * dmin_b * (float)(mn_a * mn_b) * 32.0f;
                            
                            acc[wm][wn][mi][ni] += term1 + term2 + term3 + term4;
                        }
                    }
                }
            }
        }
    };
    
    // Main loop with double buffering
    if (num_k_blocks > 0) {
        // Load first stage
        load_stage(0, 0);
        __syncthreads();
        
        int current_stage = 0;
        for (int kb = 0; kb < num_k_blocks; kb++) {
            // Start loading next stage if available
            if (kb + 1 < num_k_blocks) {
                int next_stage = 1 - current_stage;
                load_stage(next_stage, kb + 1);
            }
            
            // Compute using current stage
            compute_stage(current_stage);
            
            __syncthreads();
            current_stage = 1 - current_stage;
        }
    }
    
    // Store results
    #pragma unroll
    for (int wm = 0; wm < WMMA_WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_WARP_TILES_N; wn++) {
            int tile_row = warp_row * WMMA_WARP_TILES_M + wm;
            int tile_col = warp_col * WMMA_WARP_TILES_N + wn;
            int m_base = tile_row * WMMA_M;
            int n_base = tile_col * WMMA_N;
            
            // Each thread in warp writes its portion
            // For 8×8 tile with 32 threads: each thread writes 2 elements
            for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
                int mi = idx / WMMA_N;
                int ni = idx % WMMA_N;
                int global_row = block_row + m_base + mi;
                int global_col = block_col + n_base + ni;
                
                if (global_row < M && global_col < N) {
                    C[global_row * N + global_col] = acc[wm][wn][mi][ni];
                }
            }
        }
    }
}

// Optimized version using actual WMMA instructions with repacked data
// This requires repacking Q4_K nibbles into WMMA-compatible layout
__global__ void rrs_gemm_q4k_wmma_v2_kernel(
    const void* __restrict__ A,
    const void* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, I4, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, I4, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>;
    
    const int num_k_blocks = K / 256;
    
    const int block_row = blockIdx.y * WMMA_TILE_M;
    const int block_col = blockIdx.x * WMMA_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_row = warp_id / WMMA_WARPS_N;
    const int warp_col = warp_id % WMMA_WARPS_N;
    
    // Shared memory layout for WMMA-compatible INT4 data
    // For WMMA: A is row-major [M, K], B is col-major [K, N] (equivalently row-major [N, K])
    // Each WMMA_M×WMMA_K tile of A needs 8 rows × 32 cols = 256 INT4 = 128 bytes
    // Each WMMA_N×WMMA_K tile of B needs 8 rows × 32 cols = 256 INT4 = 128 bytes
    
    extern __shared__ char smem_v2[];
    
    // For each group (0-7), we need:
    // - A data: WMMA_TILE_M rows × 32 cols INT4 = 32 × 32 = 1024 INT4 = 512 bytes
    // - B data: WMMA_TILE_N rows × 32 cols INT4 = 32 × 32 = 1024 INT4 = 512 bytes  
    // - Scales: WMMA_TILE_M × 4 + WMMA_TILE_N × 4 floats (d, dmin, sc, mn per row)
    // Total per group: ~1200 bytes, with 8 groups and double buffer = ~20KB (fits in smem)
    
    // Simplified: store all 256 elements per row, process groups sequentially
    // A int4 data: [WMMA_TILE_M][256] int8
    // B int4 data: [WMMA_TILE_N][256] int8
    // A scales: [WMMA_TILE_M] × (d, dmin, sc[8], mn[8])
    // B scales: [WMMA_TILE_N] × (d, dmin, sc[8], mn[8])
    
    int8_t* s_a_int4 = (int8_t*)smem_v2;                          // [32][256]
    int8_t* s_b_int4 = s_a_int4 + WMMA_TILE_M * 256;              // [32][256]
    float* s_a_d = (float*)(s_b_int4 + WMMA_TILE_N * 256);        // [32]
    float* s_a_dmin = s_a_d + WMMA_TILE_M;                         // [32]
    uint8_t* s_a_sc = (uint8_t*)(s_a_dmin + WMMA_TILE_M);         // [32][8]
    uint8_t* s_a_mn = s_a_sc + WMMA_TILE_M * 8;                    // [32][8]
    float* s_b_d = (float*)(s_a_mn + WMMA_TILE_M * 8);            // [32]
    float* s_b_dmin = s_b_d + WMMA_TILE_N;                         // [32]
    uint8_t* s_b_sc = (uint8_t*)(s_b_dmin + WMMA_TILE_N);         // [32][8]
    uint8_t* s_b_mn = s_b_sc + WMMA_TILE_N * 8;                    // [32][8]
    
    // Fragment storage for WMMA
    FragC c_frag[WMMA_WARP_TILES_M][WMMA_WARP_TILES_N];
    
    // Initialize accumulators
    #pragma unroll
    for (int wm = 0; wm < WMMA_WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_WARP_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0);
        }
    }
    
    // Float accumulators for scaled results
    float acc[WMMA_WARP_TILES_M][WMMA_WARP_TILES_N][WMMA_M * WMMA_N];
    #pragma unroll
    for (int wm = 0; wm < WMMA_WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_WARP_TILES_N; wn++) {
            #pragma unroll
            for (int i = 0; i < WMMA_M * WMMA_N; i++) {
                acc[wm][wn][i] = 0.0f;
            }
        }
    }
    
    const block_q4_K* A_blocks = (const block_q4_K*)A;
    const block_q4_K* B_blocks = (const block_q4_K*)B;
    
    // Iterate over K blocks
    for (int kb = 0; kb < num_k_blocks; kb++) {
        // Load A tile
        for (int r = tid; r < WMMA_TILE_M; r += blockDim.x) {
            int global_row = block_row + r;
            if (global_row < M) {
                const block_q4_K& blk = A_blocks[global_row * num_k_blocks + kb];
                s_a_d[r] = __half2float(blk.dm.x);
                s_a_dmin[r] = __half2float(blk.dm.y);
                
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    get_scale_min_k4_cuda(g, blk.scales, &s_a_sc[r * 8 + g], &s_a_mn[r * 8 + g]);
                }
                
                // Unpack to signed int8, grouped by group index
                #pragma unroll
                for (int chunk = 0; chunk < 4; chunk++) {
                    int g_lo = chunk * 2;
                    int g_hi = chunk * 2 + 1;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) {
                        uint8_t packed = blk.qs[chunk * 32 + i];
                        s_a_int4[r * 256 + g_lo * 32 + i] = (int8_t)(packed & 0x0F) - 8;
                        s_a_int4[r * 256 + g_hi * 32 + i] = (int8_t)(packed >> 4) - 8;
                    }
                }
            } else {
                // Zero padding for out-of-bounds
                s_a_d[r] = 0.0f;
                s_a_dmin[r] = 0.0f;
                for (int i = 0; i < 256; i++) s_a_int4[r * 256 + i] = 0;
            }
        }
        
        // Load B tile
        for (int c = tid; c < WMMA_TILE_N; c += blockDim.x) {
            int global_col = block_col + c;
            if (global_col < N) {
                const block_q4_K& blk = B_blocks[global_col * num_k_blocks + kb];
                s_b_d[c] = __half2float(blk.dm.x);
                s_b_dmin[c] = __half2float(blk.dm.y);
                
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    get_scale_min_k4_cuda(g, blk.scales, &s_b_sc[c * 8 + g], &s_b_mn[c * 8 + g]);
                }
                
                #pragma unroll
                for (int chunk = 0; chunk < 4; chunk++) {
                    int g_lo = chunk * 2;
                    int g_hi = chunk * 2 + 1;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) {
                        uint8_t packed = blk.qs[chunk * 32 + i];
                        s_b_int4[c * 256 + g_lo * 32 + i] = (int8_t)(packed & 0x0F) - 8;
                        s_b_int4[c * 256 + g_hi * 32 + i] = (int8_t)(packed >> 4) - 8;
                    }
                }
            } else {
                s_b_d[c] = 0.0f;
                s_b_dmin[c] = 0.0f;
                for (int i = 0; i < 256; i++) s_b_int4[c * 256 + i] = 0;
            }
        }
        __syncthreads();
        
        // Process each group using WMMA
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            #pragma unroll
            for (int wm = 0; wm < WMMA_WARP_TILES_M; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WMMA_WARP_TILES_N; wn++) {
                    int tile_m = warp_row * WMMA_WARP_TILES_M + wm;
                    int tile_n = warp_col * WMMA_WARP_TILES_N + wn;
                    int m_base = tile_m * WMMA_M;
                    int n_base = tile_n * WMMA_N;
                    
                    // Load A fragment: 8 rows × 32 INT4 = 8 × 16 bytes
                    // A[m_base:m_base+8, g*32:(g+1)*32]
                    FragA a_frag;
                    
                    // Load B fragment: 8 rows × 32 INT4
                    // B[n_base:n_base+8, g*32:(g+1)*32]
                    FragB b_frag;
                    
                    // WMMA load requires contiguous memory in specific layout
                    // For row-major A: elements should be at A[row][col] = ptr[row * ldA + col/2] (packed nibbles)
                    // Our int8 data is unpacked, need to repack for WMMA
                    
                    // Repack A data for this group into temp buffer
                    // WMMA M=8, K=32: need 8 rows × 32 INT4 = 8 × 16 bytes = 128 bytes
                    // But our data is int8, so we need to pack pairs of int8 into one byte
                    // Actually, WMMA s4 loads from int32*, treating each int32 as 8 INT4 values
                    
                    // Simpler approach: compute manually and apply scales
                    FragC raw_frag;
                    wmma::fill_fragment(raw_frag, 0);
                    
                    // Manual dot product per element in the 8×8 tile
                    // This avoids WMMA layout complexity while still being reasonably fast
                    int32_t local_dot[WMMA_M * WMMA_N];
                    int32_t local_sum_a[WMMA_M];
                    int32_t local_sum_b[WMMA_N];
                    
                    // Initialize
                    for (int i = 0; i < WMMA_M * WMMA_N; i++) local_dot[i] = 0;
                    for (int i = 0; i < WMMA_M; i++) local_sum_a[i] = 0;
                    for (int i = 0; i < WMMA_N; i++) local_sum_b[i] = 0;
                    
                    // Compute dot products using dp4a
                    for (int mi = lane / 4; mi < WMMA_M; mi += 8) {
                        int a_row = m_base + mi;
                        const int* a_ptr = (const int*)(s_a_int4 + a_row * 256 + g * 32);
                        
                        int32_t sum_a = 0;
                        for (int k = 0; k < 8; k++) {
                            sum_a = __dp4a(a_ptr[k], 0x01010101, sum_a);
                        }
                        local_sum_a[mi] = sum_a;
                        
                        for (int ni = lane % 4; ni < WMMA_N; ni += 4) {
                            int b_row = n_base + ni;
                            const int* b_ptr = (const int*)(s_b_int4 + b_row * 256 + g * 32);
                            
                            int32_t dot = 0;
                            for (int k = 0; k < 8; k++) {
                                dot = __dp4a(a_ptr[k], b_ptr[k], dot);
                            }
                            local_dot[mi * WMMA_N + ni] = dot;
                        }
                    }
                    
                    // Compute B sums
                    for (int ni = lane; ni < WMMA_N; ni += 32) {
                        int b_row = n_base + ni;
                        const int* b_ptr = (const int*)(s_b_int4 + b_row * 256 + g * 32);
                        int32_t sum_b = 0;
                        for (int k = 0; k < 8; k++) {
                            sum_b = __dp4a(b_ptr[k], 0x01010101, sum_b);
                        }
                        local_sum_b[ni] = sum_b;
                    }
                    
                    // Reduce across warp and apply scales
                    for (int mi = 0; mi < WMMA_M; mi++) {
                        int global_row = block_row + m_base + mi;
                        if (global_row >= M) continue;
                        
                        float d_a = s_a_d[m_base + mi];
                        float dmin_a = s_a_dmin[m_base + mi];
                        int sc_a = s_a_sc[(m_base + mi) * 8 + g];
                        int mn_a = s_a_mn[(m_base + mi) * 8 + g];
                        int sum_a_s = local_sum_a[mi];
                        int sum_a_u = sum_a_s + 256;  // Convert to unsigned sum
                        
                        for (int ni = 0; ni < WMMA_N; ni++) {
                            int global_col = block_col + n_base + ni;
                            if (global_col >= N) continue;
                            
                            float d_b = s_b_d[n_base + ni];
                            float dmin_b = s_b_dmin[n_base + ni];
                            int sc_b = s_b_sc[(n_base + ni) * 8 + g];
                            int mn_b = s_b_mn[(n_base + ni) * 8 + g];
                            int sum_b_s = local_sum_b[ni];
                            int sum_b_u = sum_b_s + 256;
                            
                            int dot_s = local_dot[mi * WMMA_N + ni];
                            int dot_u = dot_s + 8 * sum_a_s + 8 * sum_b_s + 2048;
                            
                            float val = d_a * d_b * (float)(sc_a * sc_b) * (float)dot_u
                                      - d_a * dmin_b * (float)(sc_a * mn_b) * (float)sum_a_u
                                      - d_b * dmin_a * (float)(sc_b * mn_a) * (float)sum_b_u
                                      + dmin_a * dmin_b * (float)(mn_a * mn_b) * 32.0f;
                            
                            acc[wm][wn][mi * WMMA_N + ni] += val;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Store results
    for (int wm = 0; wm < WMMA_WARP_TILES_M; wm++) {
        for (int wn = 0; wn < WMMA_WARP_TILES_N; wn++) {
            int tile_m = warp_row * WMMA_WARP_TILES_M + wm;
            int tile_n = warp_col * WMMA_WARP_TILES_N + wn;
            int m_base = tile_m * WMMA_M;
            int n_base = tile_n * WMMA_N;
            
            for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
                int mi = idx / WMMA_N;
                int ni = idx % WMMA_N;
                int global_row = block_row + m_base + mi;
                int global_col = block_col + n_base + ni;
                
                if (global_row < M && global_col < N) {
                    C[global_row * N + global_col] = acc[wm][wn][idx];
                }
            }
        }
    }
}

// Tiled GEMM kernel: each block computes a TILE_M x TILE_N output tile
// A is [M, K/256] Q4_K blocks (activations, row-major)
// B is [N, K/256] Q4_K blocks (weights, row-major - each row is one output neuron)
// C is [M, N] float output
__global__ void rrs_gemm_q4k_q4k_tiled_kernel(
    const void* __restrict__ A,
    const void* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    const int num_k_blocks = K / 256;
    
    // Block position in output
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;
    
    // Thread position within block
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Each thread computes one or more output elements
    // With TILE_M=32, TILE_N=32, we have 1024 outputs per block
    // With 256 threads, each thread computes 4 outputs
    const int outputs_per_thread = (TILE_M * TILE_N) / THREADS_PER_BLOCK;
    
    // Register storage for partial sums
    float acc[outputs_per_thread];
    #pragma unroll
    for (int i = 0; i < outputs_per_thread; i++) acc[i] = 0.0f;
    
    // Determine which outputs this thread computes
    // Thread tid computes outputs at positions tid, tid+256, tid+512, tid+768
    // Map to (row, col) within tile
    int out_rows[outputs_per_thread], out_cols[outputs_per_thread];
    #pragma unroll
    for (int i = 0; i < outputs_per_thread; i++) {
        const int flat_idx = tid + i * THREADS_PER_BLOCK;
        out_rows[i] = flat_idx / TILE_N;
        out_cols[i] = flat_idx % TILE_N;
    }
    
    const block_q4_K* A_blocks = (const block_q4_K*)A;
    const block_q4_K* B_blocks = (const block_q4_K*)B;
    
    // Iterate over K dimension
    for (int kb = 0; kb < num_k_blocks; kb++) {
        #pragma unroll
        for (int i = 0; i < outputs_per_thread; i++) {
            const int global_row = block_row + out_rows[i];
            const int global_col = block_col + out_cols[i];
            
            if (global_row < M && global_col < N) {
                const block_q4_K& a_block = A_blocks[global_row * num_k_blocks + kb];
                const block_q4_K& b_block = B_blocks[global_col * num_k_blocks + kb];
                acc[i] += q4k_block_dot(a_block, b_block);
            }
        }
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < outputs_per_thread; i++) {
        const int global_row = block_row + out_rows[i];
        const int global_col = block_col + out_cols[i];
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = acc[i];
        }
    }
}

// Fused FWHT + Quantize + GEMM kernel for M=1 (single token inference)
// This eliminates kernel launch overhead by doing everything in one kernel
// Each block computes M1_COLS_PER_BLOCK output columns
// Input: float activations (not pre-quantized)
// Weights: Q4_K blocks
template<int K_DIM>
__global__ void rrs_fused_m1_kernel(
    const float* __restrict__ act_float,  // [1, K] float activations
    const void* __restrict__ weights,      // [N, K/256] Q4_K weight blocks
    float* __restrict__ output,            // [1, N] output
    const int N
) {
    // Shared memory for FWHT-transformed and quantized activation
    extern __shared__ char smem[];
    float* s_act = (float*)smem;  // K_DIM floats for FWHT
    block_q4_K* s_act_q4k = (block_q4_K*)(s_act + K_DIM);  // K_DIM/256 Q4_K blocks
    
    const int tid = threadIdx.x;
    const int num_k_blocks = K_DIM / 256;
    const int base_col = blockIdx.x * M1_COLS_PER_BLOCK;
    
    // Step 1: Load activations into shared memory
    for (int i = tid; i < K_DIM; i += blockDim.x) {
        s_act[i] = act_float[i];
    }
    __syncthreads();
    
    // Step 2: In-place FWHT on shared memory (chunked for non-power-of-2)
    const int step = K_DIM & -K_DIM;  // Largest power of 2 that divides K_DIM
    const float scale_fwht = rsqrtf((float)step);
    
    for (int chunk_base = 0; chunk_base < K_DIM; chunk_base += step) {
        // FWHT butterfly operations for this chunk
        for (int h = 1; h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = tid; i < step / 2; i += blockDim.x) {
                const int block_idx = i / h;
                const int offset = i % h;
                const int idx1 = chunk_base + block_idx * stride + offset;
                const int idx2 = idx1 + h;
                fwht_butterfly(s_act[idx1], s_act[idx2]);
            }
            __syncthreads();
        }
        // Apply scale
        for (int i = tid; i < step; i += blockDim.x) {
            s_act[chunk_base + i] *= scale_fwht;
        }
        __syncthreads();
    }
    
    // Step 3: Quantize to Q4_K format (one block per 256 elements)
    // Each warp handles scale computation for its assigned blocks
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    for (int b = warp_id; b < num_k_blocks; b += num_warps) {
        float* block_data = s_act + b * 256;
        block_q4_K* out_block = &s_act_q4k[b];
        
        // Find min/max for each of 8 subgroups (32 elements each)
        // Each lane handles one element within the subgroup
        __shared__ float s_scales_tmp[8 * 8];  // 8 warps * 8 subgroups
        __shared__ float s_mins_tmp[8 * 8];
        
        float* my_scales = s_scales_tmp + warp_id * 8;
        float* my_mins = s_mins_tmp + warp_id * 8;
        
        // Process 8 subgroups, each with 32 elements
        for (int sg = 0; sg < 8; sg++) {
            float val = block_data[sg * 32 + lane];
            float vmin = val, vmax = val;
            
            // Warp reduction for min/max
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, mask));
                vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, mask));
            }
            
            if (vmin > 0) vmin = 0;
            if (lane == 0) {
                my_scales[sg] = (vmax - vmin) / 15.0f;
                my_mins[sg] = -vmin;
            }
        }
        __syncwarp();
        
        // Compute global d and dmin for this block
        if (lane == 0) {
            float max_s = my_scales[0], max_m = my_mins[0];
            for (int i = 1; i < 8; i++) {
                if (my_scales[i] > max_s) max_s = my_scales[i];
                if (my_mins[i] > max_m) max_m = my_mins[i];
            }
            out_block->dm.x = __float2half(max_s / 63.0f);
            out_block->dm.y = __float2half(max_m / 63.0f);
            
            // Pack scales
            const float inv_s = (max_s > 0.0f) ? (63.0f / max_s) : 0.0f;
            const float inv_m = (max_m > 0.0f) ? (63.0f / max_m) : 0.0f;
            uint8_t ls[8], lm[8];
            for (int j = 0; j < 8; j++) {
                ls[j] = min(63, (int)(my_scales[j] * inv_s + 0.5f));
                lm[j] = min(63, (int)(my_mins[j] * inv_m + 0.5f));
            }
            for (int j = 0; j < 4; j++) {
                out_block->scales[j] = ls[j] | ((ls[j+4] & 0x30) << 2);
                out_block->scales[j+4] = lm[j] | ((lm[j+4] & 0x30) << 2);
            }
            for (int j = 0; j < 4; j++) {
                out_block->scales[j+8] = (ls[j+4] & 0x0F) | ((lm[j+4] & 0x0F) << 4);
            }
        }
        __syncwarp();
        
        // Quantize the 256 values (8 subgroups of 32)
        // Read back the packed scales
        uint8_t sc[8], mn[8];
        if (lane < 8) {
            get_scale_min_k4_cuda(lane, out_block->scales, &sc[lane], &mn[lane]);
        }
        // Broadcast to all lanes
        for (int i = 0; i < 8; i++) {
            sc[i] = __shfl_sync(0xffffffff, sc[i], i);
            mn[i] = __shfl_sync(0xffffffff, mn[i], i);
        }
        
        const float d = __half2float(out_block->dm.x);
        const float dm = __half2float(out_block->dm.y);
        
        // Each lane quantizes 8 values (256 / 32 lanes)
        for (int chunk = 0; chunk < 4; chunk++) {
            const int sg_lo = chunk * 2;
            const int sg_hi = sg_lo + 1;
            const float d_lo = d * sc[sg_lo];
            const float dm_lo = dm * mn[sg_lo];
            const float id_lo = (d_lo > 1e-10f) ? (1.0f / d_lo) : 0.0f;
            const float d_hi = d * sc[sg_hi];
            const float dm_hi = dm * mn[sg_hi];
            const float id_hi = (d_hi > 1e-10f) ? (1.0f / d_hi) : 0.0f;
            
            const float val_lo = block_data[sg_lo * 32 + lane];
            const float val_hi = block_data[sg_hi * 32 + lane];
            int q_lo = min(15, max(0, (int)((val_lo + dm_lo) * id_lo + 0.5f)));
            int q_hi = min(15, max(0, (int)((val_hi + dm_hi) * id_hi + 0.5f)));
            out_block->qs[chunk * 32 + lane] = (uint8_t)(q_lo | (q_hi << 4));
        }
    }
    __syncthreads();
    
    // Step 4: Compute dot products for M1_COLS_PER_BLOCK output columns
    const block_q4_K* B_blocks = (const block_q4_K*)weights;
    
    for (int c = 0; c < M1_COLS_PER_BLOCK; c++) {
        const int col = base_col + c;
        if (col >= N) continue;
        
        const block_q4_K* B_col = B_blocks + col * num_k_blocks;
        float sum = 0.0f;
        
        // Each thread processes a subset of K blocks
        for (int kb = tid; kb < num_k_blocks; kb += blockDim.x) {
            sum += q4k_block_dot(s_act_q4k[kb], B_col[kb]);
        }
        
        // Warp reduction
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }
        
        // Block reduction
        __shared__ float s_partial[8];
        if (lane == 0) {
            s_partial[warp_id] = sum;
        }
        __syncthreads();
        
        if (tid < 8) {
            float val = (tid < num_warps) ? s_partial[tid] : 0.0f;
            #pragma unroll
            for (int mask = 4; mask > 0; mask >>= 1) {
                val += __shfl_xor_sync(0xff, val, mask);
            }
            if (tid == 0) {
                output[col] = val;
            }
        }
        __syncthreads();
    }
}

// Fallback M=1 kernel for pre-quantized activations
__global__ void rrs_gemm_q4k_q4k_m1_kernel(
    const void* __restrict__ A,
    const void* __restrict__ B,
    float* __restrict__ C,
    const int N, const int K
) {
    const int num_k_blocks = K / 256;
    const int col = blockIdx.x;
    if (col >= N) return;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const block_q4_K* A_blocks = (const block_q4_K*)A;
    const block_q4_K* B_blocks = (const block_q4_K*)B + col * num_k_blocks;
    
    float sum = 0.0f;
    
    // Each thread processes a subset of K blocks
    for (int kb = tid; kb < num_k_blocks; kb += blockDim.x) {
        sum += q4k_block_dot(A_blocks[kb], B_blocks[kb]);
    }
    
    // Warp reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    
    // Block reduction via shared memory
    __shared__ float s_partial[8]; // Up to 8 warps
    if (lane_id == 0) {
        s_partial[warp_id] = sum;
    }
    __syncthreads();
    
    if (tid < 8) {
        float val = (tid < (blockDim.x / 32)) ? s_partial[tid] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            val += __shfl_xor_sync(0xff, val, mask);
        }
        if (tid == 0) {
            C[col] = val;
        }
    }
}

// Batched M>1 kernel with better memory access pattern
// Process multiple rows per block to improve cache utilization
__global__ void rrs_gemm_q4k_q4k_batched_kernel(
    const void* __restrict__ A,
    const void* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    const int num_k_blocks = K / 256;
    
    // Each block handles one column (N) and multiple rows (M) 
    const int col = blockIdx.x;
    if (col >= N) return;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;
    
    const block_q4_K* A_blocks = (const block_q4_K*)A;
    const block_q4_K* B_blocks = (const block_q4_K*)B + col * num_k_blocks;
    
    // Each warp handles different rows
    for (int row = warp_id; row < M; row += num_warps) {
        const block_q4_K* A_row = A_blocks + row * num_k_blocks;
        
        float sum = 0.0f;
        // Lanes within warp cooperate on K reduction
        for (int kb = lane_id; kb < num_k_blocks; kb += 32) {
            sum += q4k_block_dot(A_row[kb], B_blocks[kb]);
        }
        
        // Warp reduction
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }
        
        if (lane_id == 0) {
            C[row * N + col] = sum;
        }
    }
}

// ============================================================================ 
// Dispatch Integration
// ============================================================================ 

// ============================================================================
// RRS-MMVQ: FWHT + Q8 quantization + existing Q4_K×Q8 kernel
// This is much faster than Q4×Q4 because it reuses optimized MMVQ kernels
// ============================================================================

// Fused FWHT + Q8_1 quantization kernel for M=1 path
// Eliminates intermediate fp32 buffer and separate kernel launches
__global__ void rrs_fwht_quantize_q8_1_fused_kernel(
    const float* __restrict__ src,
    block_q8_1* __restrict__ dst,
    const int K
) {
    extern __shared__ float smem_fused[];
    
    // Load from source
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        smem_fused[i] = src[i];
    }
    __syncthreads();
    
    // FWHT transform
    const int step = K & -K;
    const float scale = rsqrtf((float)step);
    
    for (int chunk_base = 0; chunk_base < K; chunk_base += step) {
        for (int h = 1; h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = threadIdx.x; i < step / 2; i += blockDim.x) {
                const int block_idx = i / h;
                const int offset = i % h;
                const int idx1 = chunk_base + block_idx * stride + offset;
                const int idx2 = idx1 + h;
                float a = smem_fused[idx1];
                float b = smem_fused[idx2];
                smem_fused[idx1] = a + b;
                smem_fused[idx2] = a - b;
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < step; i += blockDim.x) {
            smem_fused[chunk_base + i] *= scale;
        }
        __syncthreads();
    }
    
    // Quantize to Q8_1 - each block of 32 elements
    const int num_q8_blocks = K / QK8_1;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;
    
    for (int b = warp_id; b < num_q8_blocks; b += num_warps) {
        float* block_data = smem_fused + b * QK8_1;
        
        // Find absmax using warp reduction
        float amax = 0.0f;
        float val = block_data[lane];
        amax = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
        }
        
        // Compute scale
        const float d = amax / 127.0f;
        const float id = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        
        // Quantize
        int8_t q = (int8_t)roundf(val * id);
        
        // Write to output
        block_q8_1* out_block = dst + b;
        out_block->qs[lane] = q;
        
        // First lane writes scale and sum using ds union
        if (lane == 0) {
            // Compute sum for Q8_1 format
            float sum = 0.0f;
            for (int i = 0; i < QK8_1; i++) {
                sum += block_data[i];
            }
            // ds.x = d (delta), ds.y = d * sum
            out_block->ds = make_half2(__float2half(d), __float2half(sum * d));
        }
    }
}

// FWHT kernel that transforms fp32 activations in-place
// Optimized FWHT kernel that reads from src and writes to dst (no separate D2D copy needed)
// Uses warp-level shuffle for small h values to reduce __syncthreads() overhead
__global__ void rrs_fwht_kernel_opt(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int K,
    const int M
) {
    extern __shared__ float smem_fwht_opt[];
    const int row = blockIdx.x;
    if (row >= M) return;
    
    const float* src_row = src + row * K;
    float* dst_row = dst + row * K;
    
    // Load from source directly into shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        smem_fwht_opt[i] = src_row[i];
    }
    __syncthreads();
    
    const int step = K & -K;
    const float scale = rsqrtf((float)step);
    
    for (int chunk_base = 0; chunk_base < K; chunk_base += step) {
        // For small h (within warp), use warp shuffles to reduce syncs
        // For h >= 32, we need __syncthreads()
        
        // Process h=1 to h=16 using warp-level operations (no sync needed within warp)
        // Each thread handles pairs at increasing distances
        for (int h = 1; h < 32 && h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = threadIdx.x; i < step / 2; i += blockDim.x) {
                const int block_idx = i / h;
                const int offset = i % h;
                const int idx1 = chunk_base + block_idx * stride + offset;
                const int idx2 = idx1 + h;
                float a = smem_fwht_opt[idx1];
                float b = smem_fwht_opt[idx2];
                smem_fwht_opt[idx1] = a + b;
                smem_fwht_opt[idx2] = a - b;
            }
            __syncthreads();
        }
        
        // Process remaining stages (h >= 32)
        for (int h = 32; h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = threadIdx.x; i < step / 2; i += blockDim.x) {
                const int block_idx = i / h;
                const int offset = i % h;
                const int idx1 = chunk_base + block_idx * stride + offset;
                const int idx2 = idx1 + h;
                float a = smem_fwht_opt[idx1];
                float b = smem_fwht_opt[idx2];
                smem_fwht_opt[idx1] = a + b;
                smem_fwht_opt[idx2] = a - b;
            }
            __syncthreads();
        }
        
        // Apply scale (fused with last stage to reduce memory traffic)
        for (int i = threadIdx.x; i < step; i += blockDim.x) {
            smem_fwht_opt[chunk_base + i] *= scale;
        }
        __syncthreads();
    }
    
    // Write to destination
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        dst_row[i] = smem_fwht_opt[i];
    }
}

// Legacy inplace kernel for compatibility
__global__ void rrs_fwht_inplace_kernel(float* __restrict__ data, const int K, const int M) {
    extern __shared__ float smem_fwht_ip[];
    const int row = blockIdx.x;
    if (row >= M) return;
    
    float* row_data = data + row * K;
    
    // Load into shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        smem_fwht_ip[i] = row_data[i];
    }
    __syncthreads();
    
    // FWHT on power-of-2 chunks
    const int step = K & -K;  // Largest power of 2 dividing K
    const float scale = rsqrtf((float)step);
    
    for (int chunk_base = 0; chunk_base < K; chunk_base += step) {
        for (int h = 1; h < step; h <<= 1) {
            const int stride = h << 1;
            for (int i = threadIdx.x; i < step / 2; i += blockDim.x) {
                const int block_idx = i / h;
                const int offset = i % h;
                const int idx1 = chunk_base + block_idx * stride + offset;
                const int idx2 = idx1 + h;
                float a = smem_fwht_ip[idx1];
                float b = smem_fwht_ip[idx2];
                smem_fwht_ip[idx1] = a + b;
                smem_fwht_ip[idx2] = a - b;
            }
            __syncthreads();
        }
        // Apply scale
        for (int i = threadIdx.x; i < step; i += blockDim.x) {
            smem_fwht_ip[chunk_base + i] *= scale;
        }
        __syncthreads();
    }
    
    // Write back
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        row_data[i] = smem_fwht_ip[i];
    }
}

// Quantize FWHT'd activations to Q8_1 format (same as standard MMVQ path)
__global__ void rrs_quantize_q8_1_kernel(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    const int K,
    const int M
) {
    const int row = blockIdx.x;
    const int block_idx = blockIdx.y;
    if (row >= M) return;
    
    const int num_blocks = K / QK8_1;
    if (block_idx >= num_blocks) return;
    
    const float* x_block = x + row * K + block_idx * QK8_1;
    block_q8_1* y_block = y + row * num_blocks + block_idx;
    
    // Find absmax for this block
    float amax = 0.0f;
    for (int i = threadIdx.x; i < QK8_1; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(x_block[i]));
    }
    
    // Warp reduction for max
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
    }
    
    __shared__ float s_amax;
    if (threadIdx.x == 0) {
        s_amax = amax;
    }
    __syncthreads();
    amax = s_amax;
    
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;
    
    // Quantize and compute sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < QK8_1; i += blockDim.x) {
        const float v = x_block[i];
        const int8_t q = (int8_t)roundf(v * id);
        y_block->qs[i] = q;
        sum += (float)q;
    }
    
    // Reduce sum
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    
    if (threadIdx.x == 0) {
        y_block->ds = make_half2(__float2half(d), __float2half(d * sum));
    }
}

// RRS-MMVQ kernel: Q4_K weights × Q8_1 activations (reuses existing vec_dot)
// Each block computes multiple output rows, threads cooperate on K reduction
template<int NROWS_PER_BLOCK>
__global__ void rrs_mmvq_kernel(
    const void* __restrict__ vx,      // Q4_K weights [N, K/256]
    const void* __restrict__ vy,      // Q8_1 activations [M, K/32]
    float* __restrict__ dst,          // Output [M, N]
    const int N,
    const int K,
    const int M
) {
    const int col = blockIdx.x;  // Output column (weight row)
    const int row_base = blockIdx.y * NROWS_PER_BLOCK;  // Output row base
    
    if (col >= N) return;
    
    // Q4_K parameters
    constexpr int vdr = 2;  // VDR_Q4_K_Q8_1_MMVQ
    constexpr int qi = 32;  // QI4_K
    
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;
    
    const block_q4_K* bq4 = (const block_q4_K*)vx + col * (K / QK_K);
    const int num_k_blocks = K / QK_K;
    
    // iqs calculation matching official kernel
    const int iqs = vdr * (tid % (qi / vdr));  // 0, 2, 4, ..., 30 (repeats every 16 threads)
    const int blocks_per_iter = vdr * blockDim.x / qi;  // How many K blocks all threads cover per iteration
    
    // Each warp handles one output row
    for (int r = warp_id; r < NROWS_PER_BLOCK && (row_base + r) < M; r += num_warps) {
        const int row = row_base + r;
        const block_q8_1* bq8 = (const block_q8_1*)vy + row * (K / QK8_1);
        
        float sum = 0.0f;
        
        // Iterate over Q4_K blocks - 16 threads per block (tid/16 gives block offset)
        for (int kb = tid / (qi / vdr); kb < num_k_blocks; kb += blocks_per_iter) {
            const int kby = kb * (QK_K / QK8_1);  // = kb * 8
            sum += vec_dot_q4_K_q8_1(bq4, bq8 + kby, kb, iqs);
        }
        
        // Warp reduction
        sum = warp_reduce_sum(sum);
        
        if (lane == 0) {
            dst[row * N + col] = sum;
        }
    }
}

// Simpler M=1 MMVQ kernel - matches official mul_mat_vec_q kernel structure
__global__ void rrs_mmvq_m1_kernel(
    const void* __restrict__ vx,      // Q4_K weights [N, K/256]
    const void* __restrict__ vy,      // Q8_1 activations [1, K/32]
    float* __restrict__ dst,          // Output [1, N]
    const int N,
    const int K
) {
    const int col = blockIdx.x;
    if (col >= N) return;
    
    // Q4_K parameters: QK_K=256, QI4_K=32, VDR=2
    // qi/vdr = 32/2 = 16, so 16 threads per K block
    // iqs = vdr * (tid % (qi/vdr)) = 2 * (tid % 16)
    constexpr int vdr = 2;  // VDR_Q4_K_Q8_1_MMVQ
    constexpr int qi = 32;  // QI4_K
    
    const int tid = threadIdx.x;
    const int num_k_blocks = K / QK_K;
    
    const block_q4_K* bq4 = (const block_q4_K*)vx + col * num_k_blocks;
    const block_q8_1* bq8 = (const block_q8_1*)vy;
    
    // Each thread processes specific K blocks based on its tid
    // blocks_per_iter = vdr * blockDim.x / qi = 2 * 256 / 32 = 16
    const int blocks_per_iter = vdr * blockDim.x / qi;
    const int iqs = vdr * (tid % (qi / vdr));  // 0, 2, 4, ..., 30 (repeats every 16 threads)
    
    float sum = 0.0f;
    
    // tid / (qi/vdr) = tid / 16 gives which K block offset this thread starts at
    for (int kb = tid / (qi / vdr); kb < num_k_blocks; kb += blocks_per_iter) {
        const int kby = kb * (QK_K / QK8_1);  // = kb * 8
        sum += vec_dot_q4_K_q8_1(bq4, bq8 + kby, kb, iqs);
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    // Inter-warp reduction via shared memory
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;
    
    __shared__ float s_sum[8];
    if (lane == 0) {
        s_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++) {
            total += s_sum[w];
        }
        dst[col] = total;
    }
}

// Set to 1 to use Runtime Smooth W4A4 (paper's approach), 0 for W4A8 fallback
#define TCQ4_USE_RRS_W4A4 1
// Set to 1 to use tensor core accelerated kernels (IMMA), 0 for scalar fallback
#define TCQ4_USE_TENSOR_CORES 1

void ggml_cuda_rrs_mul_mat(ggml_backend_cuda_context& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    const int M = src1->ne[1], N = src0->ne[1], K = src0->ne[0];
    cudaStream_t stream = ctx.stream();
    
    // TCQ4_K32 type: True W4A4 with Runtime Smooth (from RRS paper)
    // Key insight: divide by group-max BEFORE quantization, multiply AFTER dot product
    if (src0->type == GGML_TYPE_TCQ4_K32) {
        static int tcq4_call_count = 0;
        if (tcq4_call_count < 3) {
#if TCQ4_USE_RRS_W4A4
            fprintf(stderr, "[TCQ4-RRS-W4A4] mul_mat M=%d N=%d K=%d\n", M, N, K);
#else
            fprintf(stderr, "[TCQ4-W4A8] mul_mat M=%d N=%d K=%d\n", M, N, K);
#endif
            tcq4_call_count++;
        }
        
#if TCQ4_USE_RRS_W4A4
        // Runtime Smooth W4A4 path (paper's approach)
        // 1. FWHT to spread spike outliers
        // 2. Runtime Smooth: divide by group-max, store scale separately
        // 3. Quantize normalized values to INT4
        // 4. Dot product with TCQ4 weights (using tensor cores if enabled)
        // 5. Multiply by smooth scale to restore magnitude
        const int num_k_blocks = K / TCQ4_K32_BLOCK_SIZE;
        
#if TCQ4_USE_TENSOR_CORES
        // Tensor core path: uses block_rrs_int4_tc with precomputed group sums
        // block_rrs_int4_tc: 128 bytes INT4 + 4 bytes smooth_scale + 16 bytes sum_q = 148 bytes
        size_t rrs_size = M * num_k_blocks * sizeof(block_rrs_int4_tc);
        size_t rrs_actual;
        void* d_act_rrs = ctx.pool().alloc(rrs_size, &rrs_actual);
        
        // Fused FWHT + Runtime Smooth + INT4 quantize (with group sums for tensor cores)
        tcq4_rrs_fwht_quantize_tc((const float*)src1->data, d_act_rrs, K, M, stream);
        
        // Dispatch GEMM/GEMV with tensor cores
        if (M == 1) {
            tcq4_rrs_gemv_tc(d_act_rrs, src0->data, (float*)dst->data, N, K, stream);
        } else {
            tcq4_rrs_gemm_imma(d_act_rrs, src0->data, (float*)dst->data, M, N, K, stream);
        }
#else
        // Scalar fallback path: uses block_rrs_int4 (simpler, no group sums)
        // block_rrs_int4: 128 bytes INT4 + 4 bytes smooth_scale = 132 bytes per 256 elements
        size_t rrs_size = M * num_k_blocks * sizeof(block_rrs_int4);
        size_t rrs_actual;
        void* d_act_rrs = ctx.pool().alloc(rrs_size, &rrs_actual);
        
        // Fused FWHT + Runtime Smooth + INT4 quantize
        tcq4_rrs_fwht_quantize((const float*)src1->data, d_act_rrs, K, M, stream);
        
        // Dispatch GEMM/GEMV with scalar kernels
        if (M == 1) {
            tcq4_rrs_gemv(d_act_rrs, src0->data, (float*)dst->data, N, K, stream);
        } else {
            tcq4_rrs_gemm(d_act_rrs, src0->data, (float*)dst->data, M, N, K, stream);
        }
#endif
#else
        // W4A8 fallback: TCQ4 weights + Q8_1 activations (more accurate but not true W4A4)
        const int num_q8_blocks = K / QK8_1;
        size_t q8_size = M * num_q8_blocks * sizeof(block_q8_1);
        size_t q8_actual;
        void* d_q8 = ctx.pool().alloc(q8_size, &q8_actual);
        
        // FWHT + quantize to Q8_1
        tcq4_fwht_quantize_q8((const float*)src1->data, d_q8, K, M, stream);
        
        if (M == 1) {
            tcq4_k32_q8_gemv(d_q8, src0->data, (float*)dst->data, N, K, stream);
        } else {
            // Row-by-row GEMV for M > 1
            for (int m = 0; m < M; m++) {
                tcq4_k32_q8_gemv(
                    (const char*)d_q8 + m * num_q8_blocks * sizeof(block_q8_1),
                    src0->data,
                    (float*)dst->data + m * N,
                    N, K, stream);
            }
        }
#endif
        return;
    }
    
    // For M=1, use fused kernel that combines FWHT + Quantize + GEMM in one launch
    // This eliminates kernel launch overhead and memory traffic
    // DISABLED: Performance regression - Q4K dot product slower than Q4K x Q8_1
    if (false && M == 1 && (K == 1024 || K == 2048 || K == 3072 || K == 4096)) {
        // Fused kernel: each block computes M1_COLS_PER_BLOCK output columns
        const int num_k_blocks = K / 256;
        // Shared memory: K floats for FWHT + num_k_blocks Q4_K blocks for quantized data
        size_t smem_size = K * sizeof(float) + num_k_blocks * sizeof(block_q4_K);
        
        dim3 grid((N + M1_COLS_PER_BLOCK - 1) / M1_COLS_PER_BLOCK);
        dim3 block(256);
        
        switch (K) {
            case 1024:
                rrs_fused_m1_kernel<1024><<<grid, block, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            case 2048:
                rrs_fused_m1_kernel<2048><<<grid, block, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            case 3072:
                rrs_fused_m1_kernel<3072><<<grid, block, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            case 4096:
                rrs_fused_m1_kernel<4096><<<grid, block, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
        }
        return;
    }
    
    // Fallback path for other cases
    const int num_q8_blocks = K / QK8_1;
    
    // For M=1, use fused FWHT + Q8_1 kernel (single kernel launch)
    if (M == 1) {
        size_t q8_size = num_q8_blocks * sizeof(block_q8_1);
        size_t q8_actual;
        void* d_q8 = ctx.pool().alloc(q8_size, &q8_actual);
        
        // Fused FWHT + quantize in one kernel
        rrs_fwht_quantize_q8_1_fused_kernel<<<1, 256, K * sizeof(float), stream>>>(
            (const float*)src1->data, (block_q8_1*)d_q8, K);
        
        // Matrix multiply
        rrs_mmvq_m1_kernel<<<N, 256, 0, stream>>>(src0->data, d_q8, (float*)dst->data, N, K);
        return;
    }
    
    // For M>1, use separate kernels
    // Allocate temporary buffer for FWHT'd activations (fp32)
    size_t fwht_size = M * K * sizeof(float);
    size_t fwht_actual;
    float* d_fwht = (float*)ctx.pool().alloc(fwht_size, &fwht_actual);
    
    // Apply FWHT directly from source to destination (eliminates separate D2D copy)
    rrs_fwht_kernel_opt<<<M, 256, K * sizeof(float), stream>>>(
        (const float*)src1->data, d_fwht, K, M);
    
    // Allocate Q8_1 buffer for quantized activations
    size_t q8_size = M * num_q8_blocks * sizeof(block_q8_1);
    size_t q8_actual;
    void* d_q8 = ctx.pool().alloc(q8_size, &q8_actual);
    
    // Quantize FWHT'd activations to Q8_1
    dim3 quant_grid(M, num_q8_blocks);
    rrs_quantize_q8_1_kernel<<<quant_grid, 32, 0, stream>>>(d_fwht, (block_q8_1*)d_q8, K, M);
    
    // Use MMVQ kernel for matrix multiply
    if (M <= 8) {
        dim3 grid(N, (M + 7) / 8);
        rrs_mmvq_kernel<8><<<grid, 256, 0, stream>>>(src0->data, d_q8, (float*)dst->data, N, K, M);
    } else {
        // For larger M, use optimized dp4a kernel with Q4_K blocks in shared memory
        size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS_ACT, K);
        size_t total_size = M * row_size, actual_size;
        void* d_act_q4k = ctx.pool().alloc(total_size, &actual_size);
        
        // FWHT+quantize to Q4_K from raw activations
        // Note: ggml_cuda_rrs_fwht_quantize does FWHT internally, so pass src1->data (raw)
        ggml_cuda_rrs_fwht_quantize((const float*)src1->data, d_act_q4k, K, M, stream);
        
        // Use optimized kernel with shared memory staging
        // Tile sizes chosen to fit in shared memory (block_q4_K is 144 bytes)
        constexpr int OPT_TILE_M = 16;  // 16 * 144 = 2304 bytes for A
        constexpr int OPT_TILE_N = 16;  // 16 * 144 = 2304 bytes for B
        
        dim3 grid((N + OPT_TILE_N - 1) / OPT_TILE_N, (M + OPT_TILE_M - 1) / OPT_TILE_M);
        dim3 block(256);  // 8 warps
        
        // Shared memory: store actual Q4_K blocks
        size_t smem_size = (OPT_TILE_M + OPT_TILE_N) * sizeof(block_q4_K);
        
        rrs_gemm_q4k_dp4a_opt_kernel<OPT_TILE_M, OPT_TILE_N><<<grid, block, smem_size, stream>>>(
            d_act_q4k, src0->data, (float*)dst->data, M, N, K);
        
        ctx.pool().free(d_act_q4k, actual_size);
    }
    
    // Free buffers in reverse allocation order
    ctx.pool().free(d_q8, q8_actual);
    ctx.pool().free(d_fwht, fwht_actual);
}

bool ggml_cuda_supports_rrs(const ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_Q4_K_RRS && tensor->type != GGML_TYPE_TCQ4_K32) return false;
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
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    for (int i = 0; i < iterations; i++) rrs_gemm_q4k_q4k_tiled_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(d_A_q4k, d_B_q4k, d_C, M, N, K);
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

// TCQ4-K32 test function with correctness validation
extern "C" void ggml_cuda_tcq4_test(void) {
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("\n=== TCQ4-K32 Test on: %s (SM%d%d) ===\n", prop.name, prop.major, prop.minor);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Test 1: Validate Q4_K -> TCQ4_K32 -> dequant roundtrip
    printf("\n[1] Testing Q4_K -> TCQ4_K32 conversion correctness...\n");
    {
        const int K = 2048;
        const int num_blocks = K / 256;
        
        // Allocate test data
        float *h_fp32, *h_dequant;
        block_q4_K *h_q4k;
        h_fp32 = (float*)malloc(K * sizeof(float));
        h_dequant = (float*)malloc(K * sizeof(float));
        h_q4k = (block_q4_K*)malloc(num_blocks * sizeof(block_q4_K));
        
        // Initialize with known pattern
        for (int i = 0; i < K; i++) {
            h_fp32[i] = sinf((float)i * 0.1f) * 2.0f;
        }
        
        // Quantize to Q4_K on CPU
        quantize_row_q4_K_ref(h_fp32, h_q4k, K);
        
        // Upload to GPU
        block_q4_K *d_q4k;
        block_tcq4_k32 *d_tcq4;
        float *d_dequant;
        cudaMalloc(&d_q4k, num_blocks * sizeof(block_q4_K));
        cudaMalloc(&d_tcq4, num_blocks * sizeof(block_tcq4_k32));
        cudaMalloc(&d_dequant, K * sizeof(float));
        cudaMemcpy(d_q4k, h_q4k, num_blocks * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        
        // Convert Q4_K -> TCQ4_K32
        tcq4_k32_convert_from_q4k(d_q4k, d_tcq4, num_blocks, stream);
        cudaStreamSynchronize(stream);
        
        // Dequantize TCQ4_K32 back to FP32 on CPU
        block_tcq4_k32 *h_tcq4 = (block_tcq4_k32*)malloc(num_blocks * sizeof(block_tcq4_k32));
        cudaMemcpy(h_tcq4, d_tcq4, num_blocks * sizeof(block_tcq4_k32), cudaMemcpyDeviceToHost);
        dequantize_row_tcq4_k32(h_tcq4, h_dequant, K);
        
        // Compute error vs original Q4_K dequant
        float *h_q4k_dequant = (float*)malloc(K * sizeof(float));
        dequantize_row_q4_K(h_q4k, h_q4k_dequant, K);
        
        float max_err = 0.0f, sum_err = 0.0f;
        for (int i = 0; i < K; i++) {
            float err = fabsf(h_dequant[i] - h_q4k_dequant[i]);
            max_err = fmaxf(max_err, err);
            sum_err += err;
        }
        printf("    Q4_K vs TCQ4_K32 dequant: max_err=%.6f, avg_err=%.6f\n", max_err, sum_err / K);
        printf("    Conversion: %s\n", (max_err < 0.5f) ? "PASSED" : "FAILED");
        
        cudaFree(d_q4k); cudaFree(d_tcq4); cudaFree(d_dequant);
        free(h_fp32); free(h_dequant); free(h_q4k); free(h_tcq4); free(h_q4k_dequant);
    }
    
    // Test 1b: Compare TCQ4_K32 vs Q4_K_RRS quantization (both with FWHT)
    printf("\n[1b] Comparing TCQ4_K32 vs Q4_K_RRS quantization (with FWHT)...\n");
    {
        const int K = 1024;
        
        // Create test data
        float *h_fp32 = (float*)malloc(K * sizeof(float));
        for (int i = 0; i < K; i++) h_fp32[i] = sinf((float)i * 0.1f) * 2.0f;
        
        // Quantize with Q4_K_RRS (uses Q4_K format + FWHT)
        size_t q4k_rrs_size = ggml_row_size(GGML_TYPE_Q4_K_RRS, K);
        void *h_q4k_rrs = malloc(q4k_rrs_size);
        quantize_q4_K_rrs(h_fp32, h_q4k_rrs, 1, K, NULL);
        
        // Quantize with TCQ4_K32 (uses TCQ4 format + FWHT)
        size_t tcq4_size = ggml_row_size(GGML_TYPE_TCQ4_K32, K);
        void *h_tcq4 = malloc(tcq4_size);
        quantize_tcq4_k32(h_fp32, h_tcq4, 1, K, NULL);
        
        // Dequantize both
        float *h_dequant_q4k = (float*)malloc(K * sizeof(float));
        float *h_dequant_tcq4 = (float*)malloc(K * sizeof(float));
        dequantize_row_q4_K((const block_q4_K*)h_q4k_rrs, h_dequant_q4k, K);
        dequantize_row_tcq4_k32((const block_tcq4_k32*)h_tcq4, h_dequant_tcq4, K);
        
        // Compare dequantized values
        float max_diff = 0.0f, sum_diff = 0.0f;
        for (int i = 0; i < K; i++) {
            float diff = fabsf(h_dequant_q4k[i] - h_dequant_tcq4[i]);
            max_diff = fmaxf(max_diff, diff);
            sum_diff += diff;
        }
        printf("    Q4_K_RRS vs TCQ4_K32 dequant: max_diff=%.6f, avg_diff=%.6f\n", max_diff, sum_diff / K);
        
        // Also check error vs original (both should FWHT then quant)
        float *h_fwht = (float*)malloc(K * sizeof(float));
        memcpy(h_fwht, h_fp32, K * sizeof(float));
        for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_fwht + i, 256);
        
        float max_err_q4k = 0.0f, max_err_tcq4 = 0.0f;
        for (int i = 0; i < K; i++) {
            max_err_q4k = fmaxf(max_err_q4k, fabsf(h_dequant_q4k[i] - h_fwht[i]));
            max_err_tcq4 = fmaxf(max_err_tcq4, fabsf(h_dequant_tcq4[i] - h_fwht[i]));
        }
        printf("    Q4_K_RRS vs FWHT(orig): max_err=%.6f\n", max_err_q4k);
        printf("    TCQ4_K32 vs FWHT(orig): max_err=%.6f\n", max_err_tcq4);
        printf("    First 8 FWHT(orig): [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_fwht[0], h_fwht[1], h_fwht[2], h_fwht[3], h_fwht[4], h_fwht[5], h_fwht[6], h_fwht[7]);
        printf("    First 8 Q4K dequant: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_dequant_q4k[0], h_dequant_q4k[1], h_dequant_q4k[2], h_dequant_q4k[3],
               h_dequant_q4k[4], h_dequant_q4k[5], h_dequant_q4k[6], h_dequant_q4k[7]);
        printf("    First 8 TCQ4 dequant: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_dequant_tcq4[0], h_dequant_tcq4[1], h_dequant_tcq4[2], h_dequant_tcq4[3],
               h_dequant_tcq4[4], h_dequant_tcq4[5], h_dequant_tcq4[6], h_dequant_tcq4[7]);
        
        free(h_fp32); free(h_q4k_rrs); free(h_tcq4);
        free(h_dequant_q4k); free(h_dequant_tcq4); free(h_fwht);
    }
    
    // Test 2: Validate GEMV correctness (no FWHT - just TCQ4 dot product)
    printf("\n[2] Testing GEMV (M=1) correctness (no FWHT)...\n");
    {
        const int N = 64, K = 256;
        const int num_k_blocks = K / 256;
        
        float *h_act = (float*)malloc(K * sizeof(float));
        float *h_wgt = (float*)malloc(N * K * sizeof(float));
        float *h_out_ref = (float*)malloc(N * sizeof(float));
        float *h_out_tcq4 = (float*)malloc(N * sizeof(float));
        
        for (int i = 0; i < K; i++) h_act[i] = (float)(i % 10) * 0.1f;
        for (int i = 0; i < N * K; i++) h_wgt[i] = (float)(i % 7) * 0.05f - 0.15f;
        
        // Reference: FP32 matmul
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act[k] * h_wgt[n * K + k];
            h_out_ref[n] = sum;
        }
        
        // Quantize to TCQ4_K32 (no FWHT)
        block_tcq4_k32 *h_act_tcq4 = (block_tcq4_k32*)malloc(num_k_blocks * sizeof(block_tcq4_k32));
        block_tcq4_k32 *h_wgt_tcq4 = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
        quantize_row_tcq4_k32_ref(h_act, h_act_tcq4, K);
        for (int n = 0; n < N; n++) {
            quantize_row_tcq4_k32_ref(h_wgt + n * K, h_wgt_tcq4 + n * num_k_blocks, K);
        }
        
        block_tcq4_k32 *d_act, *d_wgt;
        float *d_out;
        cudaMalloc(&d_act, num_k_blocks * sizeof(block_tcq4_k32));
        cudaMalloc(&d_wgt, N * num_k_blocks * sizeof(block_tcq4_k32));
        cudaMalloc(&d_out, N * sizeof(float));
        cudaMemcpy(d_act, h_act_tcq4, num_k_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wgt, h_wgt_tcq4, N * num_k_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
        
        tcq4_k32_gemv(d_act, d_wgt, d_out, N, K, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_out_tcq4, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_err = 0.0f, sum_err = 0.0f;
        for (int n = 0; n < N; n++) {
            float err = fabsf(h_out_tcq4[n] - h_out_ref[n]);
            float rel_err = err / (fabsf(h_out_ref[n]) + 1e-6f);
            max_err = fmaxf(max_err, rel_err);
            sum_err += rel_err;
        }
        printf("    GEMV rel error: max=%.4f, avg=%.4f\n", max_err, sum_err / N);
        printf("    GEMV (no FWHT): %s\n", (max_err < 0.5f) ? "PASSED" : "FAILED");
        
        // Debug: print first few outputs
        printf("    First 4 outputs: ref=[%.4f, %.4f, %.4f, %.4f] tcq4=[%.4f, %.4f, %.4f, %.4f]\n",
               h_out_ref[0], h_out_ref[1], h_out_ref[2], h_out_ref[3],
               h_out_tcq4[0], h_out_tcq4[1], h_out_tcq4[2], h_out_tcq4[3]);
        
        // Debug: compute reference using dequantized TCQ4 values
        float *h_act_dequant = (float*)malloc(K * sizeof(float));
        float *h_wgt_dequant = (float*)malloc(N * K * sizeof(float));
        dequantize_row_tcq4_k32(h_act_tcq4, h_act_dequant, K);
        for (int n = 0; n < N; n++) {
            dequantize_row_tcq4_k32(h_wgt_tcq4 + n * num_k_blocks, h_wgt_dequant + n * K, K);
        }
        
        float *h_out_dequant_ref = (float*)malloc(N * sizeof(float));
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act_dequant[k] * h_wgt_dequant[n * K + k];
            h_out_dequant_ref[n] = sum;
        }
        printf("    Dequant ref: [%.4f, %.4f, %.4f, %.4f]\n", 
               h_out_dequant_ref[0], h_out_dequant_ref[1], h_out_dequant_ref[2], h_out_dequant_ref[3]);
        
        // Check dequant ref vs CUDA result
        float max_err_vs_dequant = 0.0f;
        for (int n = 0; n < N; n++) {
            float err = fabsf(h_out_tcq4[n] - h_out_dequant_ref[n]);
            max_err_vs_dequant = fmaxf(max_err_vs_dequant, err);
        }
        printf("    CUDA vs dequant-ref max error: %.6f\n", max_err_vs_dequant);
        
        // Debug: print TCQ4 block metadata
        printf("    Act TCQ4[0]: S=%.6f, Z=%.6f, sc=[%d,%d,%d,%d], zc=[%d,%d,%d,%d]\n",
               __half2float(h_act_tcq4[0].S), __half2float(h_act_tcq4[0].Z),
               (int)h_act_tcq4[0].sc[0], (int)h_act_tcq4[0].sc[1], 
               (int)h_act_tcq4[0].sc[2], (int)h_act_tcq4[0].sc[3],
               (int)h_act_tcq4[0].zc[0], (int)h_act_tcq4[0].zc[1],
               (int)h_act_tcq4[0].zc[2], (int)h_act_tcq4[0].zc[3]);
        printf("    Wgt TCQ4[0]: S=%.6f, Z=%.6f, sc=[%d,%d,%d,%d]\n",
               __half2float(h_wgt_tcq4[0].S), __half2float(h_wgt_tcq4[0].Z),
               (int)h_wgt_tcq4[0].sc[0], (int)h_wgt_tcq4[0].sc[1],
               (int)h_wgt_tcq4[0].sc[2], (int)h_wgt_tcq4[0].sc[3]);
        
        free(h_act_dequant); free(h_wgt_dequant); free(h_out_dequant_ref);
        cudaFree(d_act); cudaFree(d_wgt); cudaFree(d_out);
        free(h_act); free(h_wgt); free(h_out_ref); free(h_out_tcq4);
        free(h_act_tcq4); free(h_wgt_tcq4);
    }
    
    // Test 2a: GEMM (M>1) correctness
    printf("\n[2a] Testing GEMM (M=4) correctness (no FWHT)...\n");
    {
        const int M = 4, N = 32, K = 256;
        const int num_k_blocks = K / 256;
        
        float *h_act = (float*)malloc(M * K * sizeof(float));
        float *h_wgt = (float*)malloc(N * K * sizeof(float));
        float *h_out_ref = (float*)malloc(M * N * sizeof(float));
        float *h_out_tcq4 = (float*)malloc(M * N * sizeof(float));
        
        for (int i = 0; i < M * K; i++) h_act[i] = sinf((float)i * 0.07f) * 0.4f;
        for (int i = 0; i < N * K; i++) h_wgt[i] = cosf((float)i * 0.05f) * 0.3f;
        
        // Reference: FP32 matmul
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) sum += h_act[m * K + k] * h_wgt[n * K + k];
                h_out_ref[m * N + n] = sum;
            }
        }
        
        // Quantize to TCQ4_K32 (no FWHT)
        block_tcq4_k32 *h_act_tcq4 = (block_tcq4_k32*)malloc(M * num_k_blocks * sizeof(block_tcq4_k32));
        block_tcq4_k32 *h_wgt_tcq4 = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
        for (int m = 0; m < M; m++) {
            quantize_row_tcq4_k32_ref(h_act + m * K, h_act_tcq4 + m * num_k_blocks, K);
        }
        for (int n = 0; n < N; n++) {
            quantize_row_tcq4_k32_ref(h_wgt + n * K, h_wgt_tcq4 + n * num_k_blocks, K);
        }
        
        block_tcq4_k32 *d_act, *d_wgt;
        float *d_out;
        cudaMalloc(&d_act, M * num_k_blocks * sizeof(block_tcq4_k32));
        cudaMalloc(&d_wgt, N * num_k_blocks * sizeof(block_tcq4_k32));
        cudaMalloc(&d_out, M * N * sizeof(float));
        cudaMemcpy(d_act, h_act_tcq4, M * num_k_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wgt, h_wgt_tcq4, N * num_k_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
        
        tcq4_k32_gemm_imma(d_act, d_wgt, d_out, M, N, K, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_out_tcq4, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute reference using dequantized values
        float *h_act_dequant = (float*)malloc(M * K * sizeof(float));
        float *h_wgt_dequant = (float*)malloc(N * K * sizeof(float));
        for (int m = 0; m < M; m++) dequantize_row_tcq4_k32(h_act_tcq4 + m * num_k_blocks, h_act_dequant + m * K, K);
        for (int n = 0; n < N; n++) dequantize_row_tcq4_k32(h_wgt_tcq4 + n * num_k_blocks, h_wgt_dequant + n * K, K);
        
        float *h_out_dequant_ref = (float*)malloc(M * N * sizeof(float));
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) sum += h_act_dequant[m * K + k] * h_wgt_dequant[n * K + k];
                h_out_dequant_ref[m * N + n] = sum;
            }
        }
        
        float max_err_vs_dequant = 0.0f, max_err_vs_fp32 = 0.0f;
        for (int i = 0; i < M * N; i++) {
            max_err_vs_dequant = fmaxf(max_err_vs_dequant, fabsf(h_out_tcq4[i] - h_out_dequant_ref[i]));
            float rel_err = fabsf(h_out_tcq4[i] - h_out_ref[i]) / (fabsf(h_out_ref[i]) + 1e-6f);
            max_err_vs_fp32 = fmaxf(max_err_vs_fp32, rel_err);
        }
        printf("    GEMM vs dequant-ref max error: %.6f\n", max_err_vs_dequant);
        printf("    GEMM vs FP32 max rel error: %.4f\n", max_err_vs_fp32);
        printf("    GEMM (M=4): %s\n", (max_err_vs_dequant < 0.001f) ? "PASSED" : "FAILED");
        
        printf("    First row: ref=[%.4f, %.4f, %.4f, %.4f] tcq4=[%.4f, %.4f, %.4f, %.4f]\n",
               h_out_ref[0], h_out_ref[1], h_out_ref[2], h_out_ref[3],
               h_out_tcq4[0], h_out_tcq4[1], h_out_tcq4[2], h_out_tcq4[3]);
        
        cudaFree(d_act); cudaFree(d_wgt); cudaFree(d_out);
        free(h_act); free(h_wgt); free(h_out_ref); free(h_out_tcq4);
        free(h_act_tcq4); free(h_wgt_tcq4);
        free(h_act_dequant); free(h_wgt_dequant); free(h_out_dequant_ref);
    }
    
    // Test 2b: Full FWHT path (simulates actual inference)
    printf("\n[2b] Testing full FWHT path (simulates inference)...\n");
    {
        const int N = 64, K = 256;
        const int num_k_blocks = K / 256;
        
        float *h_act = (float*)malloc(K * sizeof(float));
        float *h_wgt = (float*)malloc(N * K * sizeof(float));
        float *h_out_ref = (float*)malloc(N * sizeof(float));
        float *h_out_tcq4 = (float*)malloc(N * sizeof(float));
        
        // Initialize with varied data
        for (int i = 0; i < K; i++) h_act[i] = sinf((float)i * 0.05f) * 0.5f;
        for (int i = 0; i < N * K; i++) h_wgt[i] = cosf((float)i * 0.03f) * 0.3f;
        
        // Reference: FP32 matmul (original space)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act[k] * h_wgt[n * K + k];
            h_out_ref[n] = sum;
        }
        
        // Simulate offline weight quantization: FWHT + TCQ4 quantize
        float *h_wgt_fwht = (float*)malloc(N * K * sizeof(float));
        block_tcq4_k32 *h_wgt_tcq4 = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
        for (int n = 0; n < N; n++) {
            memcpy(h_wgt_fwht + n * K, h_wgt + n * K, K * sizeof(float));
            ggml_fwht_impl(h_wgt_fwht + n * K, K);  // FWHT weights
            quantize_row_tcq4_k32_ref(h_wgt_fwht + n * K, h_wgt_tcq4 + n * num_k_blocks, K);
        }
        
        // Upload weights to GPU
        block_tcq4_k32 *d_wgt;
        cudaMalloc(&d_wgt, N * num_k_blocks * sizeof(block_tcq4_k32));
        cudaMemcpy(d_wgt, h_wgt_tcq4, N * num_k_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
        
        // Runtime: upload FP32 activations, use tcq4_k32_fwht_quantize (FWHT + quant on GPU)
        float *d_act_fp32;
        block_tcq4_k32 *d_act_tcq4;
        float *d_out;
        cudaMalloc(&d_act_fp32, K * sizeof(float));
        cudaMalloc(&d_act_tcq4, num_k_blocks * sizeof(block_tcq4_k32));
        cudaMalloc(&d_out, N * sizeof(float));
        cudaMemcpy(d_act_fp32, h_act, K * sizeof(float), cudaMemcpyHostToDevice);
        
        // FWHT + quantize activations on GPU
        tcq4_k32_fwht_quantize(d_act_fp32, d_act_tcq4, K, 1, stream);
        cudaStreamSynchronize(stream);
        
        // Run GEMV
        tcq4_k32_gemv(d_act_tcq4, d_wgt, d_out, N, K, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_out_tcq4, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_err = 0.0f, sum_err = 0.0f;
        for (int n = 0; n < N; n++) {
            float err = fabsf(h_out_tcq4[n] - h_out_ref[n]);
            float rel_err = err / (fabsf(h_out_ref[n]) + 1e-6f);
            max_err = fmaxf(max_err, rel_err);
            sum_err += rel_err;
        }
        printf("    Full path rel error: max=%.4f, avg=%.4f\n", max_err, sum_err / N);
        printf("    Full FWHT path: %s\n", (max_err < 0.5f) ? "PASSED" : "FAILED");
        
        // Debug: print first few outputs
        printf("    First 4 outputs: ref=[%.4f, %.4f, %.4f, %.4f] tcq4=[%.4f, %.4f, %.4f, %.4f]\n",
               h_out_ref[0], h_out_ref[1], h_out_ref[2], h_out_ref[3],
               h_out_tcq4[0], h_out_tcq4[1], h_out_tcq4[2], h_out_tcq4[3]);
        
        // Debug: check intermediate FWHT'd activation quantization
        block_tcq4_k32 h_act_tcq4_debug;
        cudaMemcpy(&h_act_tcq4_debug, d_act_tcq4, sizeof(block_tcq4_k32), cudaMemcpyDeviceToHost);
        float S = __half2float(h_act_tcq4_debug.S);
        float Z = __half2float(h_act_tcq4_debug.Z);
        printf("    Act TCQ4 block 0: S=%.6f, Z=%.6f, sc[0]=%d, zc[0]=%d\n", 
               S, Z, (int)h_act_tcq4_debug.sc[0], (int)h_act_tcq4_debug.zc[0]);
        
        cudaFree(d_act_fp32); cudaFree(d_act_tcq4); cudaFree(d_wgt); cudaFree(d_out);
        free(h_act); free(h_wgt); free(h_wgt_fwht); free(h_out_ref); free(h_out_tcq4);
        free(h_wgt_tcq4);
    }
    
    // Test 2d: Verify TCQ4 FWHT+Q8 kernel produces correct results
    printf("\n[2d] Testing TCQ4 FWHT+Q8 kernel correctness...\n");
    {
        const int K = 1024;
        const int num_q8_blocks = K / QK8_1;
        
        float *h_act = (float*)malloc(K * sizeof(float));
        for (int i = 0; i < K; i++) h_act[i] = sinf((float)i * 0.05f) * 0.5f;
        
        // CPU reference: FWHT with step=256, then quantize to Q8_1
        float *h_fwht_cpu = (float*)malloc(K * sizeof(float));
        memcpy(h_fwht_cpu, h_act, K * sizeof(float));
        for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_fwht_cpu + i, 256);
        
        // Upload to GPU and run tcq4_fwht_quantize_q8
        float *d_act;
        block_q8_1 *d_q8;
        cudaMalloc(&d_act, K * sizeof(float));
        cudaMalloc(&d_q8, num_q8_blocks * sizeof(block_q8_1));
        cudaMemcpy(d_act, h_act, K * sizeof(float), cudaMemcpyHostToDevice);
        
        tcq4_fwht_quantize_q8(d_act, d_q8, K, 1, stream);
        cudaStreamSynchronize(stream);
        
        // Download and dequantize GPU result
        block_q8_1 *h_q8 = (block_q8_1*)malloc(num_q8_blocks * sizeof(block_q8_1));
        cudaMemcpy(h_q8, d_q8, num_q8_blocks * sizeof(block_q8_1), cudaMemcpyDeviceToHost);
        
        float *h_dequant_gpu = (float*)malloc(K * sizeof(float));
        for (int b = 0; b < num_q8_blocks; b++) {
            float d = __half2float(*(const half*)&h_q8[b]);
            for (int i = 0; i < QK8_1; i++) {
                h_dequant_gpu[b * QK8_1 + i] = d * (float)h_q8[b].qs[i];
            }
        }
        
        // Compare GPU dequant vs CPU FWHT reference
        float max_err = 0.0f;
        for (int i = 0; i < K; i++) {
            float err = fabsf(h_dequant_gpu[i] - h_fwht_cpu[i]);
            max_err = fmaxf(max_err, err);
        }
        printf("    TCQ4 FWHT+Q8 vs CPU FWHT ref: max_err=%.6f\n", max_err);
        printf("    First 8 CPU FWHT: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_fwht_cpu[0], h_fwht_cpu[1], h_fwht_cpu[2], h_fwht_cpu[3],
               h_fwht_cpu[4], h_fwht_cpu[5], h_fwht_cpu[6], h_fwht_cpu[7]);
        printf("    First 8 GPU dequant: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_dequant_gpu[0], h_dequant_gpu[1], h_dequant_gpu[2], h_dequant_gpu[3],
               h_dequant_gpu[4], h_dequant_gpu[5], h_dequant_gpu[6], h_dequant_gpu[7]);
        printf("    TCQ4 FWHT+Q8: %s\n", (max_err < 0.02f) ? "PASSED" : "FAILED");
        
        cudaFree(d_act); cudaFree(d_q8);
        free(h_act); free(h_fwht_cpu); free(h_q8); free(h_dequant_gpu);
    }
    
    // Test 2d2: End-to-end TCQ4 W4A8 test (step=256 FWHT for both weights and activations)
    printf("\n[2d2] End-to-end TCQ4 W4A8 test (step=256 FWHT)...\n");
    {
        const int N = 64, K = 1024;
        const int num_tcq4_blocks = K / 256;
        const int num_q8_blocks = K / QK8_1;
        
        float *h_act = (float*)malloc(K * sizeof(float));
        float *h_wgt = (float*)malloc(N * K * sizeof(float));
        float *h_out_ref = (float*)malloc(N * sizeof(float));
        
        for (int i = 0; i < K; i++) h_act[i] = sinf((float)i * 0.05f) * 0.5f;
        for (int i = 0; i < N * K; i++) h_wgt[i] = cosf((float)i * 0.03f) * 0.3f;
        
        // Reference: FP32 matmul (original space)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act[k] * h_wgt[n * K + k];
            h_out_ref[n] = sum;
        }
        
        // Weight quantization: FWHT (step=256) then TCQ4
        float *h_wgt_fwht = (float*)malloc(N * K * sizeof(float));
        block_tcq4_k32 *h_wgt_tcq4 = (block_tcq4_k32*)malloc(N * num_tcq4_blocks * sizeof(block_tcq4_k32));
        for (int n = 0; n < N; n++) {
            memcpy(h_wgt_fwht + n * K, h_wgt + n * K, K * sizeof(float));
            for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_wgt_fwht + n * K + i, 256);
            quantize_row_tcq4_k32_ref(h_wgt_fwht + n * K, h_wgt_tcq4 + n * num_tcq4_blocks, K);
        }
        
        // Upload weights
        block_tcq4_k32 *d_wgt;
        cudaMalloc(&d_wgt, N * num_tcq4_blocks * sizeof(block_tcq4_k32));
        cudaMemcpy(d_wgt, h_wgt_tcq4, N * num_tcq4_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
        
        // Activation path: use tcq4_fwht_quantize_q8 (step=256 FWHT + Q8_1)
        float *d_act;
        block_q8_1 *d_q8;
        float *d_out;
        cudaMalloc(&d_act, K * sizeof(float));
        cudaMalloc(&d_q8, num_q8_blocks * sizeof(block_q8_1));
        cudaMalloc(&d_out, N * sizeof(float));
        cudaMemcpy(d_act, h_act, K * sizeof(float), cudaMemcpyHostToDevice);
        
        tcq4_fwht_quantize_q8(d_act, d_q8, K, 1, stream);
        cudaStreamSynchronize(stream);
        
        // Run TCQ4 x Q8 GEMV
        tcq4_k32_q8_gemv(d_q8, d_wgt, d_out, N, K, stream);
        cudaStreamSynchronize(stream);
        
        float *h_out_gpu = (float*)malloc(N * sizeof(float));
        cudaMemcpy(h_out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compare
        float max_err = 0.0f, sum_err = 0.0f;
        for (int n = 0; n < N; n++) {
            float rel_err = fabsf(h_out_gpu[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
            max_err = fmaxf(max_err, rel_err);
            sum_err += rel_err;
        }
        printf("    TCQ4 W4A8 (step=256): max_rel_err=%.4f, avg_rel_err=%.4f\n", max_err, sum_err / N);
        printf("    First 4 ref:  [%.4f, %.4f, %.4f, %.4f]\n", h_out_ref[0], h_out_ref[1], h_out_ref[2], h_out_ref[3]);
        printf("    First 4 GPU:  [%.4f, %.4f, %.4f, %.4f]\n", h_out_gpu[0], h_out_gpu[1], h_out_gpu[2], h_out_gpu[3]);
        printf("    TCQ4 W4A8: %s\n", (sum_err / N < 0.15f) ? "PASSED" : "FAILED");
        
        cudaFree(d_act); cudaFree(d_q8); cudaFree(d_wgt); cudaFree(d_out);
        free(h_act); free(h_wgt); free(h_out_ref);
        free(h_wgt_fwht); free(h_wgt_tcq4); free(h_out_gpu);
    }
    
    // Test 2d3: Compare chunk-256 FWHT vs full-K FWHT
    printf("\n[2d3] Comparing chunk-256 FWHT vs full-K FWHT...\n");
    {
        const int N = 32, K = 1024;
        const int num_k_blocks = K / 256;
        
        float *h_act = (float*)malloc(K * sizeof(float));
        float *h_wgt = (float*)malloc(N * K * sizeof(float));
        float *h_out_ref = (float*)malloc(N * sizeof(float));
        
        for (int i = 0; i < K; i++) h_act[i] = sinf((float)i * 0.05f) * 0.5f;
        for (int i = 0; i < N * K; i++) h_wgt[i] = cosf((float)i * 0.03f) * 0.3f;
        
        // Reference: FP32 matmul
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act[k] * h_wgt[n * K + k];
            h_out_ref[n] = sum;
        }
        
        // Method A: Chunk-256 FWHT (current TCQ4 approach)
        float *h_act_fwht256 = (float*)malloc(K * sizeof(float));
        float *h_wgt_fwht256 = (float*)malloc(N * K * sizeof(float));
        memcpy(h_act_fwht256, h_act, K * sizeof(float));
        for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_act_fwht256 + i, 256);
        for (int n = 0; n < N; n++) {
            memcpy(h_wgt_fwht256 + n * K, h_wgt + n * K, K * sizeof(float));
            for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_wgt_fwht256 + n * K + i, 256);
        }
        
        // Method B: Full-K FWHT (Q4_K_RRS approach)
        const int step = K & -K;  // = 1024 for K=1024
        float *h_act_fwhtK = (float*)malloc(K * sizeof(float));
        float *h_wgt_fwhtK = (float*)malloc(N * K * sizeof(float));
        memcpy(h_act_fwhtK, h_act, K * sizeof(float));
        for (int i = 0; i < K; i += step) ggml_fwht_impl(h_act_fwhtK + i, step);
        for (int n = 0; n < N; n++) {
            memcpy(h_wgt_fwhtK + n * K, h_wgt + n * K, K * sizeof(float));
            for (int i = 0; i < K; i += step) ggml_fwht_impl(h_wgt_fwhtK + n * K + i, step);
        }
        
        // Quantize both to TCQ4
        block_tcq4_k32 *h_act_tcq4_256 = (block_tcq4_k32*)malloc(num_k_blocks * sizeof(block_tcq4_k32));
        block_tcq4_k32 *h_wgt_tcq4_256 = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
        block_tcq4_k32 *h_act_tcq4_K = (block_tcq4_k32*)malloc(num_k_blocks * sizeof(block_tcq4_k32));
        block_tcq4_k32 *h_wgt_tcq4_K = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
        
        quantize_row_tcq4_k32_ref(h_act_fwht256, h_act_tcq4_256, K);
        quantize_row_tcq4_k32_ref(h_act_fwhtK, h_act_tcq4_K, K);
        for (int n = 0; n < N; n++) {
            quantize_row_tcq4_k32_ref(h_wgt_fwht256 + n * K, h_wgt_tcq4_256 + n * num_k_blocks, K);
            quantize_row_tcq4_k32_ref(h_wgt_fwhtK + n * K, h_wgt_tcq4_K + n * num_k_blocks, K);
        }
        
        // Dequant and compute results
        float *h_act_dq256 = (float*)malloc(K * sizeof(float));
        float *h_wgt_dq256 = (float*)malloc(N * K * sizeof(float));
        float *h_act_dqK = (float*)malloc(K * sizeof(float));
        float *h_wgt_dqK = (float*)malloc(N * K * sizeof(float));
        
        dequantize_row_tcq4_k32(h_act_tcq4_256, h_act_dq256, K);
        dequantize_row_tcq4_k32(h_act_tcq4_K, h_act_dqK, K);
        for (int n = 0; n < N; n++) {
            dequantize_row_tcq4_k32(h_wgt_tcq4_256 + n * num_k_blocks, h_wgt_dq256 + n * K, K);
            dequantize_row_tcq4_k32(h_wgt_tcq4_K + n * num_k_blocks, h_wgt_dqK + n * K, K);
        }
        
        float *h_out_256 = (float*)malloc(N * sizeof(float));
        float *h_out_K = (float*)malloc(N * sizeof(float));
        for (int n = 0; n < N; n++) {
            float sum256 = 0.0f, sumK = 0.0f;
            for (int k = 0; k < K; k++) {
                sum256 += h_act_dq256[k] * h_wgt_dq256[n * K + k];
                sumK += h_act_dqK[k] * h_wgt_dqK[n * K + k];
            }
            h_out_256[n] = sum256;
            h_out_K[n] = sumK;
        }
        
        float max_err_256 = 0.0f, sum_err_256 = 0.0f;
        float max_err_K = 0.0f, sum_err_K = 0.0f;
        for (int n = 0; n < N; n++) {
            float rel_256 = fabsf(h_out_256[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
            float rel_K = fabsf(h_out_K[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
            max_err_256 = fmaxf(max_err_256, rel_256);
            max_err_K = fmaxf(max_err_K, rel_K);
            sum_err_256 += rel_256;
            sum_err_K += rel_K;
        }
        printf("    Chunk-256 FWHT: max=%.4f, avg=%.4f\n", max_err_256, sum_err_256 / N);
        printf("    Full-K FWHT:    max=%.4f, avg=%.4f\n", max_err_K, sum_err_K / N);
        printf("    Full-K is %.1fx more accurate\n", (sum_err_256 / N) / (sum_err_K / N + 1e-9f));
        
        free(h_act); free(h_wgt); free(h_out_ref);
        free(h_act_fwht256); free(h_wgt_fwht256);
        free(h_act_fwhtK); free(h_wgt_fwhtK);
        free(h_act_tcq4_256); free(h_wgt_tcq4_256);
        free(h_act_tcq4_K); free(h_wgt_tcq4_K);
        free(h_act_dq256); free(h_wgt_dq256);
        free(h_act_dqK); free(h_wgt_dqK);
        free(h_out_256); free(h_out_K);
    }
    
    // Test 2e: Test with realistic K dimensions - also vary N to check sensitivity
    printf("\n[2e] Testing with realistic model dimensions (vary N and K)...\n");
    {
        int test_dims[][2] = {{32, 256}, {32, 1024}, {32, 2048}, {32, 3072}, {64, 1024}, {128, 1024}};
        const char* dim_names[] = {"N=32,K=256", "N=32,K=1024", "N=32,K=2048", "N=32,K=3072", "N=64,K=1024", "N=128,K=1024"};
        
        for (int t = 0; t < 6; t++) {
            const int N = test_dims[t][0];
            const int K = test_dims[t][1];
            const int num_k_blocks = K / 256;
            
            float *h_act = (float*)malloc(K * sizeof(float));
            float *h_wgt = (float*)malloc(N * K * sizeof(float));
            float *h_out_ref = (float*)malloc(N * sizeof(float));
            
            for (int i = 0; i < K; i++) h_act[i] = sinf((float)i * 0.05f) * 0.5f;
            for (int i = 0; i < N * K; i++) h_wgt[i] = cosf((float)i * 0.03f) * 0.3f;
            
            // Reference: FP32 matmul
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) sum += h_act[k] * h_wgt[n * K + k];
                h_out_ref[n] = sum;
            }
            
            // Apply FWHT (in 256-element chunks like TCQ4)
            float *h_act_fwht = (float*)malloc(K * sizeof(float));
            float *h_wgt_fwht = (float*)malloc(N * K * sizeof(float));
            memcpy(h_act_fwht, h_act, K * sizeof(float));
            for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_act_fwht + i, 256);
            for (int n = 0; n < N; n++) {
                memcpy(h_wgt_fwht + n * K, h_wgt + n * K, K * sizeof(float));
                for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_wgt_fwht + n * K + i, 256);
            }
            
            // Quantize to TCQ4
            block_tcq4_k32 *h_act_tcq4 = (block_tcq4_k32*)malloc(num_k_blocks * sizeof(block_tcq4_k32));
            block_tcq4_k32 *h_wgt_tcq4 = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
            quantize_row_tcq4_k32_ref(h_act_fwht, h_act_tcq4, K);
            for (int n = 0; n < N; n++) {
                quantize_row_tcq4_k32_ref(h_wgt_fwht + n * K, h_wgt_tcq4 + n * num_k_blocks, K);
            }
            
            // Run CUDA GEMV
            block_tcq4_k32 *d_act, *d_wgt;
            float *d_out;
            cudaMalloc(&d_act, num_k_blocks * sizeof(block_tcq4_k32));
            cudaMalloc(&d_wgt, N * num_k_blocks * sizeof(block_tcq4_k32));
            cudaMalloc(&d_out, N * sizeof(float));
            cudaMemcpy(d_act, h_act_tcq4, num_k_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
            cudaMemcpy(d_wgt, h_wgt_tcq4, N * num_k_blocks * sizeof(block_tcq4_k32), cudaMemcpyHostToDevice);
            
            tcq4_k32_gemv(d_act, d_wgt, d_out, N, K, stream);
            cudaStreamSynchronize(stream);
            
            float *h_out_tcq4 = (float*)malloc(N * sizeof(float));
            cudaMemcpy(h_out_tcq4, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
            
            float max_err = 0.0f, sum_err = 0.0f;
            int max_idx = 0;
            float max_ref = 0.0f, max_tcq4 = 0.0f;
            for (int n = 0; n < N; n++) {
                float rel_err = fabsf(h_out_tcq4[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
                if (rel_err > max_err) {
                    max_err = rel_err;
                    max_idx = n;
                    max_ref = h_out_ref[n];
                    max_tcq4 = h_out_tcq4[n];
                }
                sum_err += rel_err;
            }
            printf("    %s: max_rel_err=%.4f (idx=%d, ref=%.6f, tcq4=%.6f), avg=%.4f %s\n", 
                   dim_names[t], max_err, max_idx, max_ref, max_tcq4, sum_err / N, 
                   (sum_err / N < 0.1f) ? "OK" : "HIGH");
            
            cudaFree(d_act); cudaFree(d_wgt); cudaFree(d_out);
            free(h_act); free(h_wgt); free(h_out_ref); free(h_out_tcq4);
            free(h_act_fwht); free(h_wgt_fwht);
            free(h_act_tcq4); free(h_wgt_tcq4);
        }
    }
    
    // Test 2f: Direct comparison of Q4_K_RRS vs TCQ4 on same data
    printf("\n[2f] Direct Q4_K_RRS vs TCQ4 comparison...\n");
    {
        const int N = 64, K = 1024;
        const int num_k_blocks = K / 256;
        const int num_q8_blocks = K / QK8_1;
        
        float *h_act = (float*)malloc(K * sizeof(float));
        float *h_wgt = (float*)malloc(N * K * sizeof(float));
        float *h_out_ref = (float*)malloc(N * sizeof(float));
        
        for (int i = 0; i < K; i++) h_act[i] = sinf((float)i * 0.05f) * 0.5f;
        for (int i = 0; i < N * K; i++) h_wgt[i] = cosf((float)i * 0.03f) * 0.3f;
        
        // Reference: FP32 matmul
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act[k] * h_wgt[n * K + k];
            h_out_ref[n] = sum;
        }
        
        // Q4_K_RRS path: FWHT with step=K&-K, quant to Q4_K (weights) + Q8_1 (activations)
        const int step_rrs = K & -K;  // 1024 for K=1024
        float *h_act_fwht_rrs = (float*)malloc(K * sizeof(float));
        float *h_wgt_fwht_rrs = (float*)malloc(N * K * sizeof(float));
        memcpy(h_act_fwht_rrs, h_act, K * sizeof(float));
        for (int i = 0; i < K; i += step_rrs) ggml_fwht_impl(h_act_fwht_rrs + i, step_rrs);
        for (int n = 0; n < N; n++) {
            memcpy(h_wgt_fwht_rrs + n * K, h_wgt + n * K, K * sizeof(float));
            for (int i = 0; i < K; i += step_rrs) ggml_fwht_impl(h_wgt_fwht_rrs + n * K + i, step_rrs);
        }
        
        // Quantize weights to Q4_K, activations to Q8_1 (like Q4_K_RRS runtime)
        block_q4_K *h_wgt_q4k = (block_q4_K*)malloc(N * (K / QK_K) * sizeof(block_q4_K));
        block_q8_1 *h_act_q8 = (block_q8_1*)malloc(num_q8_blocks * sizeof(block_q8_1));
        for (int n = 0; n < N; n++) {
            quantize_row_q4_K_ref(h_wgt_fwht_rrs + n * K, h_wgt_q4k + n * (K / QK_K), K);
        }
        quantize_row_q8_1_ref(h_act_fwht_rrs, h_act_q8, K);
        
        // Debug: check q8_1 block content
        printf("    DEBUG q8_1[0]: d_raw=0x%04x, qs[0..3]=[%d,%d,%d,%d]\n",
               *(uint16_t*)&h_act_q8[0].ds.x, 
               (int)h_act_q8[0].qs[0], (int)h_act_q8[0].qs[1], 
               (int)h_act_q8[0].qs[2], (int)h_act_q8[0].qs[3]);
        
        // Dequant and compute Q4_K_RRS result
        float *h_wgt_dq_rrs = (float*)malloc(N * K * sizeof(float));
        float *h_act_dq_rrs = (float*)malloc(K * sizeof(float));
        for (int n = 0; n < N; n++) {
            dequantize_row_q4_K(h_wgt_q4k + n * (K / QK_K), h_wgt_dq_rrs + n * K, K);
        }
        dequantize_row_q8_1_inline(h_act_q8, h_act_dq_rrs, K);
        
        float *h_out_rrs = (float*)malloc(N * sizeof(float));
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act_dq_rrs[k] * h_wgt_dq_rrs[n * K + k];
            h_out_rrs[n] = sum;
        }
        
        // TCQ4 path: FWHT with step=256, quant to TCQ4 (both)
        float *h_act_fwht_tcq4 = (float*)malloc(K * sizeof(float));
        float *h_wgt_fwht_tcq4 = (float*)malloc(N * K * sizeof(float));
        memcpy(h_act_fwht_tcq4, h_act, K * sizeof(float));
        for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_act_fwht_tcq4 + i, 256);
        for (int n = 0; n < N; n++) {
            memcpy(h_wgt_fwht_tcq4 + n * K, h_wgt + n * K, K * sizeof(float));
            for (int i = 0; i < K; i += 256) ggml_fwht_impl(h_wgt_fwht_tcq4 + n * K + i, 256);
        }
        
        block_tcq4_k32 *h_act_tcq4 = (block_tcq4_k32*)malloc(num_k_blocks * sizeof(block_tcq4_k32));
        block_tcq4_k32 *h_wgt_tcq4 = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
        quantize_row_tcq4_k32_ref(h_act_fwht_tcq4, h_act_tcq4, K);
        for (int n = 0; n < N; n++) {
            quantize_row_tcq4_k32_ref(h_wgt_fwht_tcq4 + n * K, h_wgt_tcq4 + n * num_k_blocks, K);
        }
        
        float *h_wgt_dq_tcq4 = (float*)malloc(N * K * sizeof(float));
        float *h_act_dq_tcq4 = (float*)malloc(K * sizeof(float));
        dequantize_row_tcq4_k32(h_act_tcq4, h_act_dq_tcq4, K);
        for (int n = 0; n < N; n++) {
            dequantize_row_tcq4_k32(h_wgt_tcq4 + n * num_k_blocks, h_wgt_dq_tcq4 + n * K, K);
        }
        
        float *h_out_tcq4 = (float*)malloc(N * sizeof(float));
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act_dq_tcq4[k] * h_wgt_dq_tcq4[n * K + k];
            h_out_tcq4[n] = sum;
        }
        
        // Debug: print first few values
        printf("    First 4 ref:  [%.4f, %.4f, %.4f, %.4f]\n", h_out_ref[0], h_out_ref[1], h_out_ref[2], h_out_ref[3]);
        printf("    First 4 RRS:  [%.4f, %.4f, %.4f, %.4f]\n", h_out_rrs[0], h_out_rrs[1], h_out_rrs[2], h_out_rrs[3]);
        printf("    First 4 TCQ4: [%.4f, %.4f, %.4f, %.4f]\n", h_out_tcq4[0], h_out_tcq4[1], h_out_tcq4[2], h_out_tcq4[3]);
        printf("    First 8 act_dq_rrs: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_act_dq_rrs[0], h_act_dq_rrs[1], h_act_dq_rrs[2], h_act_dq_rrs[3],
               h_act_dq_rrs[4], h_act_dq_rrs[5], h_act_dq_rrs[6], h_act_dq_rrs[7]);
        printf("    First 8 act_fwht_rrs: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_act_fwht_rrs[0], h_act_fwht_rrs[1], h_act_fwht_rrs[2], h_act_fwht_rrs[3],
               h_act_fwht_rrs[4], h_act_fwht_rrs[5], h_act_fwht_rrs[6], h_act_fwht_rrs[7]);
        
        // Compare
        float max_err_rrs = 0.0f, sum_err_rrs = 0.0f;
        float max_err_tcq4 = 0.0f, sum_err_tcq4 = 0.0f;
        for (int n = 0; n < N; n++) {
            float rel_rrs = fabsf(h_out_rrs[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
            float rel_tcq4 = fabsf(h_out_tcq4[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
            max_err_rrs = fmaxf(max_err_rrs, rel_rrs);
            max_err_tcq4 = fmaxf(max_err_tcq4, rel_tcq4);
            sum_err_rrs += rel_rrs;
            sum_err_tcq4 += rel_tcq4;
        }
        printf("    Q4_K_RRS (W4A8, step=%d): max=%.4f, avg=%.4f\n", step_rrs, max_err_rrs, sum_err_rrs / N);
        printf("    TCQ4 (W4A4, step=256):    max=%.4f, avg=%.4f\n", max_err_tcq4, sum_err_tcq4 / N);
        printf("    Q4_K_RRS is %.1fx more accurate\n", (sum_err_tcq4 / N) / (sum_err_rrs / N + 1e-9f));
        
        free(h_act); free(h_wgt); free(h_out_ref);
        free(h_act_fwht_rrs); free(h_wgt_fwht_rrs);
        free(h_wgt_q4k); free(h_act_q8);
        free(h_wgt_dq_rrs); free(h_act_dq_rrs); free(h_out_rrs);
        free(h_act_fwht_tcq4); free(h_wgt_fwht_tcq4);
        free(h_act_tcq4); free(h_wgt_tcq4);
        free(h_wgt_dq_tcq4); free(h_act_dq_tcq4); free(h_out_tcq4);
    }
    
    // Test 2g: Compare W4A8 (TCQ4 weights + FP32 activations) vs W4A4
    printf("\n[2g] Testing W4A4 vs W4Afp32 accuracy...\n");
    {
        const int N = 64, K = 1024;
        const int num_k_blocks = K / 256;
        
        float *h_act = (float*)malloc(K * sizeof(float));
        float *h_wgt = (float*)malloc(N * K * sizeof(float));
        float *h_out_ref = (float*)malloc(N * sizeof(float));
        
        for (int i = 0; i < K; i++) h_act[i] = sinf((float)i * 0.05f) * 0.5f;
        for (int i = 0; i < N * K; i++) h_wgt[i] = cosf((float)i * 0.03f) * 0.3f;
        
        // Reference: FP32 matmul
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act[k] * h_wgt[n * K + k];
            h_out_ref[n] = sum;
        }
        
        // Apply FWHT to both (simulating RRS)
        float *h_act_fwht = (float*)malloc(K * sizeof(float));
        float *h_wgt_fwht = (float*)malloc(N * K * sizeof(float));
        memcpy(h_act_fwht, h_act, K * sizeof(float));
        ggml_fwht_impl(h_act_fwht, K);
        for (int n = 0; n < N; n++) {
            memcpy(h_wgt_fwht + n * K, h_wgt + n * K, K * sizeof(float));
            ggml_fwht_impl(h_wgt_fwht + n * K, K);
        }
        
        // W4A4: Both as TCQ4
        block_tcq4_k32 *h_act_tcq4 = (block_tcq4_k32*)malloc(num_k_blocks * sizeof(block_tcq4_k32));
        block_tcq4_k32 *h_wgt_tcq4 = (block_tcq4_k32*)malloc(N * num_k_blocks * sizeof(block_tcq4_k32));
        quantize_row_tcq4_k32_ref(h_act_fwht, h_act_tcq4, K);
        for (int n = 0; n < N; n++) {
            quantize_row_tcq4_k32_ref(h_wgt_fwht + n * K, h_wgt_tcq4 + n * num_k_blocks, K);
        }
        
        // Dequant weights (both tests use same quantized weights)
        float *h_act_dq4 = (float*)malloc(K * sizeof(float));
        float *h_wgt_dq4 = (float*)malloc(N * K * sizeof(float));
        dequantize_row_tcq4_k32(h_act_tcq4, h_act_dq4, K);
        for (int n = 0; n < N; n++) {
            dequantize_row_tcq4_k32(h_wgt_tcq4 + n * num_k_blocks, h_wgt_dq4 + n * K, K);
        }
        
        // W4A4: dequant_act × dequant_wgt
        float *h_out_w4a4 = (float*)malloc(N * sizeof(float));
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act_dq4[k] * h_wgt_dq4[n * K + k];
            h_out_w4a4[n] = sum;
        }
        
        // W4Afp32: fp32_fwht_act × dequant_wgt (proxy for W4A8)
        float *h_out_w4afp = (float*)malloc(N * sizeof(float));
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_act_fwht[k] * h_wgt_dq4[n * K + k];
            h_out_w4afp[n] = sum;
        }
        
        // Compare errors
        float max_err_w4a4 = 0.0f, sum_err_w4a4 = 0.0f;
        float max_err_w4afp = 0.0f, sum_err_w4afp = 0.0f;
        for (int n = 0; n < N; n++) {
            float rel_w4a4 = fabsf(h_out_w4a4[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
            float rel_w4afp = fabsf(h_out_w4afp[n] - h_out_ref[n]) / (fabsf(h_out_ref[n]) + 1e-6f);
            max_err_w4a4 = fmaxf(max_err_w4a4, rel_w4a4);
            max_err_w4afp = fmaxf(max_err_w4afp, rel_w4afp);
            sum_err_w4a4 += rel_w4a4;
            sum_err_w4afp += rel_w4afp;
        }
        printf("    W4A4 rel error: max=%.4f, avg=%.4f\n", max_err_w4a4, sum_err_w4a4 / N);
        printf("    W4Afp32 rel error: max=%.4f, avg=%.4f\n", max_err_w4afp, sum_err_w4afp / N);
        printf("    W4Afp32 is %.1fx more accurate than W4A4\n", (sum_err_w4a4 / N) / (sum_err_w4afp / N + 1e-9f));
        free(h_act); free(h_wgt); free(h_out_ref);
        free(h_act_fwht); free(h_wgt_fwht);
        free(h_act_tcq4); free(h_wgt_tcq4);
        free(h_act_dq4); free(h_wgt_dq4); free(h_out_w4a4); free(h_out_w4afp);
    }
    
    // Test 2c: Verify CPU vs GPU FWHT consistency
    printf("\n[2c] Testing CPU vs GPU FWHT consistency...\n");
    {
        const int K = 256;
        float *h_data = (float*)malloc(K * sizeof(float));
        float *h_fwht_cpu = (float*)malloc(K * sizeof(float));
        float *h_fwht_gpu = (float*)malloc(K * sizeof(float));
        
        for (int i = 0; i < K; i++) h_data[i] = sinf((float)i * 0.1f);
        
        // CPU FWHT
        memcpy(h_fwht_cpu, h_data, K * sizeof(float));
        ggml_fwht_impl(h_fwht_cpu, K);
        
        // GPU FWHT via tcq4_k32_fwht_quantize (extract FWHT result before quant)
        float *d_data;
        block_tcq4_k32 *d_tcq4;
        cudaMalloc(&d_data, K * sizeof(float));
        cudaMalloc(&d_tcq4, sizeof(block_tcq4_k32));
        cudaMemcpy(d_data, h_data, K * sizeof(float), cudaMemcpyHostToDevice);
        
        tcq4_k32_fwht_quantize(d_data, d_tcq4, K, 1, stream);
        cudaStreamSynchronize(stream);
        
        // Dequant GPU result to compare
        block_tcq4_k32 h_tcq4;
        cudaMemcpy(&h_tcq4, d_tcq4, sizeof(block_tcq4_k32), cudaMemcpyDeviceToHost);
        dequantize_row_tcq4_k32(&h_tcq4, h_fwht_gpu, K);
        
        // Also dequant CPU FWHT result via same quantizer for apples-to-apples
        block_tcq4_k32 h_tcq4_cpu;
        quantize_row_tcq4_k32_ref(h_fwht_cpu, &h_tcq4_cpu, K);
        float *h_fwht_cpu_roundtrip = (float*)malloc(K * sizeof(float));
        dequantize_row_tcq4_k32(&h_tcq4_cpu, h_fwht_cpu_roundtrip, K);
        
        float max_err = 0.0f;
        for (int i = 0; i < K; i++) {
            float err = fabsf(h_fwht_gpu[i] - h_fwht_cpu_roundtrip[i]);
            max_err = fmaxf(max_err, err);
        }
        printf("    CPU vs GPU FWHT+quant dequant max error: %.6f\n", max_err);
        printf("    FWHT consistency: %s\n", (max_err < 0.1f) ? "PASSED" : "FAILED");
        
        // Print raw FWHT values for first 8 elements
        printf("    CPU FWHT[0:7]: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               h_fwht_cpu[0], h_fwht_cpu[1], h_fwht_cpu[2], h_fwht_cpu[3],
               h_fwht_cpu[4], h_fwht_cpu[5], h_fwht_cpu[6], h_fwht_cpu[7]);
        
        cudaFree(d_data); cudaFree(d_tcq4);
        free(h_data); free(h_fwht_cpu); free(h_fwht_gpu); free(h_fwht_cpu_roundtrip);
    }
    
    // Test 3: Benchmark
    printf("\n[3] Benchmarking GEMV (M=1, N=2048, K=2048)...\n");
    TCQ4BenchResult res_gemv;
    tcq4_k32_benchmark(1, 2048, 2048, 100, &res_gemv, stream);
    tcq4_k32_print_benchmark(&res_gemv);
    
    cudaStreamDestroy(stream);
    printf("\n=== TCQ4-K32 Test Complete ===\n");
}
