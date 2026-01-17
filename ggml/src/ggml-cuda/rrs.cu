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
// GEMM Configuration
// ============================================================================ 
// Tile sizes for the optimized GEMM kernel
#define TILE_M 32       // Output tile rows per block
#define TILE_N 32       // Output tile cols per block  
#define TILE_K 8        // K-blocks (256 elements each = 2048 elements) per iteration
#define THREADS_PER_BLOCK 256
#define M1_COLS_PER_BLOCK 4  // Number of output columns per block for M=1 fused kernel

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

void ggml_cuda_rrs_mul_mat(ggml_backend_cuda_context& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    const int M = src1->ne[1], N = src0->ne[1], K = src0->ne[0];
    cudaStream_t stream = ctx.stream();
    
    // Select kernel based on M dimension
    if (M == 1) {
        // Use fused kernel for M=1 to minimize launch overhead
        // Shared memory: K floats for FWHT + (K/256) Q4_K blocks for quantized act
        const size_t smem_size = K * sizeof(float) + (K / 256) * sizeof(block_q4_K) + 64 * sizeof(float);
        const int num_blocks = (N + M1_COLS_PER_BLOCK - 1) / M1_COLS_PER_BLOCK;
        
        // Select template based on K dimension (common sizes)
        switch (K) {
            case 512:
                rrs_fused_m1_kernel<512><<<num_blocks, 256, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            case 1024:
                rrs_fused_m1_kernel<1024><<<num_blocks, 256, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            case 2048:
                rrs_fused_m1_kernel<2048><<<num_blocks, 256, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            case 2816:
                rrs_fused_m1_kernel<2816><<<num_blocks, 256, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            case 4096:
                rrs_fused_m1_kernel<4096><<<num_blocks, 256, smem_size, stream>>>(
                    (const float*)src1->data, src0->data, (float*)dst->data, N);
                break;
            default: {
                // Fallback: separate FWHT+quantize then GEMM
                size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS_ACT, K);
                size_t actual_size;
                void* d_act_q4k = ctx.pool().alloc(row_size, &actual_size);
                ggml_cuda_rrs_fwht_quantize((const float*)src1->data, d_act_q4k, K, 1, stream);
                rrs_gemm_q4k_q4k_m1_kernel<<<N, 256, 0, stream>>>(
                    d_act_q4k, src0->data, (float*)dst->data, N, K);
                ctx.pool().free(d_act_q4k, actual_size);
                break;
            }
        }
    } else {
        // M > 1: allocate and quantize activations
        size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS_ACT, K);
        size_t total_size = M * row_size, actual_size;
        void* d_act_q4k = ctx.pool().alloc(total_size, &actual_size);
        ggml_cuda_rrs_fwht_quantize((const float*)src1->data, d_act_q4k, K, M, stream);
        
        if (M <= 32) {
            // Small batch: one block per column, warps handle different rows
            rrs_gemm_q4k_q4k_batched_kernel<<<N, 256, 0, stream>>>(
                d_act_q4k, src0->data, (float*)dst->data, M, N, K);
        } else {
            // Large batch: tiled kernel
            dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
            rrs_gemm_q4k_q4k_tiled_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                d_act_q4k, src0->data, (float*)dst->data, M, N, K);
        }
        
        ctx.pool().free(d_act_q4k, actual_size);
    }
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
