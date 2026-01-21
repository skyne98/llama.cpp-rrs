/**
 * TCQ4 RRS W4A4 CUDA Implementation - Clean Rewrite
 *
 * Based on validated, optimized kernels from rrs_validation/
 * - Activation pipeline: rrs_fused_v2 (200-600x speedup over unfused)
 * - GEMM kernel: tcq4_gemm_opt_v12 (75 TFLOPS, 3x faster than Q4_K dp4a)
 *
 * Key algorithms (from reference.py):
 * - Activation quantization: scale = max(|x|), q = round(x * 7 / scale)
 * - Dequantization: x_approx = q * (scale / 7)
 * - GEMM: C += int_dot * (a_scale/7) * b_scale + sum_a * (a_scale/7) * b_zero
 *
 * Build: requires SM75+ (Turing/Ampere/Ada) for mma.sync.m16n8k32.s4.s4
 */

#include "tcq4_k32.cuh"
#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdio>

// =============================================================================
// Constants
// =============================================================================

// Use tile format constants from ggml-common.h via tcq4_k32.cuh
#define TCQ4_BLOCK_SIZE TCQ4_TILE_K
#define TCQ4_NUM_GROUPS TCQ4_TILE_GROUPS
#define TCQ4_GROUP_SIZE TCQ4_TILE_GROUP_SIZE
#define WARP_SIZE 32

// IMMA dimensions for mma.sync.m16n8k32
#define IMMA_M 16
#define IMMA_N 8
#define IMMA_K 32

// GEMM v12 tiling parameters (validated optimal for RTX 3090)
#define V12_WARPS_M 2
#define V12_WARPS_N 4
#define V12_TILES_PER_WARP_M 1
#define V12_TILES_PER_WARP_N 1
#define V12_BLOCK_M (V12_WARPS_M * V12_TILES_PER_WARP_M * IMMA_M)  // 32
#define V12_BLOCK_N (V12_WARPS_N * V12_TILES_PER_WARP_N * IMMA_N)  // 32
#define V12_THREADS (V12_WARPS_M * V12_WARPS_N * WARP_SIZE)        // 256

// =============================================================================
// INT4 Packing Utilities
// =============================================================================

__device__ __forceinline__ uint32_t pack_int4x8(const int8_t* vals) {
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result |= ((uint32_t)(vals[i] & 0xF)) << (i * 4);
    }
    return result;
}

__device__ __forceinline__ void unpack_int4_byte(uint8_t byte, int8_t& v0, int8_t& v1) {
    v0 = (int8_t)(byte & 0xF);
    v0 = (v0 >= 8) ? (v0 - 16) : v0;
    v1 = (int8_t)((byte >> 4) & 0xF);
    v1 = (v1 >= 8) ? (v1 - 16) : v1;
}

// =============================================================================
// Fused Activation Pipeline (rrs_fused_v2)
// Single kernel: permutation -> FWHT -> quantize
// 200-600x faster than unfused reference
// =============================================================================

__global__ void __launch_bounds__(256)
tcq4_rrs_fused_activation_kernel(
    const float* __restrict__ in,
    const int32_t* __restrict__ perm,  // Can be NULL for identity
    block_rrs_int4* __restrict__ out,
    int M, int K
) {
    const int m = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    const int k_start = blk * 256;

    extern __shared__ float smem[];

    // =========================================================================
    // Stage 1: Load with optional permutation
    // =========================================================================
    float val = 0.0f;
    if (k_start + tid < K) {
        int src_idx = perm ? perm[k_start + tid] : (k_start + tid);
        val = in[m * K + src_idx];
    }
    smem[tid] = val;
    __syncthreads();

    // =========================================================================
    // Stage 2: In-place FWHT (8 stages for 256 elements)
    // =========================================================================
    #pragma unroll
    for (int h = 1; h < 256; h *= 2) {
        int grp = tid / (h * 2);
        int pos = tid % (h * 2);

        if (pos < h) {
            int j = grp * h * 2 + pos;
            int k_idx = j + h;
            float a = smem[j];
            float b = smem[k_idx];
            smem[j] = a + b;
            smem[k_idx] = a - b;
        }
        __syncthreads();
    }

    // Normalize by 1/sqrt(256) = 1/16
    constexpr float norm = 1.0f / 16.0f;
    smem[tid] *= norm;
    __syncthreads();

    // =========================================================================
    // Stage 3: Warp-optimized max reduction (on normalized values)
    // =========================================================================
    float abs_val = fabsf(smem[tid]);

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        abs_val = fmaxf(abs_val, __shfl_xor_sync(0xffffffff, abs_val, offset));
    }

    // Cross-warp reduction
    __shared__ float warp_max[8];
    int lane = tid & 31;
    int warp_id = tid >> 5;

    if (lane == 0) {
        warp_max[warp_id] = abs_val;
    }
    __syncthreads();

    float max_abs;
    if (tid < 8) {
        max_abs = warp_max[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            max_abs = fmaxf(max_abs, __shfl_xor_sync(0xff, max_abs, offset));
        }
        if (tid == 0) {
            if (max_abs < 1e-10f) max_abs = 1.0f;
            warp_max[0] = max_abs;
        }
    }
    __syncthreads();
    max_abs = warp_max[0];

    // =========================================================================
    // Stage 4: Quantize and pack
    // =========================================================================
    block_rrs_int4& output = out[m * (K / 256) + blk];

    if (tid < 128) {
        // Each thread handles 2 consecutive values (already normalized in smem)
        float v0 = smem[tid * 2];
        float v1 = smem[tid * 2 + 1];

        float scaled0 = v0 * (7.0f / max_abs);
        float scaled1 = v1 * (7.0f / max_abs);

        int8_t q0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, rintf(scaled0)));
        int8_t q1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, rintf(scaled1)));

        output.qs[tid] = ((uint8_t)(q1 & 0xF) << 4) | ((uint8_t)(q0 & 0xF));
    }

    if (tid == 0) {
        output.smooth_scale = max_abs;
    }
}

// TC version with precomputed group sums for zero-point correction
__global__ void __launch_bounds__(256)
tcq4_rrs_fused_activation_tc_kernel(
    const float* __restrict__ in,
    const int32_t* __restrict__ perm,
    block_rrs_int4_tc* __restrict__ out,
    int M, int K
) {
    const int m = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    const int k_start = blk * 256;

    extern __shared__ float smem[];
    __shared__ int8_t s_quant[256];
    __shared__ int32_t s_group_sum[8];

    // Stage 1: Load with optional permutation
    float val = 0.0f;
    if (k_start + tid < K) {
        int src_idx = perm ? perm[k_start + tid] : (k_start + tid);
        val = in[m * K + src_idx];
    }
    smem[tid] = val;
    __syncthreads();

    // Stage 2: FWHT
    #pragma unroll
    for (int h = 1; h < 256; h *= 2) {
        int grp = tid / (h * 2);
        int pos = tid % (h * 2);

        if (pos < h) {
            int j = grp * h * 2 + pos;
            int k_idx = j + h;
            float a = smem[j];
            float b = smem[k_idx];
            smem[j] = a + b;
            smem[k_idx] = a - b;
        }
        __syncthreads();
    }

    constexpr float norm = 1.0f / 16.0f;
    val = smem[tid] * norm;

    // Stage 3: Max reduction
    float abs_val = fabsf(val);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        abs_val = fmaxf(abs_val, __shfl_xor_sync(0xffffffff, abs_val, offset));
    }

    __shared__ float warp_max[8];
    int lane = tid & 31;
    int warp_id = tid >> 5;

    if (lane == 0) warp_max[warp_id] = abs_val;
    __syncthreads();

    float max_abs;
    if (tid < 8) {
        max_abs = warp_max[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            max_abs = fmaxf(max_abs, __shfl_xor_sync(0xff, max_abs, offset));
        }
        if (tid == 0) {
            if (max_abs < 1e-10f) max_abs = 1.0f;
            warp_max[0] = max_abs;
        }
    }
    __syncthreads();
    max_abs = warp_max[0];

    // Stage 4: Quantize with group sums
    if (tid < 8) s_group_sum[tid] = 0;
    __syncthreads();

    float scaled = val * (7.0f / max_abs);
    int8_t q = (int8_t)fmaxf(-7.0f, fminf(7.0f, rintf(scaled)));
    s_quant[tid] = q;
    atomicAdd(&s_group_sum[tid / 32], (int32_t)q);
    __syncthreads();

    // Stage 5: Pack output
    block_rrs_int4_tc& output = out[m * (K / 256) + blk];

    if (tid < 128) {
        int8_t q0 = s_quant[tid * 2];
        int8_t q1 = s_quant[tid * 2 + 1];
        output.qs[tid] = ((uint8_t)(q1 & 0xF) << 4) | ((uint8_t)(q0 & 0xF));
    }

    if (tid == 0) output.smooth_scale = max_abs;
    if (tid < 8) output.sum_q[tid] = (int16_t)s_group_sum[tid];
}

// =============================================================================
// =============================================================================
// FUSED FWHT + GEMV v2d (M=1) - Optimized kernel for token generation
// =============================================================================
//
// Performance: 2.8-3.6x faster than separate kernels, 8-23x faster than v1
//
// Key optimization: Warp-parallel K-reduction
// - 32 outputs per block, 8 threads per output
// - Each output's K-dimension reduction is parallelized across 8 threads
// - FWHT computed once per K-tile per block (shared across 32 outputs)
//
// Grid:  ceil(N / 32) blocks
// Block: 256 threads
// Shared memory: ~1.3 KB
//

__global__ void __launch_bounds__(256)
tcq4_rrs_fused_gemv_kernel(
    const float* __restrict__ activations,  // [K] FP32 input activations
    const int32_t* __restrict__ perm,       // [K] Channel permutation (or nullptr)
    const block_tcq4_tile* __restrict__ B_tiles,  // [N/8][K/256] weight tiles
    float* __restrict__ C,                  // [N] FP32 output
    int N, int K
) {
    const int tid = threadIdx.x;
    
    // 32 outputs per block, 8 threads per output
    const int output_in_block = tid / 8;    // 0-31
    const int thread_in_output = tid % 8;   // 0-7 (each handles one group)
    
    const int n_out = blockIdx.x * 32 + output_in_block;
    const int num_k_tiles = K / TCQ4_TILE_K;
    const int n_tile = n_out / 8;           // Which weight tile row
    const int channel = n_out % 8;          // Which channel within tile
    
    // Shared memory layout
    extern __shared__ char shared_mem[];
    float* smem_fwht = (float*)shared_mem;                    // [256] FWHT workspace
    int8_t* smem_quant = (int8_t*)(smem_fwht + 256);          // [256] quantized activations
    float* smem_scale = (float*)(smem_quant + 256);           // [1] quantization scale
    int32_t* smem_group_sum = (int32_t*)(smem_scale + 1);     // [8] group sums
    
    float acc = 0.0f;
    
    // Process each K-tile (256 elements)
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_start = kt * TCQ4_TILE_K;
        
        // ===== Stage 1: Load activations with optional permutation =====
        float val = 0.0f;
        if (k_start + tid < K) {
            int src_idx = perm ? perm[k_start + tid] : (k_start + tid);
            val = activations[src_idx];
        }
        smem_fwht[tid] = val;
        __syncthreads();
        
        // ===== Stage 2: Fast Walsh-Hadamard Transform =====
        // In-place transform of 256 elements in shared memory
        #pragma unroll
        for (int h = 1; h < 256; h *= 2) {
            int grp = tid / (h * 2);
            int pos = tid % (h * 2);
            if (pos < h) {
                int j = grp * h * 2 + pos;
                float a = smem_fwht[j];
                float b = smem_fwht[j + h];
                smem_fwht[j] = a + b;
                smem_fwht[j + h] = a - b;
            }
            __syncthreads();
        }
        
        // Normalize: 1/sqrt(256) = 1/16
        val = smem_fwht[tid] * (1.0f / 16.0f);
        
        // ===== Stage 3: Find max for quantization scale =====
        float abs_val = fabsf(val);
        
        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            abs_val = fmaxf(abs_val, __shfl_xor_sync(0xffffffff, abs_val, offset));
        }
        
        // Block-level reduction
        __shared__ float warp_max[8];
        int warp_id = tid / 32;
        int lane = tid % 32;
        if (lane == 0) warp_max[warp_id] = abs_val;
        __syncthreads();
        
        float max_abs;
        if (tid < 8) {
            float m = warp_max[tid];
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                m = fmaxf(m, __shfl_xor_sync(0xff, m, offset));
            }
            if (tid == 0) *smem_scale = (m < 1e-10f) ? 1.0f : m;
        }
        __syncthreads();
        max_abs = *smem_scale;
        
        // ===== Stage 4: Quantize to INT4 and compute group sums =====
        if (tid < 8) smem_group_sum[tid] = 0;
        __syncthreads();
        
        float scaled = val * (7.0f / max_abs);
        int8_t q = (int8_t)fmaxf(-7.0f, fminf(7.0f, rintf(scaled)));
        smem_quant[tid] = q;
        atomicAdd(&smem_group_sum[tid / 32], (int32_t)q);
        __syncthreads();
        
        // ===== Stage 5: Parallel dot product (8 threads per output) =====
        // Each thread handles one of the 8 groups (32 elements each)
        // Check bounds: n_out < N AND n_tile is valid (for non-aligned N)
        const int n_tiles_total = (N + 7) / 8;
        const bool valid_output = (n_out < N) && (n_tile < n_tiles_total);
        
        float contrib = 0.0f;
        if (valid_output && thread_in_output < 8) {
            const block_tcq4_tile& tile = B_tiles[n_tile * num_k_tiles + kt];
            
            float S = __half2float(tile.S[channel]);
            float Z = __half2float(tile.Z[channel]);
            float a_scale_div7 = max_abs / 7.0f;
            
            int g = thread_in_output;  // This thread handles group g
            float b_scale = S * (float)tile.sc[channel][g] / 127.0f;
            float b_zero = Z * (float)tile.zc[channel][g] / 127.0f;
            
            // Weight bytes are at: channel * 16 + k_slice * 4 + byte_in_slice
            const int byte_base = channel * 16;
            int dot = 0;
            
            // Process 32 elements for this group (4 k_slices × 8 elements each)
            #pragma unroll
            for (int k_slice = 0; k_slice < 4; k_slice++) {
                // Load 4 bytes = 8 INT4 weights as one uint32
                uint32_t packed_w = *reinterpret_cast<const uint32_t*>(
                    &tile.tiles[g][byte_base + k_slice * 4]);
                
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int k_idx = g * 32 + k_slice * 8 + i;
                    int8_t a_q = smem_quant[k_idx];
                    
                    // Extract INT4 weight: byte = i/2, nibble = i%2
                    int shift = (i / 2) * 8 + (i % 2) * 4;
                    int8_t b_q = (packed_w >> shift) & 0xF;
                    if (b_q >= 8) b_q -= 16;  // Sign extend
                    
                    dot += (int)a_q * (int)b_q;
                }
            }
            
            int32_t sum_a = smem_group_sum[g];
            contrib = (float)dot * a_scale_div7 * b_scale 
                    + (float)sum_a * a_scale_div7 * b_zero;
        }
        
        // All threads participate in shuffle reduction (invalid threads contribute 0)
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            contrib += __shfl_xor_sync(0xffffffff, contrib, offset);
        }
        
        // Thread 0 of each 8-thread group accumulates the result
        if (thread_in_output == 0 && n_out < N) {
            acc += contrib;
        }
        __syncthreads();
    }
    
    // Write final result
    if (thread_in_output == 0 && n_out < N) {
        C[n_out] = acc;
    }
}

// Host wrapper for fused GEMV v2d
void tcq4_rrs_fused_gemv(
    const float* activations,
    const int32_t* perm,
    const void* B_tcq4,
    float* C,
    int N, int K,
    cudaStream_t stream
) {
    // Grid: ceil(N / 32) blocks, each handles 32 outputs
    int num_blocks = (N + 31) / 32;
    
    // Shared memory: 256*4 + 256 + 4 + 32 = 1316 bytes
    size_t smem_size = 256 * sizeof(float)    // FWHT workspace
                     + 256 * sizeof(int8_t)   // Quantized activations
                     + sizeof(float)          // Scale
                     + 8 * sizeof(int32_t);   // Group sums
    
    tcq4_rrs_fused_gemv_kernel<<<num_blocks, 256, smem_size, stream>>>(
        activations, perm,
        (const block_tcq4_tile*)B_tcq4,
        C, N, K
    );
}

// =============================================================================
// FUSED FWHT + GEMM for Small M (1-16)
// =============================================================================
//
// Extension of v2d kernel for small batch sizes (prompt processing with M<=16).
// Uses 2D grid: (ceil(N/32), M) - each block processes one row and 32 outputs.
//
// Performance: Maintains v2d speedups (2.8-3.6x) for small M values.
//

__global__ void __launch_bounds__(256)
tcq4_rrs_fused_gemm_smallM_kernel(
    const float* __restrict__ activations,  // [M, K] FP32 input activations (row-major)
    const int32_t* __restrict__ perm,       // [K] Channel permutation (or nullptr)
    const block_tcq4_tile* __restrict__ B_tiles, // [N/8][K/256] weight tiles
    float* __restrict__ C,                  // [M, N] FP32 output (row-major)
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int m = blockIdx.y;  // Which row (0 to M-1)
    
    if (m >= M) return;
    
    // 32 outputs per block, 8 threads per output
    const int output_in_block = tid / 8;    // 0-31
    const int thread_in_output = tid % 8;   // 0-7 (each handles one group)
    
    const int n_out = blockIdx.x * 32 + output_in_block;
    const int num_k_tiles = K / TCQ4_TILE_K;
    const int n_tile = n_out / 8;
    const int channel = n_out % 8;
    
    // Pointers for this row
    const float* act_row = activations + m * K;
    float* out_row = C + m * N;
    
    // Shared memory layout
    extern __shared__ char shared_mem[];
    float* smem_fwht = (float*)shared_mem;
    int8_t* smem_quant = (int8_t*)(smem_fwht + 256);
    float* smem_scale = (float*)(smem_quant + 256);
    int32_t* smem_group_sum = (int32_t*)(smem_scale + 1);
    
    float acc = 0.0f;
    
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_start = kt * TCQ4_TILE_K;
        
        // Stage 1: Load activations with optional permutation
        float val = 0.0f;
        if (k_start + tid < K) {
            int src_idx = perm ? perm[k_start + tid] : (k_start + tid);
            val = act_row[src_idx];
        }
        smem_fwht[tid] = val;
        __syncthreads();
        
        // Stage 2: Fast Walsh-Hadamard Transform
        #pragma unroll
        for (int h = 1; h < 256; h *= 2) {
            int grp = tid / (h * 2);
            int pos = tid % (h * 2);
            if (pos < h) {
                int j = grp * h * 2 + pos;
                float a = smem_fwht[j];
                float b = smem_fwht[j + h];
                smem_fwht[j] = a + b;
                smem_fwht[j + h] = a - b;
            }
            __syncthreads();
        }
        
        val = smem_fwht[tid] * (1.0f / 16.0f);
        
        // Stage 3: Find max for quantization scale
        float abs_val = fabsf(val);
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            abs_val = fmaxf(abs_val, __shfl_xor_sync(0xffffffff, abs_val, offset));
        }
        
        __shared__ float warp_max[8];
        int warp_id = tid / 32;
        int lane = tid % 32;
        if (lane == 0) warp_max[warp_id] = abs_val;
        __syncthreads();
        
        float max_abs;
        if (tid < 8) {
            float m_val = warp_max[tid];
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                m_val = fmaxf(m_val, __shfl_xor_sync(0xff, m_val, offset));
            }
            if (tid == 0) *smem_scale = (m_val < 1e-10f) ? 1.0f : m_val;
        }
        __syncthreads();
        max_abs = *smem_scale;
        
        // Stage 4: Quantize to INT4 and compute group sums
        if (tid < 8) smem_group_sum[tid] = 0;
        __syncthreads();
        
        float scaled = val * (7.0f / max_abs);
        int8_t q = (int8_t)fmaxf(-7.0f, fminf(7.0f, rintf(scaled)));
        smem_quant[tid] = q;
        atomicAdd(&smem_group_sum[tid / 32], (int32_t)q);
        __syncthreads();
        
        // Stage 5: Parallel dot product
        const int n_tiles_total = (N + 7) / 8;
        const bool valid_output = (n_out < N) && (n_tile < n_tiles_total);
        
        float contrib = 0.0f;
        if (valid_output && thread_in_output < 8) {
            const block_tcq4_tile& tile = B_tiles[n_tile * num_k_tiles + kt];
            
            float S = __half2float(tile.S[channel]);
            float Z = __half2float(tile.Z[channel]);
            float a_scale_div7 = max_abs / 7.0f;
            
            int g = thread_in_output;
            float b_scale = S * (float)tile.sc[channel][g] / 127.0f;
            float b_zero = Z * (float)tile.zc[channel][g] / 127.0f;
            
            const int byte_base = channel * 16;
            int dot = 0;
            
            #pragma unroll
            for (int k_slice = 0; k_slice < 4; k_slice++) {
                uint32_t packed_w = *reinterpret_cast<const uint32_t*>(
                    &tile.tiles[g][byte_base + k_slice * 4]);
                
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int k_idx = g * 32 + k_slice * 8 + i;
                    int8_t a_q = smem_quant[k_idx];
                    
                    int shift = (i / 2) * 8 + (i % 2) * 4;
                    int8_t b_q = (packed_w >> shift) & 0xF;
                    if (b_q >= 8) b_q -= 16;
                    
                    dot += (int)a_q * (int)b_q;
                }
            }
            
            int32_t sum_a = smem_group_sum[g];
            contrib = (float)dot * a_scale_div7 * b_scale 
                    + (float)sum_a * a_scale_div7 * b_zero;
        }
        
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            contrib += __shfl_xor_sync(0xffffffff, contrib, offset);
        }
        
        if (thread_in_output == 0 && n_out < N) {
            acc += contrib;
        }
        __syncthreads();
    }
    
    if (thread_in_output == 0 && n_out < N) {
        out_row[n_out] = acc;
    }
}

// Host wrapper for fused GEMM (small M)
void tcq4_rrs_fused_gemm_smallM(
    const float* activations,
    const int32_t* perm,
    const void* B_tcq4,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    // Grid: (ceil(N/32), M)
    dim3 grid((N + 31) / 32, M);
    dim3 block(256);
    
    size_t smem_size = 256 * sizeof(float)
                     + 256 * sizeof(int8_t)
                     + sizeof(float)
                     + 8 * sizeof(int32_t);
    
    tcq4_rrs_fused_gemm_smallM_kernel<<<grid, block, smem_size, stream>>>(
        activations, perm,
        (const block_tcq4_tile*)B_tcq4,
        C, M, N, K
    );
}

// =============================================================================
// RRS GEMV (M=1) - Scalar path for token generation (OLD - kept for reference)
// dp4a is more efficient than IMMA for single-row operations
// =============================================================================

__global__ void __launch_bounds__(256)
tcq4_rrs_gemv_kernel(
    const block_rrs_int4* __restrict__ A,
    const block_tcq4_tile* __restrict__ B_tiles,  // Tile format: [n_tile][k_tile]
    float* __restrict__ C,
    int N, int K
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int num_k_tiles = K / TCQ4_BLOCK_SIZE;

    // Tile-based indexing: n_tile = n / 8, channel = n % 8
    const int n_tile = n / 8;
    const int channel = n % 8;

    float sum = 0.0f;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const block_rrs_int4& a_blk = A[kt];
        const block_tcq4_tile& tile = B_tiles[n_tile * num_k_tiles + kt];

        float a_scale = a_blk.smooth_scale;
        float S = __half2float(tile.S[channel]);
        float Z = __half2float(tile.Z[channel]);

        // Process 8 K-groups per tile
        for (int g = 0; g < TCQ4_NUM_GROUPS; g++) {
            float b_scale = S * (float)tile.sc[channel][g] / 127.0f;
            float b_zero = Z * (float)tile.zc[channel][g] / 127.0f;

            int group_sum = 0;
            int dot = 0;

            // Each thread processes part of the 32-element group
            // B data is in IMMA format: lane L gets column L//4, rows [(L%4)*8 : +8]
            // For channel c, lanes c*4 to c*4+3 hold the data
            for (int i = tid; i < 32; i += blockDim.x) {
                // Unpack A (simple sequential format)
                int a_idx = g * 32 + i;
                int a_byte_idx = a_idx / 2;
                int a_nibble = a_idx % 2;
                int8_t a_q;
                if (a_nibble == 0) {
                    a_q = (int8_t)(a_blk.qs[a_byte_idx] & 0xF);
                } else {
                    a_q = (int8_t)((a_blk.qs[a_byte_idx] >> 4) & 0xF);
                }
                a_q = (a_q >= 8) ? (a_q - 16) : a_q;

                // Unpack B from IMMA tile format
                // k_slice = i / 8 (which 8-element slice), k_in_slice = i % 8
                int k_slice = i / 8;
                int k_in_slice = i % 8;
                int lane = channel * 4 + k_slice;
                int byte_offset = lane * 4 + k_in_slice / 2;
                int b_nibble = k_in_slice % 2;
                
                int8_t b_q;
                if (b_nibble == 0) {
                    b_q = (int8_t)(tile.tiles[g][byte_offset] & 0xF);
                } else {
                    b_q = (int8_t)((tile.tiles[g][byte_offset] >> 4) & 0xF);
                }
                b_q = (b_q >= 8) ? (b_q - 16) : b_q;

                dot += (int)a_q * (int)b_q;
                group_sum += (int)a_q;
            }

            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
                group_sum += __shfl_down_sync(0xffffffff, group_sum, offset);
            }

            if (tid == 0) {
                sum += (float)dot * (a_scale / 7.0f) * b_scale;
                sum += (float)group_sum * (a_scale / 7.0f) * b_zero;
            }
        }
    }

    if (tid == 0) {
        C[n] = sum;
    }
}

// TC version using precomputed group sums
__global__ void __launch_bounds__(256)
tcq4_rrs_gemv_tc_kernel(
    const block_rrs_int4_tc* __restrict__ A,
    const block_tcq4_tile* __restrict__ B_tiles,  // Tile format
    float* __restrict__ C,
    int N, int K
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int num_k_tiles = K / TCQ4_BLOCK_SIZE;

    // Tile-based indexing
    const int n_tile = n / 8;
    const int channel = n % 8;

    float sum = 0.0f;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const block_rrs_int4_tc& a_blk = A[kt];
        const block_tcq4_tile& tile = B_tiles[n_tile * num_k_tiles + kt];

        float a_scale = a_blk.smooth_scale;
        float S = __half2float(tile.S[channel]);
        float Z = __half2float(tile.Z[channel]);

        for (int g = 0; g < TCQ4_NUM_GROUPS; g++) {
            float b_scale = S * (float)tile.sc[channel][g] / 127.0f;
            float b_zero = Z * (float)tile.zc[channel][g] / 127.0f;

            int dot = 0;

            // Each thread processes part of the 32-element group
            for (int i = tid; i < 32; i += blockDim.x) {
                // Unpack A
                int a_idx = g * 32 + i;
                int a_byte_idx = a_idx / 2;
                int a_nibble = a_idx % 2;
                int8_t a_q;
                if (a_nibble == 0) {
                    a_q = (int8_t)(a_blk.qs[a_byte_idx] & 0xF);
                } else {
                    a_q = (int8_t)((a_blk.qs[a_byte_idx] >> 4) & 0xF);
                }
                a_q = (a_q >= 8) ? (a_q - 16) : a_q;

                // Unpack B from IMMA tile format
                int k_slice = i / 8;
                int k_in_slice = i % 8;
                int lane = channel * 4 + k_slice;
                int byte_offset = lane * 4 + k_in_slice / 2;
                int b_nibble = k_in_slice % 2;
                
                int8_t b_q;
                if (b_nibble == 0) {
                    b_q = (int8_t)(tile.tiles[g][byte_offset] & 0xF);
                } else {
                    b_q = (int8_t)((tile.tiles[g][byte_offset] >> 4) & 0xF);
                }
                b_q = (b_q >= 8) ? (b_q - 16) : b_q;

                dot += (int)a_q * (int)b_q;
            }

            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }

            if (tid == 0) {
                // Use precomputed sum_q for zero-point correction
                sum += (float)dot * (a_scale / 7.0f) * b_scale;
                sum += (float)a_blk.sum_q[g] * (a_scale / 7.0f) * b_zero;
            }
        }
    }

    if (tid == 0) {
        C[n] = sum;
    }
}

// =============================================================================
// RRS GEMM Scalar (M>1) - Fallback for non-TC architectures
// =============================================================================

__global__ void __launch_bounds__(256)
tcq4_rrs_gemm_scalar_kernel(
    const block_rrs_int4* __restrict__ A,
    const block_tcq4_tile* __restrict__ B_tiles,  // Tile format
    float* __restrict__ C,
    int M, int N, int K
) {
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int tid = threadIdx.x;
    const int num_k_tiles = K / TCQ4_BLOCK_SIZE;

    // Tile-based indexing
    const int n_tile = n / 8;
    const int channel = n % 8;

    float sum = 0.0f;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const block_rrs_int4& a_blk = A[m * num_k_tiles + kt];
        const block_tcq4_tile& tile = B_tiles[n_tile * num_k_tiles + kt];

        float a_scale = a_blk.smooth_scale;
        float S = __half2float(tile.S[channel]);
        float Z = __half2float(tile.Z[channel]);

        for (int g = 0; g < TCQ4_NUM_GROUPS; g++) {
            float b_scale = S * (float)tile.sc[channel][g] / 127.0f;
            float b_zero = Z * (float)tile.zc[channel][g] / 127.0f;

            int group_sum = 0;
            int dot = 0;

            for (int i = tid; i < 32; i += blockDim.x) {
                // Unpack A
                int a_idx = g * 32 + i;
                int a_byte_idx = a_idx / 2;
                int a_nibble = a_idx % 2;
                int8_t a_q;
                if (a_nibble == 0) {
                    a_q = (int8_t)(a_blk.qs[a_byte_idx] & 0xF);
                } else {
                    a_q = (int8_t)((a_blk.qs[a_byte_idx] >> 4) & 0xF);
                }
                a_q = (a_q >= 8) ? (a_q - 16) : a_q;

                // Unpack B from IMMA tile format
                int k_slice = i / 8;
                int k_in_slice = i % 8;
                int lane = channel * 4 + k_slice;
                int byte_offset = lane * 4 + k_in_slice / 2;
                int b_nibble = k_in_slice % 2;
                
                int8_t b_q;
                if (b_nibble == 0) {
                    b_q = (int8_t)(tile.tiles[g][byte_offset] & 0xF);
                } else {
                    b_q = (int8_t)((tile.tiles[g][byte_offset] >> 4) & 0xF);
                }
                b_q = (b_q >= 8) ? (b_q - 16) : b_q;

                dot += (int)a_q * (int)b_q;
                group_sum += (int)a_q;
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
                group_sum += __shfl_down_sync(0xffffffff, group_sum, offset);
            }

            if (tid == 0) {
                sum += (float)dot * (a_scale / 7.0f) * b_scale;
                sum += (float)group_sum * (a_scale / 7.0f) * b_zero;
            }
        }
    }

    if (tid == 0) {
        C[m * N + n] = sum;
    }
}

// =============================================================================
// RRS GEMM IMMA v12 - Optimized INT4 Tensor Core Kernel
// 75 TFLOPS on RTX 3090, 3x faster than Q4_K dp4a
// =============================================================================

__global__ void __launch_bounds__(V12_THREADS)
tcq4_rrs_gemm_imma_kernel(
    const block_rrs_int4_tc* __restrict__ A,
    const block_tcq4_tile* __restrict__ B_tiles,  // Tile format: [n_tile][k_tile]
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for K-tile staging (32 rows × 128 bytes for 256 INT4 values)
    __shared__ int8_t smem_A[V12_BLOCK_M * 128];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int warp_m = warp_id / V12_WARPS_N;
    const int warp_n = warp_id % V12_WARPS_N;

    const int group_id = lane_id / 4;
    const int tid_in_group = lane_id % 4;

    const int block_m = blockIdx.y * V12_BLOCK_M;
    const int block_n = blockIdx.x * V12_BLOCK_N;

    const int num_k_tiles = K / TCQ4_BLOCK_SIZE;
    const int n_tiles = (N + 7) / 8;  // Number of N-tiles (8 channels per tile)

    // FP32 accumulators (one per IMMA output element)
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Process one K-tile (256 elements = 8 groups) at a time
    for (int kt = 0; kt < num_k_tiles; kt++) {
        // Load entire K-tile to shared memory with ONE sync
        {
            constexpr int total_bytes = V12_BLOCK_M * 128;
            constexpr int LOAD_SIZE = 16;
            constexpr int total_loads = total_bytes / LOAD_SIZE;

            for (int idx = tid; idx < total_loads; idx += V12_THREADS) {
                int byte_offset = idx * LOAD_SIZE;
                int row = byte_offset / 128;
                int col = byte_offset % 128;
                int global_row = block_m + row;

                int8_t* dst = &smem_A[row * 128 + col];

                if (global_row < M) {
                    const int8_t* src = (const int8_t*)&A[global_row * num_k_tiles + kt].qs[col];
                    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
                } else {
                    *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();  // ONE sync per K-tile

        // Process tiles this warp handles
        int tile_m = block_m + warp_m * IMMA_M;
        int smem_row_base = warp_m * IMMA_M;

        // Precompute A scales (one per K-tile, shared across groups)
        int m0 = tile_m + group_id;
        int m1 = tile_m + group_id + 8;
        float smooth0 = (m0 < M) ? A[m0 * num_k_tiles + kt].smooth_scale : 0.0f;
        float smooth1 = (m1 < M) ? A[m1 * num_k_tiles + kt].smooth_scale : 0.0f;

        int tile_n = block_n + warp_n * IMMA_N;
        
        // NEW: Tile-based indexing for B
        // tile_n is the starting output channel for this warp
        // Each tile contains 8 channels, so n_tile = tile_n / 8
        int n_tile = tile_n / 8;
        
        // tid_in_group gives us which pair of channels within the IMMA N=8 output
        // c0, c1 are the two channels this thread computes
        int c0_in_tile = tid_in_group * 2;      // 0, 2, 4, 6
        int c1_in_tile = c0_in_tile + 1;        // 1, 3, 5, 7
        int c0_global = tile_n + tid_in_group * 2;
        int c1_global = c0_global + 1;

        // Get pointer to the B tile for this (n_tile, k_tile)
        const block_tcq4_tile* tile = (n_tile < n_tiles) ? &B_tiles[n_tile * num_k_tiles + kt] : nullptr;

        // Precompute B super-scales from tile (per-channel)
        float S0 = 0.0f, Z0 = 0.0f, S1 = 0.0f, Z1 = 0.0f;
        if (tile && c0_global < N) {
            S0 = __half2float(tile->S[c0_in_tile]);
            Z0 = __half2float(tile->Z[c0_in_tile]);
        }
        if (tile && c1_global < N) {
            S1 = __half2float(tile->S[c1_in_tile]);
            Z1 = __half2float(tile->Z[c1_in_tile]);
        }

        // Preload sum_q for zero-point correction
        int16_t sum_qa0[8], sum_qa1[8];
        if (m0 < M) {
            #pragma unroll
            for (int g = 0; g < 8; g++) sum_qa0[g] = A[m0 * num_k_tiles + kt].sum_q[g];
        }
        if (m1 < M) {
            #pragma unroll
            for (int g = 0; g < 8; g++) sum_qa1[g] = A[m1 * num_k_tiles + kt].sum_q[g];
        }

        // Process 8 groups from shared memory (NO additional syncs!)
        #pragma unroll
        for (int g = 0; g < TCQ4_NUM_GROUPS; g++) {
            // Load A fragment from shared memory
            uint32_t a0, a1;
            {
                int smem_row0 = smem_row_base + group_id;
                int smem_row1 = smem_row0 + 8;
                int smem_col = g * 16 + tid_in_group * 4;

                uint32_t packed0 = *reinterpret_cast<const uint32_t*>(&smem_A[smem_row0 * 128 + smem_col]);
                uint32_t packed1 = *reinterpret_cast<const uint32_t*>(&smem_A[smem_row1 * 128 + smem_col]);

                int8_t vals0[8], vals1[8];
                const int8_t* p0 = reinterpret_cast<const int8_t*>(&packed0);
                const int8_t* p1 = reinterpret_cast<const int8_t*>(&packed1);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int8_t v0_lo = p0[i] & 0xF;
                    int8_t v0_hi = (p0[i] >> 4) & 0xF;
                    int8_t v1_lo = p1[i] & 0xF;
                    int8_t v1_hi = (p1[i] >> 4) & 0xF;

                    vals0[i*2] = (v0_lo >= 8) ? (v0_lo - 16) : v0_lo;
                    vals0[i*2+1] = (v0_hi >= 8) ? (v0_hi - 16) : v0_hi;
                    vals1[i*2] = (v1_lo >= 8) ? (v1_lo - 16) : v1_lo;
                    vals1[i*2+1] = (v1_hi >= 8) ? (v1_hi - 16) : v1_hi;
                }

                a0 = pack_int4x8(vals0);
                a1 = pack_int4x8(vals1);
            }

            // NEW: Load B fragment directly from tile (already in IMMA format!)
            // tile->tiles[g] contains 128 bytes in IMMA B fragment order
            // Lane L loads bytes [L*4 : L*4+4] as uint32
            uint32_t b0 = 0;
            if (tile && tile_n < N) {
                b0 = ((const uint32_t*)tile->tiles[g])[lane_id];
            }

            // Execute IMMA mma.sync.m16n8k32.s4.s4
            int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if __CUDA_ARCH__ >= 750
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                : "r"(a0), "r"(a1), "r"(b0), "r"(d0), "r"(d1), "r"(d2), "r"(d3)
            );
#endif

            // Apply per-group B scales from tile (sc[channel][group])
            float s_b0_g = 0.0f, z_b0_g = 0.0f, s_b1_g = 0.0f, z_b1_g = 0.0f;
            if (tile && c0_global < N) {
                s_b0_g = S0 * (float)tile->sc[c0_in_tile][g] / 127.0f;
                z_b0_g = Z0 * (float)tile->zc[c0_in_tile][g] / 127.0f;
            }
            if (tile && c1_global < N) {
                s_b1_g = S1 * (float)tile->sc[c1_in_tile][g] / 127.0f;
                z_b1_g = Z1 * (float)tile->zc[c1_in_tile][g] / 127.0f;
            }

            // Accumulate with correct scale math:
            // result = int_dot * (smooth_scale/7) * b_scale + sum_a * (smooth_scale/7) * b_zero
            float a_scale0_div7 = smooth0 / 7.0f;
            float a_scale1_div7 = smooth1 / 7.0f;

            acc[0] += a_scale0_div7 * (s_b0_g * (float)d0 + z_b0_g * (float)sum_qa0[g]);
            acc[1] += a_scale0_div7 * (s_b1_g * (float)d1 + z_b1_g * (float)sum_qa0[g]);
            acc[2] += a_scale1_div7 * (s_b0_g * (float)d2 + z_b0_g * (float)sum_qa1[g]);
            acc[3] += a_scale1_div7 * (s_b1_g * (float)d3 + z_b1_g * (float)sum_qa1[g]);
        }

        __syncthreads();  // Sync before loading next K-tile
    }

    // Store results
    int tile_m = block_m + warp_m * IMMA_M;
    int tile_n = block_n + warp_n * IMMA_N;

    int row0 = tile_m + group_id;
    int row1 = tile_m + group_id + 8;
    int col0 = tile_n + tid_in_group * 2;
    int col1 = col0 + 1;

    if (row0 < M && col0 < N) C[row0 * N + col0] = acc[0];
    if (row0 < M && col1 < N) C[row0 * N + col1] = acc[1];
    if (row1 < M && col0 < N) C[row1 * N + col0] = acc[2];
    if (row1 < M && col1 < N) C[row1 * N + col1] = acc[3];
}

// =============================================================================
// Host API Functions
// =============================================================================

void tcq4_rrs_fwht_quantize(
    const float* x,
    void* y,
    int K,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid((K + 255) / 256, batch_size);
    size_t smem = 256 * sizeof(float);
    tcq4_rrs_fused_activation_kernel<<<grid, 256, smem, stream>>>(
        x, nullptr, (block_rrs_int4*)y, batch_size, K
    );
}

void tcq4_rrs_perm_fwht_quantize(
    const float* x,
    void* y,
    const int32_t* perm,
    int K,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid((K + 255) / 256, batch_size);
    size_t smem = 256 * sizeof(float);
    tcq4_rrs_fused_activation_kernel<<<grid, 256, smem, stream>>>(
        x, perm, (block_rrs_int4*)y, batch_size, K
    );
}

void tcq4_rrs_fwht_quantize_tc(
    const float* x,
    void* y,
    int K,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid((K + 255) / 256, batch_size);
    size_t smem = 256 * sizeof(float);
    tcq4_rrs_fused_activation_tc_kernel<<<grid, 256, smem, stream>>>(
        x, nullptr, (block_rrs_int4_tc*)y, batch_size, K
    );
}

void tcq4_rrs_perm_fwht_quantize_tc(
    const float* x,
    void* y,
    const int32_t* perm,
    int K,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid((K + 255) / 256, batch_size);
    size_t smem = 256 * sizeof(float);
    tcq4_rrs_fused_activation_tc_kernel<<<grid, 256, smem, stream>>>(
        x, perm, (block_rrs_int4_tc*)y, batch_size, K
    );
}

void tcq4_rrs_gemv(
    const void* A_rrs,
    const void* B_tcq4,
    float* C,
    int N, int K,
    cudaStream_t stream
) {
    tcq4_rrs_gemv_kernel<<<N, 256, 0, stream>>>(
        (const block_rrs_int4*)A_rrs,
        (const block_tcq4_tile*)B_tcq4,
        C, N, K
    );
}

void tcq4_rrs_gemv_tc(
    const void* A_rrs,
    const void* B_tcq4,
    float* C,
    int N, int K,
    cudaStream_t stream
) {
    tcq4_rrs_gemv_tc_kernel<<<N, 256, 0, stream>>>(
        (const block_rrs_int4_tc*)A_rrs,
        (const block_tcq4_tile*)B_tcq4,
        C, N, K
    );
}

void tcq4_rrs_gemm(
    const void* A_rrs,
    const void* B_tcq4,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid(N, M);
    tcq4_rrs_gemm_scalar_kernel<<<grid, 256, 0, stream>>>(
        (const block_rrs_int4*)A_rrs,
        (const block_tcq4_tile*)B_tcq4,
        C, M, N, K
    );
}

void tcq4_rrs_gemm_imma(
    const void* A_rrs,
    const void* B_tcq4,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((N + V12_BLOCK_N - 1) / V12_BLOCK_N, (M + V12_BLOCK_M - 1) / V12_BLOCK_M);
    tcq4_rrs_gemm_imma_kernel<<<grid, V12_THREADS, 0, stream>>>(
        (const block_rrs_int4_tc*)A_rrs,
        (const block_tcq4_tile*)B_tcq4,
        C, M, N, K
    );
}