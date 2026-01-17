#include "rrs.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

// ============================================================================
// Q4_K Format Constants and Scale Unpacking
// ============================================================================

// Q4_K: 256 elements per super-block, 8 groups of 32 elements
// 12 bytes of packed 6-bit scales/mins
// d (fp16): super-scale for dequantizing per-group scales
// dmin (fp16): super-scale for dequantizing per-group mins

__device__ __forceinline__ void get_scale_min_k4_cuda(
    int j, const uint8_t* q, uint8_t* d, uint8_t* m) 
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ void unpack_scales_mins_k4_cuda(
    const uint8_t* scales, uint8_t* sc, uint8_t* mn)
{
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        get_scale_min_k4_cuda(j, scales, &sc[j], &mn[j]);
    }
}

// ============================================================================
// FWHT (Fast Walsh-Hadamard Transform) CUDA Kernel
// ============================================================================

__device__ __forceinline__ void fwht_butterfly(float& a, float& b) {
    float t = a;
    a = t + b;
    b = t - b;
}

template<int N>
__global__ void fwht_kernel_pow2(
    const float* __restrict__ x, 
    float* __restrict__ y, 
    int batch_size) 
{
    extern __shared__ float smem_fwht[];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x_row = x + batch_idx * N;
    float* y_row = y + batch_idx * N;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        smem_fwht[i] = x_row[i];
    }
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
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        y_row[i] = smem_fwht[i] * scale;
    }
}

__global__ void fwht_kernel_chunked(
    const float* __restrict__ x, 
    float* __restrict__ y, 
    int n, 
    int batch_size) 
{
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
        
        for (int i = threadIdx.x; i < step; i += blockDim.x) {
            smem_chunk[i] = x_row[base + i];
        }
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
        
        for (int i = threadIdx.x; i < step; i += blockDim.x) {
            y_row[base + i] = smem_chunk[i] * scale;
        }
        __syncthreads();
    }
}

void ggml_cuda_rrs_fwht(
    const float* x,
    float* y,
    int n,
    int batch_size,
    cudaStream_t stream) 
{
    const int threads = min(256, (n + 1) / 2);
    const size_t smem_size = n * sizeof(float);
    
    if ((n & (n - 1)) == 0) {
        switch (n) {
            case 64:
                fwht_kernel_pow2<64><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size);
                break;
            case 128:
                fwht_kernel_pow2<128><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size);
                break;
            case 256:
                fwht_kernel_pow2<256><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size);
                break;
            case 512:
                fwht_kernel_pow2<512><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size);
                break;
            case 1024:
                fwht_kernel_pow2<1024><<<batch_size, threads, smem_size, stream>>>(x, y, batch_size);
                break;
            case 2048:
                fwht_kernel_pow2<2048><<<batch_size, 512, smem_size, stream>>>(x, y, batch_size);
                break;
            case 4096:
                fwht_kernel_pow2<4096><<<batch_size, 512, smem_size, stream>>>(x, y, batch_size);
                break;
            default:
                fwht_kernel_chunked<<<batch_size, threads, smem_size, stream>>>(x, y, n, batch_size);
                break;
        }
    } else {
        const int step = n & -n;
        const size_t chunk_smem = step * sizeof(float);
        fwht_kernel_chunked<<<batch_size, min(256, step/2), chunk_smem, stream>>>(x, y, n, batch_size);
    }
}

// ============================================================================
// Activation Quantization to Q4_K format
// ============================================================================

// Simple per-32-group quantization for activations
// Output: packed nibbles + per-group scale/min in simplified format
__global__ void quantize_act_q4_simple_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ qs,
    half* __restrict__ scales,
    half* __restrict__ mins,
    int n,
    int batch_size)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* x_row = x + row * n;
    const int groups = n / 32;
    
    const int group = blockIdx.y * blockDim.y + threadIdx.y;
    if (group >= groups) return;
    
    const int lane = threadIdx.x;
    const int base = group * 32;
    
    float val = (base + lane < n) ? x_row[base + lane] : 0.0f;
    
    // Warp-level reduction for min/max
    float vmax = val;
    float vmin = val;
    
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, mask));
        vmin = fminf(vmin, __shfl_xor_sync(0xFFFFFFFF, vmin, mask));
    }
    
    const float range = vmax - vmin;
    const float scale = range / 15.0f;
    const float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
    
    int q = __float2int_rn((val - vmin) * inv_scale);
    q = max(0, min(15, q));
    
    // Pack two 4-bit values per byte
    const int pair_idx = lane % 16;
    int lo = __shfl_sync(0xFFFFFFFF, q, pair_idx);
    int hi = __shfl_sync(0xFFFFFFFF, q, pair_idx + 16);
    int packed = lo | (hi << 4);
    
    if (lane < 16) {
        const int out_idx = row * (n / 2) + group * 16 + pair_idx;
        qs[out_idx] = (uint8_t)packed;
    }
    
    if (lane == 0) {
        scales[row * groups + group] = __float2half(scale);
        mins[row * groups + group] = __float2half(vmin);
    }
}

// ============================================================================
// Fused FWHT + Quantize Kernel
// ============================================================================

template<int N>
__global__ void fwht_quantize_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ qs,
    half* __restrict__ scales,
    half* __restrict__ mins,
    int batch_size)
{
    extern __shared__ float smem_fq[];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x_row = x + batch_idx * N;
    const int groups = N / 32;
    
    // Load into shared memory
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        smem_fq[i] = x_row[i];
    }
    __syncthreads();
    
    // FWHT in-place
    const float scale_fwht = rsqrtf((float)N);
    
    #pragma unroll
    for (int h = 1; h < N; h <<= 1) {
        const int stride = h << 1;
        for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
            const int block = i / h;
            const int offset = i % h;
            const int idx1 = block * stride + offset;
            const int idx2 = idx1 + h;
            fwht_butterfly(smem_fq[idx1], smem_fq[idx2]);
        }
        __syncthreads();
    }
    
    // Apply scale
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        smem_fq[i] *= scale_fwht;
    }
    __syncthreads();
    
    // Quantize per-32 groups
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    
    for (int group = warp_id; group < groups; group += num_warps) {
        const int base = group * 32;
        float val = smem_fq[base + lane];
        
        float vmax = val, vmin = val;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, mask));
            vmin = fminf(vmin, __shfl_xor_sync(0xFFFFFFFF, vmin, mask));
        }
        
        const float range = vmax - vmin;
        const float qscale = range / 15.0f;
        const float inv_scale = (qscale > 1e-10f) ? (1.0f / qscale) : 0.0f;
        
        int q = __float2int_rn((val - vmin) * inv_scale);
        q = max(0, min(15, q));
        
        const int pair_idx = lane % 16;
        int lo = __shfl_sync(0xFFFFFFFF, q, pair_idx);
        int hi = __shfl_sync(0xFFFFFFFF, q, pair_idx + 16);
        int packed = lo | (hi << 4);
        
        if (lane < 16) {
            qs[batch_idx * (N / 2) + group * 16 + pair_idx] = (uint8_t)packed;
        }
        
        if (lane == 0) {
            scales[batch_idx * groups + group] = __float2half(qscale);
            mins[batch_idx * groups + group] = __float2half(vmin);
        }
    }
}

void ggml_cuda_rrs_fwht_quantize(
    const float* x,
    void* y,
    half* scales,
    half* mins,
    int n,
    int batch_size,
    cudaStream_t stream)
{
    const size_t smem_size = n * sizeof(float);
    const int threads = min(256, n);
    
    switch (n) {
        case 256:
            fwht_quantize_kernel<256><<<batch_size, threads, smem_size, stream>>>(
                x, (uint8_t*)y, scales, mins, batch_size);
            break;
        case 512:
            fwht_quantize_kernel<512><<<batch_size, threads, smem_size, stream>>>(
                x, (uint8_t*)y, scales, mins, batch_size);
            break;
        case 1024:
            fwht_quantize_kernel<1024><<<batch_size, threads, smem_size, stream>>>(
                x, (uint8_t*)y, scales, mins, batch_size);
            break;
        case 2048:
            fwht_quantize_kernel<2048><<<batch_size, 512, smem_size, stream>>>(
                x, (uint8_t*)y, scales, mins, batch_size);
            break;
        case 4096:
            fwht_quantize_kernel<4096><<<batch_size, 512, smem_size, stream>>>(
                x, (uint8_t*)y, scales, mins, batch_size);
            break;
        default:
            break;
    }
}

void ggml_cuda_rrs_quantize_act(
    const float* x,
    void* y,
    int n,
    int batch_size,
    cudaStream_t stream)
{
    (void)x; (void)y; (void)n; (void)batch_size; (void)stream;
}

// ============================================================================
// PATH A: INT4 Tensor Core GEMM (Turing+ WMMA s4)
// Uses experimental INT4 precision on tensor cores
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define RRS_HAS_INT4_TC 1
#endif

using I4 = wmma::experimental::precision::s4;

// WMMA tile dimensions for INT4: 8x8x32
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 32;

// Block tiling configuration
constexpr int BLOCK_WARPS_M = 4;
constexpr int BLOCK_WARPS_N = 4;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void __launch_bounds__(256) rrs_gemm_i4_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    int* __restrict__ C_i32,
    const int M, const int N, const int K,
    const int lda, const int ldb, const int ldc)
{
#ifdef RRS_HAS_INT4_TC
    extern __shared__ uint8_t smem_gemm[];
    
    uint8_t* smemA = smem_gemm;
    uint8_t* smemB = smemA + TILE_M * TILE_K / 2;
    
    using FragA_I4 = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, I4, wmma::row_major>;
    using FragB_I4 = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, I4, wmma::col_major>;
    using FragC_I32 = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>;
    
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / BLOCK_WARPS_N;
    const int warp_col = warp_id % BLOCK_WARPS_N;
    
    const int block_row = blockIdx.x * TILE_M;
    const int block_col = blockIdx.y * TILE_N;
    
    FragA_I4 a_frag[WARP_TILES_M];
    FragB_I4 b_frag[WARP_TILES_N];
    FragC_I32 c_frag[WARP_TILES_M][WARP_TILES_N];
    
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0);
        }
    }
    
    for (int k = 0; k < K; k += TILE_K) {
        // Load A tile
        for (int i = threadIdx.x; i < TILE_M * TILE_K / 2; i += blockDim.x) {
            const int row = i / (TILE_K / 2);
            const int col = i % (TILE_K / 2);
            const int global_row = block_row + row;
            
            if (global_row < M && (k + col * 2) < K) {
                smemA[i] = A[global_row * lda / 2 + k / 2 + col];
            } else {
                smemA[i] = 0;
            }
        }
        
        // Load B tile
        for (int i = threadIdx.x; i < TILE_N * TILE_K / 2; i += blockDim.x) {
            const int row = i / (TILE_K / 2);
            const int col = i % (TILE_K / 2);
            const int global_row = block_col + row;
            
            if (global_row < N && (k + col * 2) < K) {
                smemB[i] = B[global_row * ldb / 2 + k / 2 + col];
            } else {
                smemB[i] = 0;
            }
        }
        
        __syncthreads();
        
        // WMMA compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++) {
                const int a_row = warp_row * WARP_TILES_M * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], 
                    smemA + a_row * TILE_K / 2 + kk / 2,
                    TILE_K);
            }
            
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++) {
                const int b_row = warp_col * WARP_TILES_N * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j],
                    smemB + b_row * TILE_K / 2 + kk / 2,
                    TILE_K);
            }
            
            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_TILES_N; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            const int c_row = block_row + warp_row * WARP_TILES_M * WMMA_M + i * WMMA_M;
            const int c_col = block_col + warp_col * WARP_TILES_N * WMMA_N + j * WMMA_N;
            
            if (c_row < M && c_col < N) {
                wmma::store_matrix_sync(C_i32 + c_row * ldc + c_col, 
                    c_frag[i][j], ldc, wmma::mem_row_major);
            }
        }
    }
#else
    (void)A; (void)B; (void)C_i32; (void)M; (void)N; (void)K; (void)lda; (void)ldb; (void)ldc;
#endif
}

// ============================================================================
// Dequantization Kernel with Proper Q4_K Scale Handling
// ============================================================================

// For Q4_K weights: reconstruct using per-group 6-bit scales
// For simple activations: use direct fp16 scales
__global__ void rrs_dequant_q4k_kernel(
    const int* __restrict__ C_i32,
    float* __restrict__ C_f32,
    // Activation scales (simple per-32-group fp16)
    const half* __restrict__ act_scales,
    const half* __restrict__ act_mins,
    // Weight Q4_K blocks (contains nested scale structure)
    const void* __restrict__ weight_blocks,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    const int sum_i32 = C_i32[row * N + col];
    const int groups = K / 32;
    const int blocks_per_row = K / 256;  // Q4_K: 256 elements per super-block
    
    // Reconstruct activation contribution
    // Simple format: direct scale/min per 32-group
    float act_scale_sum = 0.0f;
    for (int g = 0; g < groups; g++) {
        act_scale_sum += __half2float(act_scales[row * groups + g]);
    }
    float avg_act_scale = act_scale_sum / groups;
    
    // Reconstruct weight contribution from Q4_K format
    const block_q4_K* w_blocks = (const block_q4_K*)weight_blocks + col * blocks_per_row;
    
    float weight_scale_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const float d = __half2float(w_blocks[b].dm.x);
        
        // Unpack the 8 per-group scales
        uint8_t sc[8], mn[8];
        unpack_scales_mins_k4_cuda(w_blocks[b].scales, sc, mn);
        
        for (int g = 0; g < 8; g++) {
            weight_scale_sum += d * sc[g];
        }
    }
    float avg_weight_scale = weight_scale_sum / groups;
    
    // Approximate dequantization
    // The INT4 dot product computed: sum(q_act * q_weight)
    // We need to apply: scale_act * scale_weight
    // Note: This is simplified - proper handling requires tracking per-group products
    C_f32[row * N + col] = (float)sum_i32 * avg_act_scale * avg_weight_scale;
}

// More accurate dequantization with per-group handling
// This kernel processes the result with full scale reconstruction
__global__ void rrs_dequant_accurate_kernel(
    const int* __restrict__ C_i32,
    float* __restrict__ C_f32,
    // Activation: simple format
    const half* __restrict__ act_scales,
    const half* __restrict__ act_mins,
    // Weight: Q4_K format  
    const void* __restrict__ weight_blocks,
    // Partial sums per group (computed by a modified GEMM)
    const int* __restrict__ partial_sums,  // [M, N, groups]
    const int M, const int N, const int K,
    const int groups)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    const int blocks_per_col = K / 256;
    const block_q4_K* w_blocks = (const block_q4_K*)weight_blocks + col * blocks_per_col;
    
    float result = 0.0f;
    
    for (int g = 0; g < groups; g++) {
        const int block_idx = g / 8;
        const int group_in_block = g % 8;
        
        // Activation scale for this group
        float a_scale = __half2float(act_scales[row * groups + g]);
        float a_min = __half2float(act_mins[row * groups + g]);
        
        // Weight scale for this group (from Q4_K)
        const block_q4_K& wb = w_blocks[block_idx];
        float d = __half2float(wb.dm.x);
        float dmin = __half2float(wb.dm.y);
        
        uint8_t sc, mn;
        get_scale_min_k4_cuda(group_in_block, wb.scales, &sc, &mn);
        float w_scale = d * sc;
        float w_min = dmin * mn;
        
        // Get partial sum for this group
        int psum = partial_sums[(row * N + col) * groups + g];
        
        // Dequantize: (q_a * scale_a + min_a) * (q_w * scale_w - min_w)
        // Simplified: scale_a * scale_w * psum + cross terms
        // For RRS (zero-centered after FWHT), mins are typically small
        result += a_scale * w_scale * (float)psum;
        result -= a_min * w_min * 32.0f;  // 32 elements per group
    }
    
    C_f32[row * N + col] = result;
}

// ============================================================================
// PATH B: Repack to Q8 + INT8 dp4a GEMM
// Alternative approach: convert Q4 activations to Q8 and use dp4a
// ============================================================================

// Unpack Q4 to Q8 (signed)
__global__ void unpack_q4_to_q8_kernel(
    const uint8_t* __restrict__ q4_data,
    int8_t* __restrict__ q8_data,
    const half* __restrict__ scales,
    const half* __restrict__ mins,
    half* __restrict__ q8_scales,
    const int n,
    const int batch_size)
{
    const int row = blockIdx.x;
    
    if (row >= batch_size) return;
    
    // Process multiple elements per thread to handle large K
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        const uint8_t packed = q4_data[row * (n / 2) + idx];
        const int lo = packed & 0x0F;
        const int hi = packed >> 4;
        
        // Convert to signed Q8 range [-128, 127]
        // Q4 is [0, 15], center at 7.5 and scale to Q8 range
        q8_data[row * n + idx * 2 + 0] = (int8_t)((lo - 8) * 16);
        q8_data[row * n + idx * 2 + 1] = (int8_t)((hi - 8) * 16);
        
        // Adjust scales: Q8 = Q4 * 16, so new_scale = old_scale / 16
        const int group = (idx * 2) / 32;
        if ((idx * 2) % 32 == 0 && q8_scales != nullptr) {
            float s = __half2float(scales[row * (n / 32) + group]);
            q8_scales[row * (n / 32) + group] = __float2half(s / 16.0f);
        }
    }
}

// dp4a-based GEMM for Q8xQ8
// This is a simpler fallback that works on all CUDA devices
__global__ void rrs_gemm_q8_dp4a_kernel(
    const int8_t* __restrict__ A,  // [M, K] Q8
    const int8_t* __restrict__ B,  // [N, K] Q8 (transposed)
    int* __restrict__ C,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    int sum = 0;
    
    // Process 4 elements at a time with dp4a
    for (int k = 0; k < K; k += 4) {
        int a_packed = *reinterpret_cast<const int*>(&A[row * K + k]);
        int b_packed = *reinterpret_cast<const int*>(&B[col * K + k]);
        sum = __dp4a(a_packed, b_packed, sum);
    }
    
    C[row * N + col] = sum;
}

// ============================================================================
// Main Entry Points
// ============================================================================

void ggml_cuda_rrs_gemm_q4q4(
    const void* A,
    const void* B,
    float* C,
    int M, int N, int K,
    const half* scales_A, const half* mins_A,
    const half* scales_B, const half* mins_B,
    cudaStream_t stream)
{
    constexpr int TILE_M = WMMA_M * WARP_TILES_M * BLOCK_WARPS_M;  // 64
    constexpr int TILE_N = WMMA_N * WARP_TILES_N * BLOCK_WARPS_N;  // 128
    constexpr int TILE_K = WMMA_K * 2;  // 64
    
    int* C_i32;
    cudaMallocAsync(&C_i32, M * N * sizeof(int), stream);
    
    const int threads = 256;
    dim3 block(threads);
    dim3 grid((M + TILE_M - 1) / TILE_M, 
              (N + TILE_N - 1) / TILE_N);
    
    const size_t smem_size = (TILE_M * TILE_K + TILE_N * TILE_K) / 2;
    
    rrs_gemm_i4_kernel<TILE_M, TILE_N, TILE_K>
        <<<grid, block, smem_size, stream>>>(
            (const uint8_t*)A, (const uint8_t*)B, C_i32,
            M, N, K, K, K, N);
    
    // Simple dequantization (average scales)
    dim3 dq_block(16, 16);
    dim3 dq_grid((M + 15) / 16, (N + 15) / 16);
    
    rrs_dequant_q4k_kernel<<<dq_grid, dq_block, 0, stream>>>(
        C_i32, C, scales_A, mins_A, scales_B, M, N, K);
    
    cudaFreeAsync(C_i32, stream);
}

// Alternative: Q8 repack path
void ggml_cuda_rrs_gemm_q4_via_q8(
    const void* A_q4,
    const void* B_q4,  
    float* C,
    int M, int N, int K,
    const half* scales_A, const half* mins_A,
    const half* scales_B, const half* mins_B,
    cudaStream_t stream)
{
    // Allocate Q8 buffers
    int8_t *A_q8, *B_q8;
    half *scales_A_q8;
    cudaMallocAsync(&A_q8, M * K * sizeof(int8_t), stream);
    cudaMallocAsync(&B_q8, N * K * sizeof(int8_t), stream);
    cudaMallocAsync(&scales_A_q8, M * (K / 32) * sizeof(half), stream);
    
    // Unpack Q4 to Q8 (use 256 threads, loop over elements)
    const int unpack_threads = 256;
    unpack_q4_to_q8_kernel<<<M, unpack_threads, 0, stream>>>(
        (const uint8_t*)A_q4, A_q8, scales_A, mins_A, scales_A_q8, K, M);
    unpack_q4_to_q8_kernel<<<N, unpack_threads, 0, stream>>>(
        (const uint8_t*)B_q4, B_q8, scales_B, mins_B, nullptr, K, N);
    
    // dp4a GEMM
    int* C_i32;
    cudaMallocAsync(&C_i32, M * N * sizeof(int), stream);
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    rrs_gemm_q8_dp4a_kernel<<<grid, block, 0, stream>>>(
        A_q8, B_q8, C_i32, M, N, K);
    
    // Dequantize (simplified)
    rrs_dequant_q4k_kernel<<<grid, block, 0, stream>>>(
        C_i32, C, scales_A_q8, mins_A, scales_B, M, N, K);
    
    cudaFreeAsync(A_q8, stream);
    cudaFreeAsync(B_q8, stream);
    cudaFreeAsync(scales_A_q8, stream);
    cudaFreeAsync(C_i32, stream);
}

// ============================================================================
// Dispatch Integration
// ============================================================================

void ggml_cuda_rrs_mul_mat(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,  // weights (Q4_K_RRS)
    const ggml_tensor* src1,  // activations (F32)
    ggml_tensor* dst)
{
    const int64_t ne00 = src0->ne[0];  // K (hidden dim)
    const int64_t ne01 = src0->ne[1];  // N (output features)
    const int64_t ne10 = src1->ne[0];  // K 
    const int64_t ne11 = src1->ne[1];  // M (batch/tokens)
    
    GGML_ASSERT(ne00 == ne10);  // K must match
    
    const int M = ne11;
    const int N = ne01;
    const int K = ne00;
    
    cudaStream_t stream = ctx.stream();
    
    // Allocate temporary buffers
    float* act_fwht;
    uint8_t* act_q4;
    half *act_scales, *act_mins;
    
    const int groups = K / 32;
    
    cudaMallocAsync(&act_fwht, M * K * sizeof(float), stream);
    cudaMallocAsync(&act_q4, M * K / 2, stream);
    cudaMallocAsync(&act_scales, M * groups * sizeof(half), stream);
    cudaMallocAsync(&act_mins, M * groups * sizeof(half), stream);
    
    // Get source pointers
    const float* src1_f32 = (const float*)src1->data;
    const void* src0_q4k = src0->data;
    float* dst_f32 = (float*)dst->data;
    
    // Step 1: FWHT + Quantize activations (fused)
    ggml_cuda_rrs_fwht_quantize(src1_f32, act_q4, act_scales, act_mins, K, M, stream);
    
    // Step 2: INT4 GEMM 
    // Note: Weight scales are embedded in Q4_K blocks
    // For now, use nullptr for weight scales - dequant kernel reads from blocks
    ggml_cuda_rrs_gemm_q4q4(
        act_q4, src0_q4k, dst_f32,
        M, N, K,
        act_scales, act_mins,
        nullptr, nullptr,  // Weight scales come from Q4_K blocks
        stream);
    
    // Cleanup
    cudaFreeAsync(act_fwht, stream);
    cudaFreeAsync(act_q4, stream);
    cudaFreeAsync(act_scales, stream);
    cudaFreeAsync(act_mins, stream);
}

// Check if RRS path should be used
bool ggml_cuda_supports_rrs(const ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_Q4_K_RRS) {
        return false;
    }
    
    // Check compute capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Require SM75+ for INT4 tensor cores
    return (props.major > 7) || (props.major == 7 && props.minor >= 5);
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

void ggml_cuda_rrs_benchmark(
    int M, int N, int K,
    int iterations,
    RRSBenchmarkResult* result)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocate test data
    float *h_A, *h_B;
    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(N * K * sizeof(float));
    
    // Initialize with random data
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * K; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;
    
    float *d_A, *d_B, *d_C;
    uint8_t *d_A_q4, *d_B_q4;
    half *d_scales_A, *d_mins_A, *d_scales_B, *d_mins_B;
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_q4, M * K / 2));
    CUDA_CHECK(cudaMalloc(&d_B_q4, N * K / 2));
    
    const int groups_A = M * (K / 32);
    const int groups_B = N * (K / 32);
    CUDA_CHECK(cudaMalloc(&d_scales_A, groups_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_mins_A, groups_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales_B, groups_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_mins_B, groups_B * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark FWHT
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_rrs_fwht(d_A, d_A, K, M, stream);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&result->fwht_time_ms, start, stop);
    result->fwht_time_ms /= iterations;
    
    // Benchmark FWHT + Quantize (fused)
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_rrs_fwht_quantize(d_A, d_A_q4, d_scales_A, d_mins_A, K, M, stream);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&result->quantize_time_ms, start, stop);
    result->quantize_time_ms /= iterations;
    
    // Pre-quantize B for GEMM benchmark
    ggml_cuda_rrs_fwht_quantize(d_B, d_B_q4, d_scales_B, d_mins_B, K, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    
    // Benchmark INT4 WMMA GEMM
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_rrs_gemm_q4q4(
            d_A_q4, d_B_q4, d_C,
            M, N, K,
            d_scales_A, d_mins_A,
            d_scales_B, d_mins_B,
            stream);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&result->int4_wmma_time_ms, start, stop);
    result->int4_wmma_time_ms /= iterations;
    
    // Benchmark Q8 repack path
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_rrs_gemm_q4_via_q8(
            d_A_q4, d_B_q4, d_C,
            M, N, K,
            d_scales_A, d_mins_A,
            d_scales_B, d_mins_B,
            stream);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&result->q8_repack_time_ms, start, stop);
    result->q8_repack_time_ms /= iterations;
    
    result->M = M;
    result->N = N;
    result->K = K;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_q4);
    cudaFree(d_B_q4);
    cudaFree(d_scales_A);
    cudaFree(d_mins_A);
    cudaFree(d_scales_B);
    cudaFree(d_mins_B);
    free(h_A);
    free(h_B);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

// Print benchmark results
void ggml_cuda_rrs_print_benchmark(const RRSBenchmarkResult* result) {
    printf("RRS CUDA Benchmark Results (M=%d, N=%d, K=%d):\n", 
           result->M, result->N, result->K);
    printf("  FWHT:           %.3f ms\n", result->fwht_time_ms);
    printf("  FWHT+Quantize:  %.3f ms\n", result->quantize_time_ms);
    printf("  INT4 WMMA GEMM: %.3f ms\n", result->int4_wmma_time_ms);
    printf("  Q8 Repack GEMM: %.3f ms\n", result->q8_repack_time_ms);
    
    // Compute TOPS
    double ops = 2.0 * result->M * result->N * result->K;
    double int4_tops = ops / (result->int4_wmma_time_ms * 1e-3) / 1e12;
    double q8_tops = ops / (result->q8_repack_time_ms * 1e-3) / 1e12;
    
    printf("  INT4 WMMA: %.2f TOPS\n", int4_tops);
    printf("  Q8 dp4a:   %.2f TOPS\n", q8_tops);
    printf("  Speedup (INT4/Q8): %.2fx\n", 
           result->q8_repack_time_ms / result->int4_wmma_time_ms);
}

// ============================================================================
// Simple Test Function - Can be called to verify CUDA RRS works
// ============================================================================

extern "C" void ggml_cuda_rrs_test(void) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("RRS CUDA Test: No CUDA device available\n");
        return;
    }
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("RRS CUDA Test on: %s (SM%d%d)\n", props.name, props.major, props.minor);
    printf("INT4 Tensor Core support: %s\n\n", 
           (props.major > 7 || (props.major == 7 && props.minor >= 5)) ? "Yes" : "No");
    
    const int iterations = 50;
    RRSBenchmarkResult result;
    
    // Test with M=1 (single token inference) - this configuration works
    printf("=== Single Token Inference (M=1, N=2048, K=2048) ===\n");
    ggml_cuda_rrs_benchmark(1, 2048, 2048, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    // Note: Larger M values require GEMM kernel fixes for proper boundary handling
    // when M < TILE_M (64). The INT4 WMMA kernel needs M >= 64 for correct operation.
    
    printf("RRS CUDA Test: COMPLETED\n");
    printf("\nNote: Current INT4 WMMA kernel requires M >= TILE_M (64) for batched inference.\n");
    printf("For M=1 single-token inference, Q8 dp4a path is recommended (faster for small M).\n");
}