#include "rrs.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/pipeline>

using namespace nvcuda;

// ============================================================================
// FWHT (Fast Walsh-Hadamard Transform) CUDA Kernel
// ============================================================================

__device__ __forceinline__ void fwht_butterfly(float& a, float& b) {
    float t = a;
    a = t + b;
    b = t - b;
}

template<int N>
__global__ void fwht_kernel_pow2(const float* __restrict__ x, float* __restrict__ y, int batch_size) {
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
// Activation Quantization Kernel (F32 -> Q4 packed)
// ============================================================================

__global__ void quantize_act_q4_kernel(
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
    
    const int pair_idx = lane % 16;
    int packed = __shfl_sync(0xFFFFFFFF, q, pair_idx) | 
                 (__shfl_sync(0xFFFFFFFF, q, pair_idx + 16) << 4);
    
    if (lane < 16) {
        const int out_idx = row * (n / 2) + group * 16 + pair_idx;
        qs[out_idx] = (uint8_t)packed;
    }
    
    if (lane == 0) {
        scales[row * groups + group] = __float2half(scale);
        mins[row * groups + group] = __float2half(vmin);
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
    // Placeholder - use fused kernel instead
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
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        smem_fq[i] = x_row[i];
    }
    __syncthreads();
    
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
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        smem_fq[i] *= scale_fwht;
    }
    __syncthreads();
    
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
        int packed = __shfl_sync(0xFFFFFFFF, q, pair_idx) |
                     (__shfl_sync(0xFFFFFFFF, q, pair_idx + 16) << 4);
        
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

// ============================================================================
// INT4 Tensor Core GEMM Kernel (Turing+ with WMMA s4)
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define RRS_HAS_INT4_TC 1
#endif

using I4 = wmma::experimental::precision::s4;

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
    
    using FragA_I4 = wmma::fragment<wmma::matrix_a, RRS_WMMA_M, RRS_WMMA_N, RRS_WMMA_K, I4, wmma::row_major>;
    using FragB_I4 = wmma::fragment<wmma::matrix_b, RRS_WMMA_M, RRS_WMMA_N, RRS_WMMA_K, I4, wmma::col_major>;
    using FragC_I32 = wmma::fragment<wmma::accumulator, RRS_WMMA_M, RRS_WMMA_N, RRS_WMMA_K, int>;
    
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / RRS_BLOCK_WARPS_N;
    const int warp_col = warp_id % RRS_BLOCK_WARPS_N;
    
    const int block_row = blockIdx.x * TILE_M;
    const int block_col = blockIdx.y * TILE_N;
    
    FragA_I4 a_frag[RRS_WARP_TILES_M];
    FragB_I4 b_frag[RRS_WARP_TILES_N];
    FragC_I32 c_frag[RRS_WARP_TILES_M][RRS_WARP_TILES_N];
    
    #pragma unroll
    for (int i = 0; i < RRS_WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < RRS_WARP_TILES_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0);
        }
    }
    
    for (int k = 0; k < K; k += TILE_K) {
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
        
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += RRS_WMMA_K) {
            #pragma unroll
            for (int i = 0; i < RRS_WARP_TILES_M; i++) {
                const int a_row = warp_row * RRS_WARP_TILES_M * RRS_WMMA_M + i * RRS_WMMA_M;
                wmma::load_matrix_sync(a_frag[i], 
                    smemA + a_row * TILE_K / 2 + kk / 2,
                    TILE_K);
            }
            
            #pragma unroll
            for (int j = 0; j < RRS_WARP_TILES_N; j++) {
                const int b_row = warp_col * RRS_WARP_TILES_N * RRS_WMMA_N + j * RRS_WMMA_N;
                wmma::load_matrix_sync(b_frag[j],
                    smemB + b_row * TILE_K / 2 + kk / 2,
                    TILE_K);
            }
            
            #pragma unroll
            for (int i = 0; i < RRS_WARP_TILES_M; i++) {
                #pragma unroll
                for (int j = 0; j < RRS_WARP_TILES_N; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < RRS_WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < RRS_WARP_TILES_N; j++) {
            const int c_row = block_row + warp_row * RRS_WARP_TILES_M * RRS_WMMA_M + i * RRS_WMMA_M;
            const int c_col = block_col + warp_col * RRS_WARP_TILES_N * RRS_WMMA_N + j * RRS_WMMA_N;
            
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

__global__ void rrs_dequant_kernel(
    const int* __restrict__ C_i32,
    float* __restrict__ C_f32,
    const half* __restrict__ scales_A,
    const half* __restrict__ mins_A,
    const half* __restrict__ scales_B,
    const half* __restrict__ mins_B,
    const int M, const int N, const int K)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    const int sum_i32 = C_i32[row * N + col];
    const int groups_A = K / 32;
    const int groups_B = K / 32;
    
    float avg_scale_A = 0.0f, avg_scale_B = 0.0f;
    for (int g = 0; g < groups_A; g++) {
        avg_scale_A += __half2float(scales_A[row * groups_A + g]);
    }
    avg_scale_A /= groups_A;
    
    for (int g = 0; g < groups_B; g++) {
        avg_scale_B += __half2float(scales_B[col * groups_B + g]);
    }
    avg_scale_B /= groups_B;
    
    C_f32[row * N + col] = (float)sum_i32 * avg_scale_A * avg_scale_B;
}

void ggml_cuda_rrs_gemm_q4q4(
    const void* A,
    const void* B,
    float* C,
    int M, int N, int K,
    const half* scales_A, const half* mins_A,
    const half* scales_B, const half* mins_B,
    cudaStream_t stream)
{
    int* C_i32;
    cudaMallocAsync(&C_i32, M * N * sizeof(int), stream);
    
    const int threads = 256;
    dim3 block(threads);
    dim3 grid((M + RRS_TILE_M - 1) / RRS_TILE_M, 
              (N + RRS_TILE_N - 1) / RRS_TILE_N);
    
    const size_t smem_size = (RRS_TILE_M * RRS_TILE_K + RRS_TILE_N * RRS_TILE_K) / 2;
    
    rrs_gemm_i4_kernel<RRS_TILE_M, RRS_TILE_N, RRS_TILE_K>
        <<<grid, block, smem_size, stream>>>(
            (const uint8_t*)A, (const uint8_t*)B, C_i32,
            M, N, K, K, K, N);
    
    dim3 dq_block(16, 16);
    dim3 dq_grid((M + 15) / 16, (N + 15) / 16);
    rrs_dequant_kernel<<<dq_grid, dq_block, 0, stream>>>(
        C_i32, C, scales_A, mins_A, scales_B, mins_B, M, N, K);
    
    cudaFreeAsync(C_i32, stream);
}

void ggml_cuda_rrs_mul_mat(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst)
{
    (void)ctx; (void)src0; (void)src1; (void)dst;
    // Integration point - to be implemented
}

