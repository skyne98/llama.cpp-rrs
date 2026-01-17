#include "rrs.cuh"
#include "common.cuh"

// Debug flag - set to 1 to enable debug output
#define RRS_DEBUG 1
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace nvcuda;

#if RRS_DEBUG
extern "C" void ggml_quantize_row_q4_K_rrs_act(const float * x, void * y, int64_t k);

static inline void rrs_get_scale_min_k4_host(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

static inline int rrs_decode_cpu_q4k_value(const block_q4_K* b, int idx) {
    const int chunk = idx / 64;
    const int offset = idx % 64;
    const int byte = chunk * 32 + (offset & 31);
    const uint8_t packed = b->qs[byte];
    return (offset < 32) ? (packed & 0x0F) : (packed >> 4);
}

static inline int rrs_decode_cuda_q4_value(const uint8_t* q, int idx) {
    const int group = idx / 32;
    const int idx_in_group = idx % 32;
    const int byte = group * 16 + (idx_in_group & 15);
    const uint8_t packed = q[byte];
    const int nibble = (idx_in_group < 16) ? (packed & 0x0F) : (packed >> 4);
    const int q_signed = (nibble >= 8) ? (nibble - 16) : nibble;
    return q_signed + 8;
}

static void rrs_debug_compare_act_quant(
    const float* h_act_f32,
    int k,
    const uint8_t* h_act_q4,
    const half* h_act_scales,
    const half* h_act_mins) {

    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS_ACT, k);
    uint8_t* cpu_row = (uint8_t*)malloc(row_size);
    if (!cpu_row) {
        fprintf(stderr, "[RRS CUDA] Debug compare: failed to allocate CPU row buffer\n");
        return;
    }

    ggml_quantize_row_q4_K_rrs_act(h_act_f32, cpu_row, k);

    const block_q4_K* b = (const block_q4_K*)cpu_row;
    uint8_t sc0 = 0, mn0 = 0;
    rrs_get_scale_min_k4_host(0, b->scales, &sc0, &mn0);

    const float cpu_scale0 = GGML_FP16_TO_FP32(*(const uint16_t *)&b->dm.x) * sc0;
    const float cpu_min0 = GGML_FP16_TO_FP32(*(const uint16_t *)&b->dm.y) * mn0;
    const float cpu_offset0 = 8.0f * cpu_scale0 - cpu_min0;
    const float cuda_scale0 = h_act_scales ? __half2float(h_act_scales[0]) : 0.0f;
    const float cuda_offset0 = h_act_mins ? __half2float(h_act_mins[0]) : 0.0f;

    fprintf(stderr,
        "[RRS CUDA] ACT scale/min g0 CPU: scale=%f min=%f offset=%f | CUDA: scale=%f offset=%f\n",
        cpu_scale0, cpu_min0, cpu_offset0, cuda_scale0, cuda_offset0);

    fprintf(stderr, "[RRS CUDA] ACT q4 first 16 CPU:");
    for (int i = 0; i < 16; i++) {
        fprintf(stderr, " %d", rrs_decode_cpu_q4k_value(b, i));
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "[RRS CUDA] ACT q4 first 16 CUDA:");
    for (int i = 0; i < 16; i++) {
        fprintf(stderr, " %d", rrs_decode_cuda_q4_value(h_act_q4, i));
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "[RRS CUDA] ACT dequant first 8 CPU:");
    for (int i = 0; i < 8; i++) {
        const int q = rrs_decode_cpu_q4k_value(b, i);
        const float v = cpu_scale0 * (float)q - cpu_min0;
        fprintf(stderr, " %f", v);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "[RRS CUDA] ACT dequant first 8 CUDA:");
    for (int i = 0; i < 8; i++) {
        const int q = rrs_decode_cuda_q4_value(h_act_q4, i);
        const float v = cuda_scale0 * (float)(q - 8) + cuda_offset0;
        fprintf(stderr, " %f", v);
    }
    fprintf(stderr, "\n");

    free(cpu_row);
}
#endif

// ============================================================================
// SmemTensor helpers (from QuadMul)
// ============================================================================

#define CUDA_DEVICE_INLINE __device__ __forceinline__

template <typename T, int ShapeX, int ShapeY, int ShapeZ>
class SmemTensor3D {
public:
    T* startPtr;
    T* endPtr;
    
    CUDA_DEVICE_INLINE SmemTensor3D(void* ptr) 
        : startPtr(reinterpret_cast<T*>(ptr)), 
          endPtr(reinterpret_cast<T*>(ptr) + ShapeX * ShapeY * ShapeZ) {}
    
    CUDA_DEVICE_INLINE T* get_ptr(int x, int y, int z) {
        return &startPtr[x * ShapeY * ShapeZ + y * ShapeZ + z];
    }
};

template <typename T>
class GMemTensor2D {
private:
    T* startPtr;
    int shapeY;
public:
    CUDA_DEVICE_INLINE GMemTensor2D(T* ptr, int x, int y) 
        : startPtr(ptr), shapeY(y) {}
    
    CUDA_DEVICE_INLINE T* get_ptr(int x, int y) {
        return &startPtr[x * shapeY + y];
    }
};

// ============================================================================
// Q4_K Format Constants and Scale Unpacking
// ============================================================================

// Q4_K: 256 elements per super-block, 8 groups of 32 elements
// 12 bytes of packed 6-bit scales/mins
// d (fp16): super-scale for dequantizing per-group scales
// dmin (fp16): super-scale for dequantizing per-group mins

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
// Output: packed SIGNED INT4 nibbles + per-group scale/offset (asymmetric, matching Q4_K)
// WMMA INT4 requires signed values in [-8, 7] range
__global__ void quantize_act_q4_simple_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ qs,
    half* __restrict__ scales,
    half* __restrict__ mins,  // Stores offset for dequant: 8*scale - min
    int n,
    int batch_size)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* x_row = x + row * n;
    const int groups = n / 32;
    const int blocks_per_row = n / 256;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    __shared__ float s_scales[8];
    __shared__ float s_mins[8];
    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ float s_inv_scale;
    __shared__ float s_inv_min;

    for (int block = 0; block < blocks_per_row; ++block) {
        if (warp_id < 8) {
            const int base = block * 256 + warp_id * 32;
            float val = x_row[base + lane];

            float vmin = val;
            float vmax = val;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                vmin = fminf(vmin, __shfl_xor_sync(0xFFFFFFFF, vmin, mask));
                vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, mask));
            }

            if (vmin > 0) vmin = 0;
            const float gscale = (vmax - vmin) / 15.0f;
            const float gmin_pos = -vmin;

            if (lane == 0) {
                s_scales[warp_id] = gscale;
                s_mins[warp_id] = gmin_pos;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            float max_scale = s_scales[0];
            float max_min = s_mins[0];
            #pragma unroll
            for (int i = 1; i < 8; i++) {
                if (s_scales[i] > max_scale) max_scale = s_scales[i];
                if (s_mins[i] > max_min) max_min = s_mins[i];
            }

            s_d = (max_scale > 0.0f) ? (max_scale / 63.0f) : 0.0f;
            s_dmin = (max_min > 0.0f) ? (max_min / 63.0f) : 0.0f;
            s_inv_scale = (max_scale > 0.0f) ? (63.0f / max_scale) : 0.0f;
            s_inv_min = (max_min > 0.0f) ? (63.0f / max_min) : 0.0f;
        }

        __syncthreads();

        if (warp_id < 8) {
            const float gscale = s_scales[warp_id];
            const float gmin_pos = s_mins[warp_id];

            int ls = (int)(s_inv_scale * gscale + 0.5f);
            int lm = (int)(s_inv_min * gmin_pos + 0.5f);
            if (ls < 0) ls = 0; if (ls > 63) ls = 63;
            if (lm < 0) lm = 0; if (lm > 63) lm = 63;

            const float d = s_d;
            const float dmin = s_dmin;
            const float d_group = d * (float)ls;
            const float dm_group = dmin * (float)lm;
            const float inv_d = (d_group > 1e-10f) ? (1.0f / d_group) : 0.0f;

            const int base = block * 256 + warp_id * 32;
            const float val = x_row[base + lane];
            
            // Asymmetric quant: q = (val + min) / scale
            int q = (int)((val + dm_group) * inv_d + 0.5f);
            if (q < 0) q = 0;
            if (q > 15) q = 15;

            // Map 0..15 to -8..7 for signed storage
            const int q_signed = q - 8;
            const int q_4bit = q_signed & 0xF;

            const int pair_idx = lane & 15;
            const int lo = __shfl_sync(0xFFFFFFFF, q_4bit, pair_idx);
            const int hi = __shfl_sync(0xFFFFFFFF, q_4bit, pair_idx + 16);
            const int packed = lo | (hi << 4);

            const int group = block * 8 + warp_id;
            if (lane < 16) {
                qs[row * (n / 2) + group * 16 + pair_idx] = (uint8_t)packed;
            }

            if (lane == 0) {
                const float scale = d_group;
                const float offset = 8.0f * scale - dm_group;
                scales[row * groups + group] = __float2half(scale);
                mins[row * groups + group] = __float2half(offset);
            }
        }

        __syncthreads();
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
    
    // Quantize per-256 block using Q4_K layout (matches CPU quantize_row_q4_K_fast)
    // WMMA INT4 requires signed values, so we convert unsigned [0,15] to signed [-8,7]
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int blocks_per_row = N / 256;

    __shared__ float s_scales[8];
    __shared__ float s_mins[8];
    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ float s_inv_scale;
    __shared__ float s_inv_min;

    for (int block = 0; block < blocks_per_row; ++block) {
        if (warp_id < 8) {
            const int base = block * 256 + warp_id * 32;
            float val = smem_fq[base + lane];

            float vmin = val;
            float vmax = val;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                vmin = fminf(vmin, __shfl_xor_sync(0xFFFFFFFF, vmin, mask));
                vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, mask));
            }

            if (vmin > 0) vmin = 0;
            const float gscale = (vmax - vmin) / 15.0f;
            const float gmin_pos = -vmin;

            if (lane == 0) {
                s_scales[warp_id] = gscale;
                s_mins[warp_id] = gmin_pos;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            float max_scale = s_scales[0];
            float max_min = s_mins[0];
            #pragma unroll
            for (int i = 1; i < 8; i++) {
                if (s_scales[i] > max_scale) max_scale = s_scales[i];
                if (s_mins[i] > max_min) max_min = s_mins[i];
            }

            s_d = (max_scale > 0.0f) ? (max_scale / 63.0f) : 0.0f;
            s_dmin = (max_min > 0.0f) ? (max_min / 63.0f) : 0.0f;
            s_inv_scale = (max_scale > 0.0f) ? (63.0f / max_scale) : 0.0f;
            s_inv_min = (max_min > 0.0f) ? (63.0f / max_min) : 0.0f;
        }

        __syncthreads();

        if (warp_id < 8) {
            const float gscale = s_scales[warp_id];
            const float gmin_pos = s_mins[warp_id];

            int ls = (int)(s_inv_scale * gscale + 0.5f);
            int lm = (int)(s_inv_min * gmin_pos + 0.5f);
            if (ls < 0) ls = 0; if (ls > 63) ls = 63;
            if (lm < 0) lm = 0; if (lm > 63) lm = 63;

            const float d = s_d;
            const float dmin = s_dmin;
            const float d_group = d * (float)ls;
            const float dm_group = dmin * (float)lm;
            const float inv_d = (d_group > 1e-10f) ? (1.0f / d_group) : 0.0f;

            const int base = block * 256 + warp_id * 32;
            const float val = smem_fq[base + lane];
            
            // Asymmetric quant: q = (val + min) / scale
            int q = (int)((val + dm_group) * inv_d + 0.5f);
            if (q < 0) q = 0;
            if (q > 15) q = 15;

            // Map 0..15 to -8..7 for signed storage
            const int q_signed = q - 8;
            const int q_4bit = q_signed & 0xF;

            const int pair_idx = lane & 15;
            const int lo = __shfl_sync(0xFFFFFFFF, q_4bit, pair_idx);
            const int hi = __shfl_sync(0xFFFFFFFF, q_4bit, pair_idx + 16);
            const int packed = lo | (hi << 4);

            const int group = block * 8 + warp_id;
            if (lane < 16) {
                qs[batch_idx * (N / 2) + group * 16 + pair_idx] = (uint8_t)packed;
            }

            if (lane == 0) {
                const float scale = d_group;
                const float offset = 8.0f * scale - dm_group;
                scales[batch_idx * groups + group] = __float2half(scale);
                mins[batch_idx * groups + group] = __float2half(offset);
            }
        }

        __syncthreads();
    }
}

__global__ void fwht_quantize_kernel_chunked(
    const float* __restrict__ x,
    uint8_t* __restrict__ qs,
    half* __restrict__ scales,
    half* __restrict__ mins,
    int n,
    int batch_size)
{
    extern __shared__ float smem_chunk[];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x_row = x + batch_idx * n;
    const int groups_total = n / 32;
    
    const int step = n & -n;
    const int num_chunks = n / step;
    const float scale_fwht = rsqrtf((float)step);
    
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int blocks_per_chunk = step / 256;

    __shared__ float s_scales[8];
    __shared__ float s_mins[8];
    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ float s_inv_scale;
    __shared__ float s_inv_min;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int base_n = chunk * step;
        
        // Load into shared memory
        for (int i = threadIdx.x; i < step; i += blockDim.x) {
            smem_chunk[i] = x_row[base_n + i];
        }
        __syncthreads();
        
        // FWHT in-place for chunk
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
        
        // Apply scale
        for (int i = threadIdx.x; i < step; i += blockDim.x) {
            smem_chunk[i] *= scale_fwht;
        }
        __syncthreads();

        // Quantize chunk
        for (int block = 0; block < blocks_per_chunk; ++block) {
            if (warp_id < 8) {
                const int base_s = block * 256 + warp_id * 32;
                float val = smem_chunk[base_s + lane];

                float vmin = val;
                float vmax = val;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    vmin = fminf(vmin, __shfl_xor_sync(0xFFFFFFFF, vmin, mask));
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, mask));
                }

                if (vmin > 0) vmin = 0;
                const float gscale = (vmax - vmin) / 15.0f;
                const float gmin_pos = -vmin;

                if (lane == 0) {
                    s_scales[warp_id] = gscale;
                    s_mins[warp_id] = gmin_pos;
                }
            }

            __syncthreads();

            if (threadIdx.x == 0) {
                float max_scale = s_scales[0];
                float max_min = s_mins[0];
                #pragma unroll
                for (int i = 1; i < 8; i++) {
                    if (s_scales[i] > max_scale) max_scale = s_scales[i];
                    if (s_mins[i] > max_min) max_min = s_mins[i];
                }

                s_d = (max_scale > 0.0f) ? (max_scale / 63.0f) : 0.0f;
                s_dmin = (max_min > 0.0f) ? (max_min / 63.0f) : 0.0f;
                s_inv_scale = (max_scale > 0.0f) ? (63.0f / max_scale) : 0.0f;
                s_inv_min = (max_min > 0.0f) ? (63.0f / max_min) : 0.0f;
            }

            __syncthreads();

            if (warp_id < 8) {
                const float gscale = s_scales[warp_id];
                const float gmin_pos = s_mins[warp_id];

                int ls = (int)(s_inv_scale * gscale + 0.5f);
                int lm = (int)(s_inv_min * gmin_pos + 0.5f);
                if (ls < 0) ls = 0; if (ls > 63) ls = 63;
                if (lm < 0) lm = 0; if (lm > 63) lm = 63;

                const float d = s_d;
                const float dmin = s_dmin;
                const float d_group = d * (float)ls;
                const float dm_group = dmin * (float)lm;
                const float inv_d = (d_group > 1e-10f) ? (1.0f / d_group) : 0.0f;

                const int base_s = block * 256 + warp_id * 32;
                const float val = smem_chunk[base_s + lane];
                int q = (int)((val + dm_group) * inv_d + 0.5f);
                if (q < 0) q = 0;
                if (q > 15) q = 15;

                const int q_signed = q - 8;
                const int q_4bit = q_signed & 0xF;

                const int pair_idx = lane & 15;
                const int lo = __shfl_sync(0xFFFFFFFF, q_4bit, pair_idx);
                const int hi = __shfl_sync(0xFFFFFFFF, q_4bit, pair_idx + 16);
                const int packed = lo | (hi << 4);

                const int group_idx = (chunk * blocks_per_chunk + block) * 8 + warp_id;
                if (lane < 16) {
                    qs[batch_idx * (n / 2) + group_idx * 16 + pair_idx] = (uint8_t)packed;
                }

                if (lane == 0) {
                    const float scale = d_group;
                    const float offset = 8.0f * scale - dm_group;
                    scales[batch_idx * groups_total + group_idx] = __float2half(scale);
                    mins[batch_idx * groups_total + group_idx] = __float2half(offset);
                }
            }
            __syncthreads();
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
    if ((n & (n - 1)) == 0) {
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
            case 8192:
                fwht_quantize_kernel<8192><<<batch_size, 512, smem_size, stream>>>(
                    x, (uint8_t*)y, scales, mins, batch_size);
                break;
            default:
                fprintf(stderr, "[RRS CUDA] Error: Unsupported FWHT size N=%d\n", n);
                break;
        }
    } else {
        const int step = n & -n;
        const size_t smem_size = step * sizeof(float);
        fwht_quantize_kernel_chunked<<<batch_size, min(256, step), smem_size, stream>>>(
            x, (uint8_t*)y, scales, mins, n, batch_size);
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

#ifdef RRS_HAS_INT4_TC
using I4 = wmma::experimental::precision::s4;
#endif

// ============================================================================
// QuadMul-style INT4 WMMA GEMM with 2-Stage Async Pipeline
// ============================================================================

#define WARP_SIZE 32
#define ALIGN_SIZE 32  // Async copy alignment in bytes

// Configuration template (following QuadMul IGemmConfig)
template <int BlockRowWarps, int BlockColWarps, int WarpRowTiles, int WarpColTiles, 
          int PatchM, int PatchN, int ChunkK, int NumStages,
          int kWMMA_M, int kWMMA_N, int kWMMA_K>
struct RRSGemmConfig {
    static constexpr int kBlockRowWarps = BlockRowWarps;
    static constexpr int kBlockColWarps = BlockColWarps;
    static constexpr int kWarpRowTiles = WarpRowTiles;
    static constexpr int kWarpColTiles = WarpColTiles;
    static constexpr int kPatchM = PatchM;
    static constexpr int kPatchN = PatchN;
    static constexpr int kChunkK = ChunkK;
    static constexpr int kNumStages = NumStages;
    
    // Derived
    static constexpr int kBlockRowTiles = kWarpRowTiles * kBlockRowWarps;
    static constexpr int kBlockColTiles = kWarpColTiles * kBlockColWarps;
    static constexpr int kTileSizeM = kWMMA_M * kBlockRowTiles;
    static constexpr int kTileSizeN = kWMMA_N * kBlockColTiles;
    static constexpr int kTileSizeK = kWMMA_K * kChunkK;
    
    static constexpr int WMMA_M = kWMMA_M;
    static constexpr int WMMA_N = kWMMA_N;
    static constexpr int WMMA_K = kWMMA_K;
};

// Default configs for different M sizes (tuned for SM86/RTX 3090)
// Thread count = WARP_SIZE * (BlockRowWarps/PatchM) * (BlockColWarps/PatchN)
// Must be <= 256 for __launch_bounds__(256)
//
// Config params: <BlockRowWarps, BlockColWarps, WarpRowTiles, WarpColTiles, 
//                 PatchM, PatchN, ChunkK, NumStages, WMMA_M, WMMA_N, WMMA_K>
//
// TileSizeM = WMMA_M * WarpRowTiles * BlockRowWarps
// TileSizeN = WMMA_N * WarpColTiles * BlockColWarps
// TileSizeK = WMMA_K * ChunkK

// Small M (single token decode): 32 * 2 * 4 = 256 threads
// TileM=32, TileN=64, TileK=64
using ConfigM1   = RRSGemmConfig<2, 4, 2, 2, 1, 1, 2, 2, 8, 8, 32>;

// Medium M (small batch): 32 * 2 * 4 = 256 threads  
// TileM=32, TileN=128, TileK=64 - wider N for better occupancy
using ConfigM32  = RRSGemmConfig<2, 4, 2, 4, 1, 1, 2, 2, 8, 8, 32>;

// Large M (prefill): 32 * 4 * 2 = 256 threads
// TileM=128, TileN=64, TileK=64 - taller M for prefill
using ConfigM256 = RRSGemmConfig<4, 2, 4, 4, 1, 1, 2, 2, 8, 8, 32>;

// Very large M (big prefill): 32 * 4 * 2 = 256 threads, with PatchM=2 for register tiling
// TileM=128, TileN=64, TileK=128 - larger K chunk for better compute/memory ratio
using ConfigM512 = RRSGemmConfig<4, 2, 4, 4, 2, 1, 4, 2, 8, 8, 32>;

template<typename Config>
__global__ void __launch_bounds__(256) rrs_gemm_i4_async_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    int32_t* __restrict__ C,
    const int M, const int N, const int K)
{
#ifdef RRS_HAS_INT4_TC
    extern __shared__ int8_t shared_memory[];
    
    // Fragment types for INT4 WMMA
    using FragA = wmma::fragment<wmma::matrix_a, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, I4, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, I4, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, int32_t>;
    
    // Set up multi-stage shared memory (double-buffered)
    // A: [NumStages][TileSizeM][TileSizeK/2] (packed int4)
    // B: [NumStages][TileSizeN][TileSizeK/2] (packed int4)
    SmemTensor3D<uint8_t, Config::kNumStages, Config::kTileSizeM, Config::kTileSizeK / 2> 
        smemA(shared_memory);
    SmemTensor3D<uint8_t, Config::kNumStages, Config::kTileSizeN, Config::kTileSizeK / 2> 
        smemB(smemA.endPtr);
    
    // Global memory views
    GMemTensor2D<uint8_t> gmemA((uint8_t*)A, M, K / 2);
    GMemTensor2D<uint8_t> gmemB((uint8_t*)B, N, K / 2);
    GMemTensor2D<int32_t> gmemC(C, M, N);
    
    // Warp/thread indexing
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / (Config::kBlockColWarps / Config::kPatchN);
    const int warp_col = warp_id % (Config::kBlockColWarps / Config::kPatchN);
    
    // Block position
    const int block_row_start = blockIdx.x * Config::kTileSizeM;
    const int block_col_start = blockIdx.y * Config::kTileSizeN;
    
    // Accumulator fragments
    FragA a_frag[Config::kPatchM][Config::kWarpRowTiles];
    FragB b_frag[Config::kPatchN][Config::kWarpColTiles];
    FragC c_frag[Config::kPatchM][Config::kPatchN][Config::kWarpRowTiles][Config::kWarpColTiles];
    
    // Initialize accumulators
    #pragma unroll
    for (int pm = 0; pm < Config::kPatchM; pm++) {
        #pragma unroll
        for (int pn = 0; pn < Config::kPatchN; pn++) {
            #pragma unroll
            for (int i = 0; i < Config::kWarpRowTiles; i++) {
                #pragma unroll
                for (int j = 0; j < Config::kWarpColTiles; j++) {
                    wmma::fill_fragment(c_frag[pm][pn][i][j], 0);
                }
            }
        }
    }
    
    // Lambda: async load A tile into shared memory stage
    // Memory layout: smemA[stage][row][col_packed] where col_packed = TileSizeK/2 bytes
    // Each thread loads 16 bytes (32 int4 values) per iteration
    auto load_A_tile = [&](int stage, int k_offset) {
        constexpr int BYTES_PER_LOAD = 16;  // 16 bytes = 32 int4 values
        constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD * 2;  // 32 int4 elements
        constexpr int total_bytes = Config::kTileSizeM * Config::kTileSizeK / 2;  // packed int4
        constexpr int num_loads = total_bytes / BYTES_PER_LOAD;
        
        for (int i = threadIdx.x; i < num_loads; i += blockDim.x) {
            // Linear index in bytes within the tile
            const int byte_offset = i * BYTES_PER_LOAD;
            const int row = byte_offset / (Config::kTileSizeK / 2);
            const int col_bytes = byte_offset % (Config::kTileSizeK / 2);
            const int global_row = block_row_start + row;
            const int global_col = k_offset + col_bytes * 2;  // col in int4 elements
            
            if (global_row < M && global_col + ELEMENTS_PER_LOAD <= K) {
                uint8_t* shared_ptr = smemA.get_ptr(stage, row, col_bytes);
                uint8_t* global_ptr = gmemA.get_ptr(global_row, col_bytes + k_offset / 2);
                __pipeline_memcpy_async(shared_ptr, global_ptr, BYTES_PER_LOAD);
            } else {
                // Zero-fill for boundary handling
                uint8_t* shared_ptr = smemA.get_ptr(stage, row, col_bytes);
                for (int b = 0; b < BYTES_PER_LOAD; b++) {
                    shared_ptr[b] = 0;
                }
            }
        }
    };
    
    // Lambda: async load B tile into shared memory stage
    auto load_B_tile = [&](int stage, int k_offset) {
        constexpr int BYTES_PER_LOAD = 16;
        constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD * 2;
        constexpr int total_bytes = Config::kTileSizeN * Config::kTileSizeK / 2;
        constexpr int num_loads = total_bytes / BYTES_PER_LOAD;
        
        for (int i = threadIdx.x; i < num_loads; i += blockDim.x) {
            const int byte_offset = i * BYTES_PER_LOAD;
            const int row = byte_offset / (Config::kTileSizeK / 2);
            const int col_bytes = byte_offset % (Config::kTileSizeK / 2);
            const int global_row = block_col_start + row;
            const int global_col = k_offset + col_bytes * 2;
            
            if (global_row < N && global_col + ELEMENTS_PER_LOAD <= K) {
                uint8_t* shared_ptr = smemB.get_ptr(stage, row, col_bytes);
                uint8_t* global_ptr = gmemB.get_ptr(global_row, col_bytes + k_offset / 2);
                __pipeline_memcpy_async(shared_ptr, global_ptr, BYTES_PER_LOAD);
            } else {
                uint8_t* shared_ptr = smemB.get_ptr(stage, row, col_bytes);
                for (int b = 0; b < BYTES_PER_LOAD; b++) {
                    shared_ptr[b] = 0;
                }
            }
        }
    };
    
    // Shared memory for partial tile stores (reuse part of shared_memory after GEMM is done)
    // We need space for one WMMA fragment: 8x8 = 64 int32s = 256 bytes
    int32_t* smem_c = reinterpret_cast<int32_t*>(shared_memory);
    
    // Lambda: store C tile to global memory with partial tile handling
    auto store_C_tile = [&]() {
        #pragma unroll
        for (int pm = 0; pm < Config::kPatchM; pm++) {
            #pragma unroll
            for (int pn = 0; pn < Config::kPatchN; pn++) {
                #pragma unroll
                for (int i = 0; i < Config::kWarpRowTiles; i++) {
                    #pragma unroll
                    for (int j = 0; j < Config::kWarpColTiles; j++) {
                        const int row = block_row_start + 
                            ((warp_row * Config::kPatchM + pm) * Config::kWarpRowTiles + i) * Config::WMMA_M;
                        const int col = block_col_start + 
                            ((warp_col * Config::kPatchN + pn) * Config::kWarpColTiles + j) * Config::WMMA_N;
                        
                        // Skip if completely out of bounds
                        if (row >= M || col >= N) continue;
                        
                        // Check if full WMMA tile fits in bounds
                        if (row + Config::WMMA_M <= M && col + Config::WMMA_N <= N) {
                            // Fast path: full tile store
                            wmma::store_matrix_sync(gmemC.get_ptr(row, col), 
                                c_frag[pm][pn][i][j], N, wmma::mem_row_major);
                        } else {
                            // Slow path: partial tile - store to shared memory then scalar copy
                            // Only one warp at a time can use smem_c, so serialize
                            __syncthreads();
                            wmma::store_matrix_sync(smem_c, c_frag[pm][pn][i][j], 
                                Config::WMMA_N, wmma::mem_row_major);
                            __syncthreads();
                            
                            // Each thread in warp copies some elements
                            const int lane = threadIdx.x % WARP_SIZE;
                            for (int elem = lane; elem < Config::WMMA_M * Config::WMMA_N; elem += WARP_SIZE) {
                                const int local_row = elem / Config::WMMA_N;
                                const int local_col = elem % Config::WMMA_N;
                                const int global_row = row + local_row;
                                const int global_col = col + local_col;
                                
                                if (global_row < M && global_col < N) {
                                    gmemC.get_ptr(global_row, global_col)[0] = smem_c[elem];
                                }
                            }
                            __syncthreads();
                        }
                    }
                }
            }
        }
    };
    
    // ========== 2-Stage Async Pipeline ==========
    
    // Stage 0: Initial load
    load_A_tile(0, 0);
    load_B_tile(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    
    int current_stage = 0;
    
    for (int k = 0; k < K; k += Config::kTileSizeK) {
        // Start loading next stage if available (overlap with compute)
        if (k + Config::kTileSizeK < K) {
            const int next_stage = 1 - current_stage;
            load_A_tile(next_stage, k + Config::kTileSizeK);
            load_B_tile(next_stage, k + Config::kTileSizeK);
            __pipeline_commit();
        }
        
        // Compute using current stage
        #pragma unroll
        for (int kk = 0; kk < Config::kTileSizeK; kk += Config::WMMA_K) {
            // Load A fragments
            #pragma unroll
            for (int pm = 0; pm < Config::kPatchM; pm++) {
                #pragma unroll
                for (int i = 0; i < Config::kWarpRowTiles; i++) {
                    const int a_row = (warp_row * Config::kPatchM + pm) * Config::kWarpRowTiles * Config::WMMA_M 
                                      + i * Config::WMMA_M;
                    wmma::load_matrix_sync(a_frag[pm][i],
                        smemA.get_ptr(current_stage, a_row, kk / 2),
                        Config::kTileSizeK);
                }
            }
            
            // Load B fragments
            #pragma unroll
            for (int pn = 0; pn < Config::kPatchN; pn++) {
                #pragma unroll
                for (int j = 0; j < Config::kWarpColTiles; j++) {
                    const int b_row = (warp_col * Config::kPatchN + pn) * Config::kWarpColTiles * Config::WMMA_N 
                                      + j * Config::WMMA_N;
                    wmma::load_matrix_sync(b_frag[pn][j],
                        smemB.get_ptr(current_stage, b_row, kk / 2),
                        Config::kTileSizeK);
                }
            }
            
            // Matrix multiply-accumulate
            #pragma unroll
            for (int pm = 0; pm < Config::kPatchM; pm++) {
                #pragma unroll
                for (int pn = 0; pn < Config::kPatchN; pn++) {
                    #pragma unroll
                    for (int i = 0; i < Config::kWarpRowTiles; i++) {
                        #pragma unroll
                        for (int j = 0; j < Config::kWarpColTiles; j++) {
                            wmma::mma_sync(c_frag[pm][pn][i][j], 
                                          a_frag[pm][i], b_frag[pn][j], 
                                          c_frag[pm][pn][i][j]);
                        }
                    }
                }
            }
        }
        
        // Wait for next stage to finish loading
        __pipeline_wait_prior(0);
        __syncthreads();
        
        // Swap stages
        current_stage = 1 - current_stage;
    }
    
    // Store results
    store_C_tile();
    
#else
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

// ============================================================================
// Legacy kernel (kept for fallback/comparison)
// ============================================================================

// WMMA tile dimensions for INT4: 8x8x32
#ifdef RRS_HAS_INT4_TC
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 32;

// Block tiling configuration (legacy)
constexpr int BLOCK_WARPS_M = 4;
constexpr int BLOCK_WARPS_N = 4;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;
#endif

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void __launch_bounds__(256) rrs_gemm_i4_kernel_legacy(
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
// Kernel Launch Helpers
// ============================================================================

template<typename Config>
void launch_rrs_gemm_i4_async(
    const uint8_t* A, const uint8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + Config::kTileSizeM - 1) / Config::kTileSizeM,
              (N + Config::kTileSizeN - 1) / Config::kTileSizeN);
    dim3 block(WARP_SIZE * (Config::kBlockRowWarps / Config::kPatchM) * 
                           (Config::kBlockColWarps / Config::kPatchN));
    
    // Shared memory: 2 stages * (A tile + B tile)
    size_t smem_size = Config::kNumStages * 
        (Config::kTileSizeM * Config::kTileSizeK / 2 + 
         Config::kTileSizeN * Config::kTileSizeK / 2);
    
    rrs_gemm_i4_async_kernel<Config><<<grid, block, smem_size, stream>>>(A, B, C, M, N, K);
}

// Select config based on M dimension
inline void rrs_gemm_i4_dispatch(
    const uint8_t* A, const uint8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream)
{
    if (M <= 16) {
        launch_rrs_gemm_i4_async<ConfigM1>(A, B, C, M, N, K, stream);
    } else if (M <= 64) {
        launch_rrs_gemm_i4_async<ConfigM32>(A, B, C, M, N, K, stream);
    } else if (M <= 256) {
        launch_rrs_gemm_i4_async<ConfigM256>(A, B, C, M, N, K, stream);
    } else {
        launch_rrs_gemm_i4_async<ConfigM512>(A, B, C, M, N, K, stream);
    }
}

// ============================================================================
// Dequantization Kernel with Proper Q4_K Scale Handling
// ============================================================================

// Simple dequant kernel for benchmark - uses separate scales arrays (not Q4_K block format)
// If weight_scales is nullptr, uses a default scale (for Q4_K_RRS with embedded scales)
// Dequantization for signed INT4 GEMM result
// Both activations and weights use symmetric signed INT4 quantization
// Act: val_a = q_a * scale_a (symmetric, centered at 0)
// Wgt: val_w = q_w * scale_w + offset_w (converted from Q4_K asymmetric)
// 
// Dot product: sum(val_a * val_w) = sum(q_a * scale_a * (q_w * scale_w + offset_w))
//            = scale_a * scale_w * sum(q_a * q_w) + scale_a * offset_w * sum(q_a)
//            = scale_a * scale_w * C_i32 + scale_a * offset_w * sum_q_a
// 
// For simplicity, we approximate with average scales (per-group tracking would be more accurate)
__global__ void rrs_mmvq_q4_k_accurate_kernel(
    const uint8_t* __restrict__ A,      // [K/2] packed activations
    const void*    __restrict__ B,      // [N, K/256] block_q4_K weights
    float*         __restrict__ C,      // [N] output
    const half*    __restrict__ a_scales,
    const half*    __restrict__ a_mins,
    const int N, const int K)
{
    const int i = blockIdx.x; // row in B = output feature index
    if (i >= N) return;

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    const block_q4_K* blocks_w = (const block_q4_K*)B + i * (K / 256);
    float total_sum = 0.0f;

    for (int b = tid; b < K / 256; b += n_threads) {
        const block_q4_K& wb = blocks_w[b];
        const float d_w = __half2float(wb.dm.x);
        const float dmin_w = __half2float(wb.dm.y);

        uint8_t sc_w[8], mn_w[8];
        unpack_scales_mins_k4_cuda(wb.scales, sc_w, mn_w);

        for (int g = 0; g < 8; g++) {
            const float scale_w = d_w * (float)sc_w[g];
            const float offset_w = 8.0f * scale_w - dmin_w * (float)mn_w[g];

            const float scale_a = __half2float(a_scales[b * 8 + g]);
            const float offset_a = __half2float(a_mins[b * 8 + g]);

            const int p = g / 2;
            const uint8_t* qs_w = wb.qs + p * 32;
            const bool is_hi = (g % 2 != 0);
            const uint8_t* qs_a = A + (b * 256 + g * 32) / 2;

            int dot_i32 = 0;
            int sum_q_a = 0;
            int sum_q_w = 0;

            #pragma unroll
            for (int l = 0; l < 16; l++) {
                const uint8_t p_a = qs_a[l];
                const uint8_t p_w_lo = qs_w[l];
                const uint8_t p_w_hi = qs_w[l + 16];

                // Activations (A): Packed signed 4-bit integers [-8, 7]
                // Need sign extension from 4 bits
                int q_a_lo = (p_a & 0x0F); if (q_a_lo >= 8) q_a_lo -= 16;
                int q_a_hi = (p_a >> 4);   if (q_a_hi >= 8) q_a_hi -= 16;

                // Weights (B): Packed unsigned 4-bit integers [0, 15]
                // Need to subtract 8 to center at 0
                const int q_w_lo = (int)(is_hi ? (p_w_lo >> 4) : (p_w_lo & 0x0F)) - 8;
                const int q_w_hi = (int)(is_hi ? (p_w_hi >> 4) : (p_w_hi & 0x0F)) - 8;

                dot_i32 += q_a_lo * q_w_lo + q_a_hi * q_w_hi;
                sum_q_a += q_a_lo + q_a_hi;
                sum_q_w += q_w_lo + q_w_hi;
            }

            total_sum += scale_a * scale_w * (float)dot_i32;
            total_sum += scale_a * offset_w * (float)sum_q_a;
            total_sum += scale_w * offset_a * (float)sum_q_w;
            total_sum += offset_a * offset_w * 32.0f;
        }
    }

    // Reduction
    __shared__ float s_part[32];
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    float warp_sum = total_sum;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, mask);
    }

    if (lane == 0) s_part[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        float final_sum = (tid < (n_threads / 32)) ? s_part[tid] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            final_sum += __shfl_xor_sync(0xffffffff, final_sum, mask);
        }
        if (tid == 0) C[i] = final_sum;
    }
}

__global__ void rrs_gemm_accurate_kernel(
    const uint8_t* __restrict__ A,      // [M, K/2] packed activations
    const void*    __restrict__ B,      // [N, K/256] block_q4_K weights
    float*         __restrict__ C,      // [M, N] output
    const half*    __restrict__ a_scales,
    const half*    __restrict__ a_mins,
    const int M, const int N, const int K)
{
    const int col = blockIdx.x; // N
    const int row = blockIdx.y; // M
    
    if (col >= N || row >= M) return;

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    const block_q4_K* blocks_w = (const block_q4_K*)B + col * (K / 256);
    
    // Pointers for this row of A
    const uint8_t* A_row = A + row * (K / 2);
    const half* a_scales_row = a_scales + row * (K / 32);
    const half* a_mins_row = a_mins + row * (K / 32);

    float total_sum = 0.0f;

    for (int b = tid; b < K / 256; b += n_threads) {
        const block_q4_K& wb = blocks_w[b];
        const float d_w = __half2float(wb.dm.x);
        const float dmin_w = __half2float(wb.dm.y);

        uint8_t sc_w[8], mn_w[8];
        unpack_scales_mins_k4_cuda(wb.scales, sc_w, mn_w);

        for (int g = 0; g < 8; g++) {
            const float scale_w = d_w * (float)sc_w[g];
            const float offset_w = 8.0f * scale_w - dmin_w * (float)mn_w[g];

            const float scale_a = __half2float(a_scales_row[b * 8 + g]);
            const float offset_a = __half2float(a_mins_row[b * 8 + g]);

            const int p = g / 2;
            const uint8_t* qs_w = wb.qs + p * 32;
            const bool is_hi = (g % 2 != 0);
            const uint8_t* qs_a = A_row + (b * 256 + g * 32) / 2;

            int dot_i32 = 0;
            int sum_q_a = 0;
            int sum_q_w = 0;

            #pragma unroll
            for (int l = 0; l < 16; l++) {
                const uint8_t p_a = qs_a[l];
                const uint8_t p_w_lo = qs_w[l];
                const uint8_t p_w_hi = qs_w[l + 16];

                // Activations (A): Packed signed 4-bit integers [-8, 7]
                int q_a_lo = (p_a & 0x0F); if (q_a_lo >= 8) q_a_lo -= 16;
                int q_a_hi = (p_a >> 4);   if (q_a_hi >= 8) q_a_hi -= 16;

                // Weights (B): Packed unsigned 4-bit integers [0, 15]
                // Need to subtract 8 to center at 0
                const int q_w_lo = (int)(is_hi ? (p_w_lo >> 4) : (p_w_lo & 0x0F)) - 8;
                const int q_w_hi = (int)(is_hi ? (p_w_hi >> 4) : (p_w_hi & 0x0F)) - 8;

                dot_i32 += q_a_lo * q_w_lo + q_a_hi * q_w_hi;
                sum_q_a += q_a_lo + q_a_hi;
                sum_q_w += q_w_lo + q_w_hi;
            }

            total_sum += scale_a * scale_w * (float)dot_i32;
            total_sum += scale_a * offset_w * (float)sum_q_a;
            total_sum += scale_w * offset_a * (float)sum_q_w;
            total_sum += offset_a * offset_w * 32.0f;

#if RRS_DEBUG
            if (row == 0 && col == 0 && tid == 0 && b == 0 && g == 0) {
                // Debug print for first group of first block
                printf("[RRS KERNEL] b=0 g=0: scale_a=%f sum_q_a=%d | scale_w=%f offset_w=%f sum_q_w=%d | dot_i32=%d term1=%f term2=%f\n",
                    scale_a, sum_q_a,
                    scale_w, offset_w, sum_q_w,
                    dot_i32,
                    scale_a * scale_w * (float)dot_i32,
                    scale_a * offset_w * (float)sum_q_a);
            }
#endif
        }
    }

    // Reduction
    __shared__ float s_part[32];
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    float warp_sum = total_sum;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, mask);
    }

    if (lane == 0) s_part[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        float final_sum = (tid < (n_threads / 32)) ? s_part[tid] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            final_sum += __shfl_xor_sync(0xffffffff, final_sum, mask);
        }
        if (tid == 0) C[row * N + col] = final_sum;
    }
}

__global__ void rrs_dequant_simple_kernel(
    const int* __restrict__ C_i32,
    float* __restrict__ C_f32,
    const half* __restrict__ act_scales,
    const half* __restrict__ act_mins,     // Not used (symmetric quant)
    const half* __restrict__ weight_scales,
    const half* __restrict__ weight_mins,  // Actually offsets from unsigned->signed conversion
    const int M, const int N, const int K)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    const int sum_i32 = C_i32[row * N + col];
    const int groups = K / 32;
    
    float correction_term = 0.0f;
    float avg_act_scale = 0.0f;
    float avg_weight_scale = 0.0f;

    // Loop over groups to compute average scales and the cross-term correction
    for (int g = 0; g < groups; g++) {
        float s_a = __half2float(act_scales[row * groups + g]);
        float s_w = __half2float(weight_scales[col * groups + g]);
        
        // act_mins stores (scale_a * sum_q_a)
        // weight_mins stores offset_w
        // correction = sum( (scale_a * sum_q_a) * offset_w )
        float term_a = __half2float(act_mins[row * groups + g]); 
        float term_w = __half2float(weight_mins[col * groups + g]);
        
        correction_term += term_a * term_w;
        
        avg_act_scale += s_a;
        avg_weight_scale += s_w;
    }
    
    avg_act_scale /= groups;
    avg_weight_scale /= groups;
    
    // Result = sum_i32 * avg_s_a * avg_s_w + correction_term
    // We use average scales for the main dot product as an approximation (standard block quantization)
    // But we add the exact correction term for the cross product
    
    float result = (float)sum_i32 * avg_act_scale * avg_weight_scale + correction_term;
    
    C_f32[row * N + col] = result;
}

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
// Q4_K Block Extraction: Extract packed INT4 quants from Q4_K blocks
// Q4_K layout: [dm(4B), scales(12B), qs(128B)] per 256-element block
// 
// IMPORTANT: Q4_K stores UNSIGNED INT4 [0,15] but WMMA needs SIGNED INT4 [-8,7]
// We convert during extraction by XORing with 0x88 (flips bit 3 in each nibble)
// This transforms: 0->-8, 7->-1, 8->0, 15->7 (exactly what we need!)
// ============================================================================

__global__ void extract_q4k_quants_kernel(
    const void* __restrict__ q4k_blocks,
    uint8_t* __restrict__ packed_quants,
    half* __restrict__ weight_scales,    // Output: per-group scales for dequant
    half* __restrict__ weight_mins,      // Output: per-group mins for dequant (offset from unsigned->signed)
    const int N,   // Number of rows (output features)
    const int K)   // Number of columns (hidden dim)
{
    // Each thread handles one Q4_K block
    const int blocks_per_row = K / 256;
    const int total_blocks = N * blocks_per_row;
    
    for (int block_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         block_idx < total_blocks; 
         block_idx += blockDim.x * gridDim.x) 
    {
        const int row = block_idx / blocks_per_row;
        const int block_in_row = block_idx % blocks_per_row;
        
        const block_q4_K* src_block = (const block_q4_K*)q4k_blocks + block_idx;
        
        // Extract packed quants and convert unsigned [0,15] to signed [-8,7]
        // Q4_K stores unsigned values where q represents: val = d*sc*q - dmin*m
        // For WMMA, we need signed 2's complement: subtract 8 from each nibble
        // unsigned 0 -> signed -8 (0b1000), unsigned 8 -> signed 0 (0b0000), etc.
        uint8_t* dst_qs = packed_quants + row * (K / 2) + block_in_row * 128;
        
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            uint8_t orig = src_block->qs[i];
            // Extract nibbles, subtract 8, mask to 4 bits for 2's complement
            int lo = (orig & 0xF) - 8;
            int hi = (orig >> 4) - 8;
            dst_qs[i] = (lo & 0xF) | ((hi & 0xF) << 4);
        }
        
        // Extract and expand per-group scales for dequantization
        // Each Q4_K block has 8 groups of 32 elements
        const float d = __half2float(src_block->dm.x);
        const float dmin = __half2float(src_block->dm.y);
        
        uint8_t sc[8], mn[8];
        unpack_scales_mins_k4_cuda(src_block->scales, sc, mn);
        
        // Store expanded scales/mins for each of the 8 groups in this block
        // The dequant formula changes due to unsigned->signed conversion:
        // Original: val = d*sc*q_unsigned - dmin*m
        // After conversion: q_signed = q_unsigned - 8, so q_unsigned = q_signed + 8
        // Therefore: val = d*sc*(q_signed + 8) - dmin*m = d*sc*q_signed + (8*d*sc - dmin*m)
        // We store scale = d*sc, and offset = 8*d*sc - dmin*m (to add per element in dequant)
        const int base_group = row * (K / 32) + block_in_row * 8;
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            float scale = d * sc[g];
            float offset = 8.0f * scale - dmin * mn[g];
            weight_scales[base_group + g] = __float2half(scale);
            weight_mins[base_group + g] = __float2half(offset);  // Bias to add: 32 elements * offset per group
        }
    }
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
        sum = ggml_cuda_dp4a(a_packed, b_packed, sum);
    }
    
    C[row * N + col] = sum;
}

// ============================================================================
// Main Entry Points
// ============================================================================

void ggml_cuda_rrs_gemm_q4q4(
    const void* A,           // Quantized activations (packed Q4)
    const void* B,           // Weight data: packed Q4 (not Q4_K blocks!)
    float* C,
    int M, int N, int K,
    const half* scales_A, const half* mins_A,
    const half* scales_B, const half* mins_B,
    cudaStream_t stream)
{
    int32_t* C_i32;
    cudaMallocAsync(&C_i32, M * N * sizeof(int32_t), stream);
    
    // Zero-initialize output buffer - WMMA kernel may not write partial tiles
    // This ensures out-of-bounds locations contain zeros
    cudaMemsetAsync(C_i32, 0, M * N * sizeof(int32_t), stream);
    
    // Use new async pipeline kernel with automatic config selection
    rrs_gemm_i4_dispatch(
        (const uint8_t*)A, (const uint8_t*)B, C_i32,
        M, N, K, stream);
    
    dim3 dq_block(16, 16);
    dim3 dq_grid((M + 15) / 16, (N + 15) / 16);
    
#if RRS_DEBUG
    {
        cudaStreamSynchronize(stream);
        int32_t h_c_i32[4];
        cudaMemcpy(h_c_i32, C_i32, 4*sizeof(int32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[RRS CUDA] GEMM C_i32 first 4: %d %d %d %d\n", h_c_i32[0], h_c_i32[1], h_c_i32[2], h_c_i32[3]);
    }
#endif
    
    // Simple dequantization with separate scale arrays
    rrs_dequant_simple_kernel<<<dq_grid, dq_block, 0, stream>>>(
        C_i32, C, scales_A, mins_A, scales_B, mins_B, M, N, K);
    
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
    
    // Dequantize using simple kernel (separate scale arrays)
    rrs_dequant_simple_kernel<<<grid, block, 0, stream>>>(
        C_i32, C, scales_A_q8, mins_A, scales_B, mins_B, M, N, K);
    
    cudaFreeAsync(A_q8, stream);
    cudaFreeAsync(B_q8, stream);
    cudaFreeAsync(scales_A_q8, stream);
    cudaFreeAsync(C_i32, stream);
}

// ============================================================================
// Dispatch Integration
// ============================================================================

extern "C" void ggml_quantize_row_q4_K_rrs_act(const float * x, void * y, int64_t k);

__global__ void rrs_gemm_q4k_q4k_kernel(
    const void* __restrict__ A,      // [M, K/256] block_q4_K activations
    const void* __restrict__ B,      // [N, K/256] block_q4_K weights
    float*      __restrict__ C,      // [M, N] output
    const int M, const int N, const int K)
{
    const int col = blockIdx.x; // N
    const int row = blockIdx.y; // M
    
    if (col >= N || row >= M) return;

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    const block_q4_K* blocks_a = (const block_q4_K*)A + row * (K / 256);
    const block_q4_K* blocks_w = (const block_q4_K*)B + col * (K / 256);
    
    float total_sum = 0.0f;

    for (int b = tid; b < K / 256; b += n_threads) {
        const block_q4_K& ab = blocks_a[b];
        const block_q4_K& wb = blocks_w[b];

        // Unpack A
        const float d_a = __half2float(ab.dm.x);
        const float dmin_a = __half2float(ab.dm.y);
        uint8_t sc_a[8], mn_a[8];
        unpack_scales_mins_k4_cuda(ab.scales, sc_a, mn_a);

        // Unpack B (Weights)
        const float d_w = __half2float(wb.dm.x);
        const float dmin_w = __half2float(wb.dm.y);
        uint8_t sc_w[8], mn_w[8];
        unpack_scales_mins_k4_cuda(wb.scales, sc_w, mn_w);

        for (int g = 0; g < 8; g++) {
            const int p = g / 2;
            const bool is_hi = (g % 2 != 0);
            
            const uint8_t* qs_a_ptr = ab.qs + p * 32;
            const uint8_t* qs_w_ptr = wb.qs + p * 32;

            int dot_qq = 0;
            int sum_qa = 0;
            int sum_qw = 0;

            #pragma unroll
            for (int l = 0; l < 16; l++) {
                const uint8_t pa_0 = qs_a_ptr[l];
                const uint8_t pa_1 = qs_a_ptr[l+16];
                const uint8_t pw_0 = qs_w_ptr[l];
                const uint8_t pw_1 = qs_w_ptr[l+16];
                
                int qa0 = (int)(is_hi ? (pa_0 >> 4) : (pa_0 & 0x0F));
                int qa1 = (int)(is_hi ? (pa_1 >> 4) : (pa_1 & 0x0F));
                
                int qw0 = (int)(is_hi ? (pw_0 >> 4) : (pw_0 & 0x0F));
                int qw1 = (int)(is_hi ? (pw_1 >> 4) : (pw_1 & 0x0F));
                
                dot_qq += qa0 * qw0 + qa1 * qw1;
                sum_qa += qa0 + qa1;
                sum_qw += qw0 + qw1;
            }

            // Exactly match CPU formula:
            // S = (da*dw)*sum(dot_qq*sc_a*sc_w) - (da*dmin_w)*sum(sum_qa*sc_a*mn_w) - (dw*dmin_a)*sum(sum_qw*sc_w*mn_a) + (dmin_a*dmin_w)*32*mn_a*mn_w
            total_sum += (d_a * d_w) * (float)(dot_qq * (int)sc_a[g] * sc_w[g]);
            total_sum -= (d_a * dmin_w) * (float)(sum_qa * (int)sc_a[g] * mn_w[g]);
            total_sum -= (d_w * dmin_a) * (float)(sum_qw * (int)sc_w[g] * mn_a[g]);
            total_sum += (dmin_a * dmin_w) * 32.0f * (float)((int)mn_a[g] * mn_w[g]);
        }
    }

    // Reduction
    __shared__ float s_part[32];
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    float warp_sum = total_sum;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, mask);
    }

    if (lane == 0) s_part[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        float final_sum = (tid < (n_threads / 32)) ? s_part[tid] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            final_sum += __shfl_xor_sync(0xffffffff, final_sum, mask);
        }
        if (tid == 0) C[row * N + col] = final_sum;
    }
}

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
    
    // CPU Quantization Path
    
    // 1. Copy src1 (activations) to host (src1 is on GPU)
    size_t src1_size = M * K * sizeof(float);
    float* h_src1 = (float*)malloc(src1_size);
    if (!h_src1) {
        fprintf(stderr, "[RRS CUDA] Failed to allocate host buffer for src1\n");
        return;
    }
    
    // Handle non-contiguous src1 if necessary, though activations are usually contiguous
    if (src1->nb[1] == (size_t)K * sizeof(float)) {
        cudaMemcpy(h_src1, src1->data, src1_size, cudaMemcpyDeviceToHost);
    } else {
        for (int i = 0; i < M; i++) {
            cudaMemcpy(h_src1 + i * K, (const char*)src1->data + i * src1->nb[1], K * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    
    // 2. Allocate host buffer for quantized activations
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS_ACT, K);
    size_t total_size = M * row_size;
    
    void* h_act_q4k = malloc(total_size);
    if (!h_act_q4k) {
        fprintf(stderr, "[RRS CUDA] Failed to allocate host buffer for act_q4k\n");
        free(h_src1);
        return;
    }
    
    // 3. Quantize on CPU (includes RRS transform)
    uint8_t* dst_ptr = (uint8_t*)h_act_q4k;
    for (int i = 0; i < M; i++) {
        ggml_quantize_row_q4_K_rrs_act(h_src1 + i * K, dst_ptr + i * row_size, K);
    }
    
    // 4. Copy to GPU
    void* d_act_q4k;
    cudaMallocAsync(&d_act_q4k, total_size, stream);
    cudaMemcpyAsync(d_act_q4k, h_act_q4k, total_size, cudaMemcpyHostToDevice, stream);
    
    // 5. Run Kernel
    dim3 grid(N, M);
    rrs_gemm_q4k_q4k_kernel<<<grid, 256, 0, stream>>>(
        d_act_q4k, src0->data, (float*)dst->data, M, N, K);
        
    // Cleanup
    cudaFreeAsync(d_act_q4k, stream);
    cudaStreamSynchronize(stream);
    free(h_src1);
    free(h_act_q4k);
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

#define RRS_CHECK(call) do { \
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
    RRS_CHECK(cudaStreamCreate(&stream));
    
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
    
    RRS_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    RRS_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    RRS_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    RRS_CHECK(cudaMalloc(&d_A_q4, M * K / 2));
    RRS_CHECK(cudaMalloc(&d_B_q4, N * K / 2));
    
    const int groups_A = M * (K / 32);
    const int groups_B = N * (K / 32);
    RRS_CHECK(cudaMalloc(&d_scales_A, groups_A * sizeof(half)));
    RRS_CHECK(cudaMalloc(&d_mins_A, groups_A * sizeof(half)));
    RRS_CHECK(cudaMalloc(&d_scales_B, groups_B * sizeof(half)));
    RRS_CHECK(cudaMalloc(&d_mins_B, groups_B * sizeof(half)));
    
    RRS_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    RRS_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark FWHT
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_rrs_fwht(d_A, d_A, K, M, stream);
    }
    cudaEventRecord(stop, stream);
    RRS_CHECK(cudaEventSynchronize(stop));
    RRS_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&result->fwht_time_ms, start, stop);
    result->fwht_time_ms /= iterations;
    
    // Benchmark FWHT + Quantize (fused)
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        ggml_cuda_rrs_fwht_quantize(d_A, d_A_q4, d_scales_A, d_mins_A, K, M, stream);
    }
    cudaEventRecord(stop, stream);
    RRS_CHECK(cudaEventSynchronize(stop));
    RRS_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&result->quantize_time_ms, start, stop);
    result->quantize_time_ms /= iterations;
    
    // Pre-quantize B for GEMM benchmark
    ggml_cuda_rrs_fwht_quantize(d_B, d_B_q4, d_scales_B, d_mins_B, K, N, stream);
    RRS_CHECK(cudaStreamSynchronize(stream));
    RRS_CHECK(cudaGetLastError());
    
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
    RRS_CHECK(cudaEventSynchronize(stop));
    RRS_CHECK(cudaGetLastError());
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
    RRS_CHECK(cudaEventSynchronize(stop));
    RRS_CHECK(cudaGetLastError());
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
    
    // Test with M=1 (single token inference)
    printf("=== Single Token Inference (M=1, N=2048, K=2048) ===\n");
    ggml_cuda_rrs_benchmark(1, 2048, 2048, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    // Test with M=32 (small batch)
    printf("=== Small Batch (M=32, N=2048, K=2048) ===\n");
    ggml_cuda_rrs_benchmark(32, 2048, 2048, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    // Test with M=128 (medium batch)
    printf("=== Medium Batch (M=128, N=2048, K=2048) ===\n");
    ggml_cuda_rrs_benchmark(128, 2048, 2048, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    // Test with M=512 (large batch / prefill)
    printf("=== Large Batch / Prefill (M=512, N=2048, K=2048) ===\n");
    ggml_cuda_rrs_benchmark(512, 2048, 2048, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    // Test with larger N,K (typical LLM dimensions)
    printf("=== LLM-scale (M=256, N=8192, K=8192) ===\n");
    ggml_cuda_rrs_benchmark(256, 8192, 8192, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    // Test with very large M (big prefill)
    printf("=== Big Prefill (M=1024, N=8192, K=8192) ===\n");
    ggml_cuda_rrs_benchmark(1024, 8192, 8192, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    // Test with QuadMul reference size (from their benchmarks)
    printf("=== QuadMul Reference (M=2048, N=8192, K=8192) ===\n");
    ggml_cuda_rrs_benchmark(2048, 8192, 8192, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
    printf("\n");
    
    printf("RRS CUDA Test: COMPLETED\n");
    printf("\nReference: QuadMul 4090 achieves 700-760 TOPS for M=1024-2048, N=8192, K=8192\n");
    printf("RTX 3090 theoretical INT4 peak: ~568 TOPS\n");
}