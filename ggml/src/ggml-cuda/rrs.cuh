#pragma once

#include "common.cuh"
#include <mma.h>

// RRS (Rotated Runtime Smooth) CUDA Kernels
// Enables true W4A4 (4-bit weights, 4-bit activations) using INT4 Tensor Cores

// Check for INT4 WMMA support (Turing SM75+)
#if __CUDA_ARCH__ >= 750
#define RRS_INT4_TENSOR_CORES 1
#endif

// WMMA tile dimensions for INT4
// Turing/Ampere: 8x8x32 (s4 precision)
#define RRS_WMMA_M 8
#define RRS_WMMA_N 8  
#define RRS_WMMA_K 32

// Block configuration for RRS GEMM
#define RRS_BLOCK_WARPS_M 4
#define RRS_BLOCK_WARPS_N 4
#define RRS_WARP_TILES_M 2
#define RRS_WARP_TILES_N 4
#define RRS_CHUNK_K 2
#define RRS_NUM_STAGES 2

// Derived constants
#define RRS_TILE_M (RRS_WMMA_M * RRS_WARP_TILES_M * RRS_BLOCK_WARPS_M)  // 64
#define RRS_TILE_N (RRS_WMMA_N * RRS_WARP_TILES_N * RRS_BLOCK_WARPS_N)  // 128
#define RRS_TILE_K (RRS_WMMA_K * RRS_CHUNK_K)                           // 64

// Q4_K block parameters (256 elements per super-block)
#define RRS_QK_K 256
#define RRS_K_SCALE_SIZE 12

// Simplified RRS block format for CUDA (optimized for tensor cores)
// Per 256-element block: scale (fp16) + min (fp16) + 128 bytes of packed int4
struct block_q4_rrs {
    half d;           // scale
    half dmin;        // min value
    uint8_t qs[128];  // packed int4 quants (2 per byte)
};

// Forward declarations
void ggml_cuda_rrs_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,  // weights (Q4_K RRS)
    const ggml_tensor * src1,  // activations (F32, will be FWHT'd and quantized)
    ggml_tensor * dst);

// FWHT (Fast Walsh-Hadamard Transform) for activation preprocessing
void ggml_cuda_rrs_fwht(
    const float * x,
    float * y,
    int n,
    int batch_size,
    cudaStream_t stream);

// Activation quantization (F32 -> Q4 packed)
void ggml_cuda_rrs_quantize_act(
    const float * x,
    void * y,
    int n,
    int batch_size, 
    cudaStream_t stream);

// INT4 Tensor Core GEMM (both operands Q4)
void ggml_cuda_rrs_gemm_q4q4(
    const void * A,      // activations (Q4 packed, FWHT'd)
    const void * B,      // weights (Q4_K RRS format)
    float * C,           // output (F32)
    int M, int N, int K,
    const half * scales_A, const half * mins_A,  // activation scales
    const half * scales_B, const half * mins_B,  // weight scales  
    cudaStream_t stream);

// Fused FWHT + Quantize kernel
void ggml_cuda_rrs_fwht_quantize(
    const float * x,
    void * y,
    half * scales,
    half * mins,
    int n,
    int batch_size,
    cudaStream_t stream);

