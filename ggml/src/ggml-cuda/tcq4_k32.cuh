#pragma once

#include "common.cuh"
#include <cuda_fp16.h>

// ============================================================================
// TCQ4 Tile Format: Tensor Core Native INT4 for RRS W4A4
// ============================================================================
//
// Tile structure (2048 weights = 8 channels × 256 K = 8 channels × 8 groups × 32):
//   - tiles[8][128]: IMMA B fragments (8 K-groups, each 128 bytes in IMMA order)
//   - S[8] (fp16): per-channel super-scales
//   - Z[8] (fp16): per-channel super-zeros
//   - sc[8][8] (int8): per-channel per-group scale codes
//   - zc[8][8] (int8): per-channel per-group zero codes
//
// Weight dequantization: w = (S[c] * sc[c][g] / 127) * q + (Z[c] * zc[c][g] / 127)
//
// IMMA compatibility: 
//   - mma.sync.m16n8k32.s4.s4 uses K=32, matching group size
//   - tiles[g] pre-packed so lane L loads [L*4:L*4+4] as uint32 for B operand
//   - 8 channels per tile matches IMMA N=8 output
//
// Storage: 1184 bytes / 2048 weights = 4.625 bits/weight
// ============================================================================

// Tile constants (from ggml-common.h)
#ifndef TCQ4_TILE_K
#define TCQ4_TILE_K         256
#define TCQ4_TILE_CHANNELS  8
#define TCQ4_TILE_GROUPS    8
#define TCQ4_TILE_GROUP_SIZE 32
#define TCQ4_TILE_WEIGHTS   2048
#endif

// ============================================================================
// RRS W4A4 Activation Structures
// ============================================================================

// Runtime Smooth quantized activations - scalar path (144 bytes per 256 elements)
// Padded to 16-byte alignment for vectorized loads
struct __align__(16) block_rrs_int4 {
    uint8_t qs[128];    // 256 signed INT4 values packed
    float smooth_scale; // max(|x|) for this block (dequant: q * smooth_scale / 7)
    uint8_t _pad[12];   // Padding to 144 bytes (multiple of 16)
};

// Runtime Smooth activations - tensor core path (160 bytes per 256 elements)
// Includes precomputed sum(q) per group for zero-point correction
// Padded to 16-byte alignment for vectorized loads
struct __align__(16) block_rrs_int4_tc {
    uint8_t qs[128];      // 256 signed INT4 values packed
    float smooth_scale;   // max(|x|) for this block
    int16_t sum_q[8];     // Precomputed sum of q for each group of 32
    uint8_t _pad[12];     // Padding to 160 bytes (multiple of 16)
};

// ============================================================================
// Fused Activation Pipeline (Perm + FWHT + Quantize)
// ============================================================================

// Fused activation quantization - scalar output
// Pipeline: (optional permutation) -> FWHT -> Runtime Smooth -> INT4 quantize
void tcq4_rrs_fwht_quantize(
    const float* __restrict__ x,
    void* __restrict__ y,
    int K,
    int batch_size,
    cudaStream_t stream);

// Fused activation quantization with permutation - scalar output
void tcq4_rrs_perm_fwht_quantize(
    const float* __restrict__ x,
    void* __restrict__ y,
    const int32_t* __restrict__ perm,
    int K,
    int batch_size,
    cudaStream_t stream);

// Fused activation quantization - tensor core output (with group sums)
void tcq4_rrs_fwht_quantize_tc(
    const float* __restrict__ x,
    void* __restrict__ y,
    int K,
    int batch_size,
    cudaStream_t stream);

// Fused activation quantization with permutation - tensor core output
void tcq4_rrs_perm_fwht_quantize_tc(
    const float* __restrict__ x,
    void* __restrict__ y,
    const int32_t* __restrict__ perm,
    int K,
    int batch_size,
    cudaStream_t stream);

// ============================================================================
// RRS W4A4 GEMV (M=1) - Token Generation
// ============================================================================

// Scalar GEMV
void tcq4_rrs_gemv(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int N, int K,
    cudaStream_t stream);

// Tensor core GEMV (uses precomputed group sums)
void tcq4_rrs_gemv_tc(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int N, int K,
    cudaStream_t stream);

// ============================================================================
// RRS W4A4 GEMM (M>1) - Prompt Processing
// ============================================================================

// Scalar GEMM (fallback)
void tcq4_rrs_gemm(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int M, int N, int K,
    cudaStream_t stream);

// IMMA tensor core GEMM - optimized v12 kernel
// 75 TFLOPS on RTX 3090, 3x faster than Q4_K dp4a
void tcq4_rrs_gemm_imma(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int M, int N, int K,
    cudaStream_t stream);