#pragma once

#include "common.cuh"
#include <mma.h>

// ============================================================================
// RRS (Rotated Runtime Smooth) CUDA Kernels
// Enables true W4A4 (4-bit weights, 4-bit activations) using INT4 Tensor Cores
// ============================================================================

// Hardware Requirements: SM75+ (Turing/Ampere/Ada/Hopper)
// RTX 3090 (SM86): ~568 TOPS INT4 theoretical peak

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct RRSBenchmarkResult {
    float int4_wmma_time_ms;   // INT4 WMMA tensor core path
    float q8_repack_time_ms;   // Q4->Q8 repack + dp4a path
    float fwht_time_ms;        // FWHT transform only
    float quantize_time_ms;    // Fused FWHT + quantize
    int M, N, K;               // Matrix dimensions
};

// ============================================================================
// Main Entry Points
// ============================================================================

// Full RRS mul_mat dispatch (activations: F32 -> FWHT -> Q4, weights: Q4_K_RRS)
void ggml_cuda_rrs_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,  // weights (Q4_K_RRS format)
    const ggml_tensor * src1,  // activations (F32, will be FWHT'd and quantized)
    ggml_tensor * dst);

// Check if tensor type supports RRS CUDA acceleration
bool ggml_cuda_supports_rrs(const ggml_tensor * tensor);

// ============================================================================
// Component Kernels
// ============================================================================

// FWHT (Fast Walsh-Hadamard Transform) for activation preprocessing
// Supports dimensions: 64, 128, 256, 512, 1024, 2048, 4096 (power of 2)
// Non-power-of-2 handled via chunked transform
void ggml_cuda_rrs_fwht(
    const float * x,       // input [batch_size, n]
    float * y,             // output [batch_size, n]
    int n,                 // dimension (should be power of 2 for best perf)
    int batch_size,
    cudaStream_t stream);

// Activation quantization (F32 -> Q4 packed with per-32-group scales)
void ggml_cuda_rrs_quantize_act(
    const float * x,
    void * y,
    int n,
    int batch_size, 
    cudaStream_t stream);

// Fused FWHT + Quantize (preferred - single pass)
void ggml_cuda_rrs_fwht_quantize(
    const float * x,       // input F32 [batch_size, n]
    void * y,              // output Q4 packed [batch_size, n/2]
    half * scales,         // output scales [batch_size, n/32]
    half * mins,           // output mins [batch_size, n/32]
    int n,
    int batch_size,
    cudaStream_t stream);

// ============================================================================
// GEMM Kernels
// ============================================================================

// PATH A: INT4 Tensor Core GEMM (WMMA s4, SM75+)
// Both operands must be Q4 packed
void ggml_cuda_rrs_gemm_q4q4(
    const void * A,        // activations Q4 [M, K/2] packed
    const void * B,        // weights Q4 [N, K/2] packed (transposed)
    float * C,             // output F32 [M, N]
    int M, int N, int K,
    const half * scales_A, const half * mins_A,  // activation scales [M, K/32]
    const half * scales_B, const half * mins_B,  // weight scales (or nullptr for Q4_K blocks)
    cudaStream_t stream);

// PATH B: Q4 -> Q8 repack + dp4a GEMM (fallback, all CUDA devices)
void ggml_cuda_rrs_gemm_q4_via_q8(
    const void * A_q4,
    const void * B_q4,
    float * C,
    int M, int N, int K,
    const half * scales_A, const half * mins_A,
    const half * scales_B, const half * mins_B,
    cudaStream_t stream);

// ============================================================================
// Benchmarking
// ============================================================================

// Run benchmark comparing INT4 WMMA vs Q8 repack paths
void ggml_cuda_rrs_benchmark(
    int M, int N, int K,
    int iterations,
    RRSBenchmarkResult * result);

// Print benchmark results to stdout
void ggml_cuda_rrs_print_benchmark(const RRSBenchmarkResult * result);