#pragma once

#include "common.cuh"
#include <mma.h>
#include <unordered_map>
#include <string>

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
// Shared Helpers
// ============================================================================

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

// TCQ4-specific FWHT with fixed step=256 (for embedding inverse transform)
// TCQ4_K32 is quantized with 256-element FWHT chunks, so inverse must match
void ggml_cuda_tcq4_fwht_step256(
    const float * x,       // input [batch_size, n]
    float * y,             // output [batch_size, n]
    int n,                 // dimension (must be multiple of 256)
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
// Output: block_q4_K array [batch_size, n/256]
void ggml_cuda_rrs_fwht_quantize(
    const float * x,       // input F32 [batch_size, n]
    void * y,              // output block_q4_K [batch_size, n/256]
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

// Simple test function - verifies CUDA RRS kernels work
extern "C" void ggml_cuda_rrs_test(void);

// ============================================================================
// Channel Permutation Support (RRS Paper Section 3.2)
// ============================================================================
// Channel reordering groups outlier channels together to improve Runtime Smooth.
// Permutations are loaded from GGUF metadata and stored in a registry.

// Register a channel permutation for a tensor (called during model load)
// tensor_name: name of the weight tensor (e.g., "blk.0.attn_q.weight")
// h_perm: host pointer to permutation indices [K]
// K: number of channels
// Copies permutation to device memory
void ggml_cuda_rrs_register_perm(
    const char* tensor_name,
    const int32_t* h_perm,
    int K);

// Get device pointer to channel permutation for a tensor
// Returns NULL if no permutation registered for this tensor
const int32_t* ggml_cuda_rrs_get_perm(const char* tensor_name);

// Check if channel reordering is enabled for a tensor
bool ggml_cuda_rrs_has_perm(const char* tensor_name);

// Clear all registered permutations (called on model unload)
void ggml_cuda_rrs_clear_perms();

// Get/set global flag for whether to use channel reordering
void ggml_cuda_rrs_set_reorder_enabled(bool enabled);
bool ggml_cuda_rrs_get_reorder_enabled();