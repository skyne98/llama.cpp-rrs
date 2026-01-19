#pragma once

#include "common.cuh"
#include <cuda_fp16.h>

// ============================================================================
// TCQ4-K32: Tensor Core Native INT4 Format
// ============================================================================
//
// Block structure (256 weights = 8 groups × 32):
//   - qs[128]: 256 signed INT4 values packed in IMMA-friendly order
//   - S (fp16): scale-of-scales
//   - Z (fp16): scale-of-zeros  
//   - sc[8] (int8): per-group scale codes
//   - zc[8] (int8): per-group zero/offset codes
//
// Dequantization: w = (S * sc[g]) * q + (Z * zc[g])
//
// IMMA compatibility: m16n8k32 uses K=32, matching our group size exactly.
// Each 256-weight block = one (K=32, N=8) IMMA tile for weights.
//
// ============================================================================

// Block size constants (also in ggml-common.h)
#define TCQ4_K32_BLOCK_SIZE 256
#define TCQ4_K32_NUM_GROUPS 8
#define TCQ4_K32_GROUP_SIZE 32
#define TCQ4_K32_BYTES_PER_BLOCK 148  // 128 + 4 + 16

// ============================================================================
// Conversion Functions
// ============================================================================

// Convert Q4_K weights to TCQ4_K32 format (offline, on GPU)
// Input:  Q4_K blocks [num_blocks]
// Output: TCQ4_K32 blocks [num_blocks]
void tcq4_k32_convert_from_q4k(
    const void* __restrict__ src_q4k,
    void* __restrict__ dst_tcq4,
    int num_blocks,
    cudaStream_t stream);

// Convert TCQ4_K32 back to Q4_K (for validation/export)
void tcq4_k32_convert_to_q4k(
    const void* __restrict__ src_tcq4,
    void* __restrict__ dst_q4k,
    int num_blocks,
    cudaStream_t stream);

// ============================================================================
// GEMM Kernels - INT4 × INT4 → FP32
// ============================================================================

// GEMM for M > 1 (prompt processing) using IMMA tensor cores
// A: TCQ4_K32 activations [M, K/256 blocks]
// B: TCQ4_K32 weights [N, K/256 blocks] (row-major, each row is one output channel)
// C: FP32 output [M, N]
void tcq4_k32_gemm_imma(
    const void* __restrict__ A,
    const void* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    cudaStream_t stream);

// GEMV for M = 1 (token generation) - optimized single-row path
// Uses warp-level reduction with dp4a (IMMA overhead not worth it for M=1)
// A: TCQ4_K32 activations [1, K/256 blocks]
// B: TCQ4_K32 weights [N, K/256 blocks]
// C: FP32 output [1, N]
void tcq4_k32_gemv(
    const void* __restrict__ A,
    const void* __restrict__ B,
    float* __restrict__ C,
    int N, int K,
    cudaStream_t stream);

// GEMV for M = 1 with Q8_1 activations (W4A8 mode - better accuracy)
// A: Q8_1 activations [K/32 blocks]
// B: TCQ4_K32 weights [N, K/256 blocks]
// C: FP32 output [N]
void tcq4_k32_q8_gemv(
    const void* __restrict__ A_q8,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int N, int K,
    cudaStream_t stream);

// ============================================================================
// Fused Kernels for RRS Pipeline
// ============================================================================

// Fused FWHT + Quantize to TCQ4_K32
// Input: FP32 activations [batch_size, K]
// Output: TCQ4_K32 blocks [batch_size, K/256]
void tcq4_k32_fwht_quantize(
    const float* __restrict__ x,
    void* __restrict__ y,
    int K,
    int batch_size,
    cudaStream_t stream);

// Fused FWHT + Quantize to Q8_1 for W4A8 mode
// Uses step=256 FWHT to match TCQ4 weight quantization (not K&-K like Q4_K_RRS)
// Input: FP32 activations [batch_size, K]
// Output: Q8_1 blocks [batch_size, K/32]
void tcq4_fwht_quantize_q8(
    const float* __restrict__ src,
    void* __restrict__ dst,
    int K,
    int batch_size,
    cudaStream_t stream);

// ============================================================================
// Runtime Smooth W4A4 Kernels (Paper: "Rotated Runtime Smooth")
// ============================================================================
// Key insight: divide by group-max BEFORE quantization, multiply AFTER dot product
// This normalizes activations to ~[-1,1] range, improving INT4 utilization

// Structure for Runtime Smooth quantized activations (132 bytes per 256 elements)
struct block_rrs_int4 {
    uint8_t qs[128];    // 256 signed INT4 values packed
    float smooth_scale; // Runtime smooth scale for this block
};

// Structure for Runtime Smooth activations with tensor core support (148 bytes per 256 elements)
// Includes precomputed sum(q_a) per group, needed for weight zero-point correction
struct block_rrs_int4_tc {
    uint8_t qs[128];      // 256 signed INT4 values packed in IMMA-friendly layout
    float smooth_scale;   // Runtime smooth scale for this block
    int16_t sum_q[8];     // Precomputed sum of q_a for each group of 32 (for zero-point term)
};

// Fused FWHT + Runtime Smooth + INT4 quantization
// Input: FP32 activations [batch_size, K]
// Output: block_rrs_int4 blocks [batch_size, K/256]
void tcq4_rrs_fwht_quantize(
    const float* __restrict__ x,
    void* __restrict__ y,
    int K,
    int batch_size,
    cudaStream_t stream);

// Runtime Smooth W4A4 GEMV (M=1)
// A: block_rrs_int4 [K/256] - RRS quantized activations with smooth scales
// B: TCQ4_K32 [N, K/256] - pre-quantized weights
// C: FP32 [N] - output
void tcq4_rrs_gemv(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int N, int K,
    cudaStream_t stream);

// Runtime Smooth W4A4 GEMM (M>1)
// A: block_rrs_int4 [M, K/256] - RRS quantized activations
// B: TCQ4_K32 [N, K/256] - pre-quantized weights
// C: FP32 [M, N] - output
void tcq4_rrs_gemm(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int M, int N, int K,
    cudaStream_t stream);

// ============================================================================
// Tensor Core Accelerated Runtime Smooth W4A4 Kernels
// ============================================================================
// Uses mma.sync.m16n8k32.s4.s4 for INT4×INT4 tensor core acceleration
// Requires SM75+ (Turing/Ampere/Ada)

// Fused FWHT + Runtime Smooth + INT4 quantization with tensor core layout
// Also precomputes sum(q_a) per group for zero-point correction
// Input: FP32 activations [batch_size, K]
// Output: block_rrs_int4_tc blocks [batch_size, K/256]
void tcq4_rrs_fwht_quantize_tc(
    const float* __restrict__ x,
    void* __restrict__ y,
    int K,
    int batch_size,
    cudaStream_t stream);

// Runtime Smooth W4A4 GEMV with tensor cores (M=1)
// A: block_rrs_int4_tc [K/256] - RRS quantized activations with group sums
// B: TCQ4_K32 [N, K/256] - pre-quantized weights
// C: FP32 [N] - output
void tcq4_rrs_gemv_tc(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int N, int K,
    cudaStream_t stream);

// Runtime Smooth W4A4 GEMM with IMMA tensor cores (M>1)
// Uses mma.sync.m16n8k32.s4.s4 for maximum throughput
// A: block_rrs_int4_tc [M, K/256] - RRS quantized activations with group sums
// B: TCQ4_K32 [N, K/256] - pre-quantized weights (IMMA-friendly layout)
// C: FP32 [M, N] - output
void tcq4_rrs_gemm_imma(
    const void* __restrict__ A_rrs,
    const void* __restrict__ B_tcq4,
    float* __restrict__ C,
    int M, int N, int K,
    cudaStream_t stream);

// ============================================================================
// Device Helper Functions
// ============================================================================

// Unpack INT4 pair from byte (low nibble first)
__device__ __forceinline__ void tcq4_unpack_int4(uint8_t packed, int8_t& lo, int8_t& hi) {
    // Signed INT4: values in [-8, 7]
    int8_t lo_raw = packed & 0x0F;
    int8_t hi_raw = (packed >> 4) & 0x0F;
    // Sign extend from 4 bits
    lo = (lo_raw >= 8) ? (lo_raw - 16) : lo_raw;
    hi = (hi_raw >= 8) ? (hi_raw - 16) : hi_raw;
}

// Pack two INT4 values into one byte
__device__ __forceinline__ uint8_t tcq4_pack_int4(int8_t lo, int8_t hi) {
    return ((uint8_t)(hi & 0x0F) << 4) | ((uint8_t)(lo & 0x0F));
}

// Dequantize a single TCQ4_K32 value
// g: group index (0-7), i: index within group (0-31)
__device__ __forceinline__ float tcq4_k32_dequant(
    const uint8_t* qs,    // packed quants
    half S, half Z,
    int8_t sc_g, int8_t zc_g,
    int g, int i
) {
    int idx = g * 32 + i;
    int byte_idx = idx / 2;
    uint8_t packed = qs[byte_idx];
    
    int8_t q;
    if (idx & 1) {
        q = ((packed >> 4) & 0x0F);
        q = (q >= 8) ? (q - 16) : q;
    } else {
        q = (packed & 0x0F);
        q = (q >= 8) ? (q - 16) : q;
    }
    
    float s_g = __half2float(S) * (float)sc_g;
    float z_g = __half2float(Z) * (float)zc_g;
    
    return s_g * (float)q + z_g;
}

// Compute dot product of one TCQ4_K32 block pair using dp4a
// Returns sum of (a_i * b_i) for all 256 elements
__device__ __forceinline__ float tcq4_k32_block_dot_dp4a(
    const uint8_t* __restrict__ a_qs,
    half a_S, half a_Z,
    const int8_t* __restrict__ a_sc,
    const int8_t* __restrict__ a_zc,
    const uint8_t* __restrict__ b_qs,
    half b_S, half b_Z,
    const int8_t* __restrict__ b_sc,
    const int8_t* __restrict__ b_zc
) {
    // For each group g: dot = sum_i (s_a[g] * q_a[i] + z_a[g]) * (s_b[g] * q_b[i] + z_b[g])
    // Expand: s_a * s_b * sum(q_a * q_b) + s_a * z_b * sum(q_a) + z_a * s_b * sum(q_b) + z_a * z_b * 32
    
    float total = 0.0f;
    
    float S_a = __half2float(a_S);
    float Z_a = __half2float(a_Z);
    float S_b = __half2float(b_S);
    float Z_b = __half2float(b_Z);
    
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        float s_a = S_a * (float)a_sc[g];
        float z_a = Z_a * (float)a_zc[g];
        float s_b = S_b * (float)b_sc[g];
        float z_b = Z_b * (float)b_zc[g];
        
        // Compute sums for this group using dp4a
        int32_t dot_qq = 0;    // sum(q_a * q_b)
        int32_t sum_qa = 0;    // sum(q_a)
        int32_t sum_qb = 0;    // sum(q_b)
        
        // Process 32 elements = 16 bytes = 4 int32 words
        const int32_t* a_words = (const int32_t*)(a_qs + g * 16);
        const int32_t* b_words = (const int32_t*)(b_qs + g * 16);
        
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            int32_t a_word = a_words[w];
            int32_t b_word = b_words[w];
            
            // Extract 8 INT4 values from each word and accumulate
            // dp4a works on INT8, so we need to handle INT4 differently
            // For now, do scalar extraction (TODO: optimize with bit tricks)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int8_t qa = ((a_word >> (j * 4)) & 0xF);
                qa = (qa >= 8) ? (qa - 16) : qa;
                int8_t qb = ((b_word >> (j * 4)) & 0xF);
                qb = (qb >= 8) ? (qb - 16) : qb;
                
                dot_qq += (int32_t)qa * (int32_t)qb;
                sum_qa += qa;
                sum_qb += qb;
            }
        }
        
        // Accumulate: s_a*s_b*dot_qq + s_a*z_b*sum_qa + z_a*s_b*sum_qb + z_a*z_b*32
        total += s_a * s_b * (float)dot_qq;
        total += s_a * z_b * (float)sum_qa;
        total += z_a * s_b * (float)sum_qb;
        total += z_a * z_b * 32.0f;
    }
    
    return total;
}

// ============================================================================
// IMMA Fragment Helpers
// ============================================================================

// For mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32:
// A (activations): 16 rows × 32 cols (row-major)
// B (weights): 32 rows × 8 cols (col-major)
// C (output): 16 rows × 8 cols
//
// Fragment storage per thread (32 threads/warp):
// A: 4 registers (16 INT4 values = 64 bits = 2 int32)
// B: 2 registers (8 INT4 values = 32 bits = 1 int32)
// C: 4 registers (4 INT32 values)

// Thread-to-element mapping for A fragment (row-major, 16×32)
__device__ __forceinline__ void tcq4_imma_a_thread_map(int lane, int& row, int& col_base) {
    // Each thread owns 16 consecutive INT4 values across 2 rows
    row = (lane / 4) % 8;
    col_base = (lane % 4) * 8;
}

// Thread-to-element mapping for B fragment (col-major, 32×8)
__device__ __forceinline__ void tcq4_imma_b_thread_map(int lane, int& row_base, int& col) {
    // Each thread owns 8 consecutive INT4 values down one column
    row_base = (lane / 4) * 8;
    col = lane % 4;
}

// ============================================================================
// Benchmark / Test
// ============================================================================

// Run micro-benchmark comparing TCQ4_K32 IMMA vs dp4a vs Q4_K
struct TCQ4BenchResult {
    float tcq4_imma_us;     // TCQ4_K32 with IMMA path
    float tcq4_dp4a_us;     // TCQ4_K32 with dp4a fallback
    float q4k_dp4a_us;      // Original Q4_K with dp4a
    float convert_us;        // Q4_K -> TCQ4_K32 conversion time
    int M, N, K;
    int iterations;
};

void tcq4_k32_benchmark(
    int M, int N, int K,
    int iterations,
    TCQ4BenchResult* result,
    cudaStream_t stream);

void tcq4_k32_print_benchmark(const TCQ4BenchResult* result);

// Validate TCQ4_K32 conversion correctness
bool tcq4_k32_validate_conversion(int num_blocks, cudaStream_t stream);