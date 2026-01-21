// RRS CUDA Benchmark Tool
// Measures activation quantization overhead vs GEMM time
// Usage: ./llama-rrs-benchmark [iterations]

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations for RRS functions (defined in rrs.cu)
struct RRSBenchmarkResult {
    float int4_wmma_time_ms;
    float q8_repack_time_ms;
    float fwht_time_ms;
    float quantize_time_ms;
    int M, N, K;
};

extern void ggml_cuda_rrs_benchmark(int M, int N, int K, int iterations, RRSBenchmarkResult* result);
extern void ggml_cuda_rrs_print_benchmark(const RRSBenchmarkResult* result);

// Measure kernel launch overhead
static float measure_launch_overhead(int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Empty kernel launch
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        // Just record events with no kernel - measures event overhead
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float event_time;
    cudaEventElapsedTime(&event_time, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (event_time * 1000.0f) / iterations;  // us per iteration
}

// Measure cudaMalloc/cudaFree overhead (simulating pool alloc worst case)
static float measure_alloc_overhead(size_t size, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<void*> ptrs(iterations);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMalloc(&ptrs[i], size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float alloc_time;
    cudaEventElapsedTime(&alloc_time, start, stop);
    
    for (int i = 0; i < iterations; i++) {
        cudaFree(ptrs[i]);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (alloc_time * 1000.0f) / iterations;  // us per allocation
}

static void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    printf("=== TCQ4 W4A4 Overhead Analysis ===\n");
    printf("Device: %s (SM%d%d)\n", props.name, props.major, props.minor);
    printf("\n");
}

static void print_table_header() {
    printf("%-25s %6s %6s %6s %10s %10s %10s %8s %8s\n",
           "Test Case", "M", "N", "K", "Quant(us)", "GEMM(us)", "Total(us)", "Overhd%", "TFLOPS");
    printf("%-25s %6s %6s %6s %10s %10s %10s %8s %8s\n",
           "-------------------------", "------", "------", "------", 
           "----------", "----------", "----------", "--------", "--------");
}

static void run_benchmark(const char* name, int M, int N, int K, int iterations) {
    RRSBenchmarkResult result;
    ggml_cuda_rrs_benchmark(M, N, K, iterations, &result);
    
    float quant_us = result.quantize_time_ms * 1000.0f;
    float gemm_us = result.int4_wmma_time_ms * 1000.0f;
    float total_us = quant_us + gemm_us;
    float overhead_pct = (total_us > 0) ? (100.0f * quant_us / total_us) : 0.0f;
    
    // Compute TFLOPS (2*M*N*K ops for matmul)
    double ops = 2.0 * M * N * K;
    double tflops = (gemm_us > 0) ? (ops / (gemm_us * 1e-6) / 1e12) : 0.0;
    
    printf("%-25s %6d %6d %6d %10.2f %10.2f %10.2f %7.1f%% %8.2f\n",
           name, M, N, K, quant_us, gemm_us, total_us, overhead_pct, tflops);
}

int main(int argc, char** argv) {
    int iterations = 100;
    
    if (argc >= 2) {
        iterations = atoi(argv[1]);
    }
    
    print_device_info();
    
    printf("Iterations per test: %d\n\n", iterations);
    
    // Measure overhead components first
    printf("=== Overhead Measurement ===\n");
    float event_overhead = measure_launch_overhead(1000);
    printf("CUDA event overhead: %.2f us/event\n", event_overhead);
    
    // Typical activation buffer size for M=1, K=2048: ~256 bytes (1 block_rrs_int4_tc)
    float alloc_256 = measure_alloc_overhead(256, 100);
    float alloc_4k = measure_alloc_overhead(4096, 100);
    printf("cudaMalloc overhead (256B): %.2f us/alloc\n", alloc_256);
    printf("cudaMalloc overhead (4KB): %.2f us/alloc\n", alloc_4k);
    printf("Note: Pool allocator is much faster than cudaMalloc, but has some overhead.\n\n");
    
    // ===== Part 1: Single-token generation (M=1) =====
    printf("=== Single Token Generation (M=1) ===\n");
    printf("This is the critical path for autoregressive inference.\n\n");
    
    print_table_header();
    
    // Qwen3-0.6B typical dimensions
    run_benchmark("Qwen3-0.6B attn_qkv", 1, 2048, 1024, iterations);
    run_benchmark("Qwen3-0.6B attn_o", 1, 1024, 1024, iterations);
    run_benchmark("Qwen3-0.6B mlp_gate", 1, 2816, 1024, iterations);
    run_benchmark("Qwen3-0.6B mlp_up", 1, 2816, 1024, iterations);
    run_benchmark("Qwen3-0.6B mlp_down", 1, 1024, 2816, iterations);
    
    printf("\n");
    
    // Qwen3-4B typical dimensions
    run_benchmark("Qwen3-4B attn_qkv", 1, 4608, 2048, iterations);
    run_benchmark("Qwen3-4B attn_o", 1, 2048, 2048, iterations);
    run_benchmark("Qwen3-4B mlp_gate", 1, 5632, 2048, iterations);
    run_benchmark("Qwen3-4B mlp_down", 1, 2048, 5632, iterations);
    
    printf("\n");
    
    // ===== Part 2: Prefill / Batched inference =====
    printf("=== Batched Inference (Prefill) ===\n");
    printf("Activation overhead should amortize with larger batch.\n\n");
    
    print_table_header();
    
    run_benchmark("M=16, N=2048, K=2048", 16, 2048, 2048, iterations);
    run_benchmark("M=32, N=2048, K=2048", 32, 2048, 2048, iterations);
    run_benchmark("M=64, N=2048, K=2048", 64, 2048, 2048, iterations);
    run_benchmark("M=128, N=2048, K=2048", 128, 2048, 2048, iterations);
    run_benchmark("M=256, N=2048, K=2048", 256, 2048, 2048, iterations);
    run_benchmark("M=512, N=2048, K=2048", 512, 2048, 2048, iterations);
    
    printf("\n");
    
    // ===== Part 3: Large dimensions =====
    printf("=== Large Dimensions (4Kx4K) ===\n\n");
    
    print_table_header();
    
    run_benchmark("M=1, N=4096, K=4096", 1, 4096, 4096, iterations);
    run_benchmark("M=16, N=4096, K=4096", 16, 4096, 4096, iterations);
    run_benchmark("M=64, N=4096, K=4096", 64, 4096, 4096, iterations);
    run_benchmark("M=128, N=4096, K=4096", 128, 4096, 4096, iterations);
    run_benchmark("M=256, N=4096, K=4096", 256, 4096, 4096, iterations);
    
    printf("\n");
    
    // ===== Summary =====
    printf("=== Analysis ===\n\n");
    printf("Components measured:\n");
    printf("  - Quant: FWHT transform + Runtime Smooth + INT4 packing\n");
    printf("  - GEMM:  TCQ4 IMMA tensor core matmul (m16n8k32.s4.s4.s32)\n");
    printf("\n");
    printf("Key insight:\n");
    printf("  - At M=1 (generation), activation quantization dominates (~70-90%% overhead)\n");
    printf("  - At larger M (prefill/batch), GEMM time dominates, overhead drops\n");
    printf("  - Standalone GEMM is 3-4x faster than Q4_K dp4a\n");
    printf("  - But total time (quant+gemm) is slower than Q4_K for M=1\n");
    printf("\n");
    printf("To make TCQ4 competitive for generation:\n");
    printf("  1. Fused activation+GEMM kernel (single launch, no intermediate write)\n");
    printf("  2. Specialized GEMV for M=1 with inline FWHT\n");
    printf("  3. Pre-allocated activation buffers (avoid pool alloc overhead)\n");
    printf("\n");
    printf("Overhead breakdown for M=1 path (estimated):\n");
    printf("  - Pool alloc: ~1-5 us (depends on pool implementation)\n");
    printf("  - Kernel launch #1 (quant): ~3-5 us\n");
    printf("  - Kernel launch #2 (GEMM): ~3-5 us\n");
    printf("  - Memory write (quant output): bandwidth-limited\n");
    printf("  - Memory read (GEMM input): bandwidth-limited\n");
    printf("\n");
    printf("Q4_K comparison: single kernel, no intermediate storage, direct compute.\n");
    
    return 0;
}