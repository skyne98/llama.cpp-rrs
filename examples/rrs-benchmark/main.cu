// RRS CUDA Benchmark Tool
// Compares INT4 WMMA tensor core path vs Q8 repack + dp4a path
// Usage: ./llama-rrs-benchmark [M] [N] [K] [iterations]

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

static void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    printf("=== RRS CUDA Benchmark ===\n");
    printf("Device: %s (SM%d%d)\n", props.name, props.major, props.minor);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    
    // Check INT4 tensor core support
    bool has_int4_tc = (props.major > 7) || (props.major == 7 && props.minor >= 5);
    printf("INT4 Tensor Core Support: %s\n", has_int4_tc ? "Yes" : "No");
    
    if (!has_int4_tc) {
        printf("WARNING: This device does not support INT4 tensor cores (requires SM75+)\n");
        printf("         The INT4 WMMA path will not be available.\n");
    }
    
    // Theoretical peak INT4 TOPS
    // RTX 3090: 142 TFLOPS FP16, INT4 is ~4x = ~568 TOPS
    float clock_ghz = props.clockRate / 1e6f;
    int sm_count = props.multiProcessorCount;
    
    printf("SM Count: %d\n", sm_count);
    printf("Clock: %.2f GHz\n", clock_ghz);
    printf("\n");
}

static void run_benchmarks(int M, int N, int K, int iterations) {
    printf("Matrix Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Iterations: %d\n", iterations);
    printf("\n");
    
    RRSBenchmarkResult result;
    ggml_cuda_rrs_benchmark(M, N, K, iterations, &result);
    ggml_cuda_rrs_print_benchmark(&result);
}

int main(int argc, char** argv) {
    // Default dimensions (typical for small LLM layer)
    int M = 32;     // batch size / tokens
    int N = 2048;   // output features
    int K = 2048;   // input features  
    int iterations = 100;
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        iterations = atoi(argv[4]);
    }
    
    // Validate dimensions (must be multiples of 32 for Q4 quantization)
    if (K % 32 != 0) {
        fprintf(stderr, "Error: K must be a multiple of 32 (got %d)\n", K);
        return 1;
    }
    if (N % 32 != 0) {
        fprintf(stderr, "Error: N must be a multiple of 32 (got %d)\n", N);
        return 1;
    }
    
    // Print device info
    print_device_info();
    
    // Run benchmarks with different configurations
    printf("=== Benchmark 1: Small Batch (typical inference) ===\n");
    run_benchmarks(M, N, K, iterations);
    printf("\n");
    
    // Larger batch for throughput testing
    if (M < 128) {
        printf("=== Benchmark 2: Larger Batch (M=128) ===\n");
        run_benchmarks(128, N, K, iterations);
        printf("\n");
    }
    
    // Different K sizes (common hidden dimensions)
    if (K == 2048) {
        printf("=== Benchmark 3: K=4096 (larger hidden dim) ===\n");
        run_benchmarks(M, N, 4096, iterations);
        printf("\n");
    }
    
    printf("=== Summary ===\n");
    printf("The benchmark compares two approaches for RRS W4A4 GEMM:\n");
    printf("1. INT4 WMMA: Direct INT4 tensor core path (SM75+ only)\n");
    printf("2. Q8 Repack: Convert Q4->Q8 and use dp4a (all CUDA devices)\n");
    printf("\n");
    printf("For production use:\n");
    printf("- Use INT4 WMMA on Turing+ GPUs (RTX 20xx, 30xx, 40xx)\n");
    printf("- Use Q8 Repack as fallback on older GPUs\n");
    
    return 0;
}