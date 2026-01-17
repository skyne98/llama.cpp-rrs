#include "rrs.h"
#include "quants.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Micro-benchmark for RRS W4A4 dot product kernel performance tuning.
// Compares current RRS Q4_K x Q4_K dot product against native Q4_K x Q8_K.

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void fill_random(float * x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

int main(int argc, char ** argv) {
    int n = 4096;
    int iterations = 100000;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    printf("Benchmarking dot product with n=%d, iterations=%d\n", n, iterations);

    float * x_f = malloc(n * sizeof(float));
    float * y_f = malloc(n * sizeof(float));
    fill_random(x_f, n);
    fill_random(y_f, n);

    // Prepare Q4_K blocks (RRS style, weights and activations)
    void * x_q4 = malloc(ggml_row_size(GGML_TYPE_Q4_K, n));
    void * y_q4 = malloc(ggml_row_size(GGML_TYPE_Q4_K, n));
    quantize_row_q4_K(x_f, x_q4, n);
    quantize_row_q4_K(y_f, y_q4, n);

    // Prepare Q8_K blocks (Native style, activations)
    void * y_q8 = malloc(ggml_row_size(GGML_TYPE_Q8_K, n));
    quantize_row_q8_K(y_f, y_q8, n);

    float result = 0;

    // 1. Benchmark Native Q4_K x Q8_K (The gold standard for llama.cpp CPU)
    double t0 = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_q8_K(n, &result, 0, x_q4, 0, y_q8, 0, 1);
    }
    double t1 = get_time_ms();
    printf("Native Q4_K x Q8_K:  %8.2f ms | Result: %f\n", t1 - t0, result);

    // 2. Benchmark RRS Q4_K x Q4_K (Current implementation)
    double t2 = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_rrs_q4_K_rrs(n, &result, 0, x_q4, 0, y_q4, 0, 1);
    }
    double t3 = get_time_ms();
    printf("RRS Q4_K x Q4_K:     %8.2f ms | Result: %f\n", t3 - t2, result);

    // Validation (Numerical Check)
    float dot_f = 0;
    for (int i = 0; i < n; i++) dot_f += x_f[i] * y_f[i];
    printf("Reference (Float):            %f\n", dot_f);

    free(x_f);
    free(y_f);
    free(x_q4);
    free(y_q4);
    free(y_q8);

    return 0;
}

// Dummy symbols for linking if necessary
#include "ggml-cpu-impl.h"
#include "traits.h"

// We need to link with quants.c, ggml.c, rrs.c, etc.
// Building this:
// gcc -O3 -mavx2 -mfma -I. -I../../include -I../.. rrs_bench.c rrs.c quants.c ../../ggml.c -lm -lpthread -o rrs_bench