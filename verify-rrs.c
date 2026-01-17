#include "ggml/src/ggml-cpu/rrs.h"
#include "ggml.h"
#include <stdio.h>

// Dummy implementations for standalone testing
void quantize_row_q4_0(const float * x, void * y, int64_t k) {
    (void)x; (void)y; (void)k;
}
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    (void)type; (void)ne;
    return 0;
}
float ggml_fp16_to_fp32(ggml_fp16_t x) {
    (void)x;
    return 0.0f;
}

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple test to verify the FWHT implementation
// 1. FWHT(FWHT(x)) should be x (self-inverse property)
// 2. FWHT should preserve L2 norm (orthogonality)

static void print_vec(const char * label, const float * data, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++) {
        printf("%s%.4f", i == 0 ? "" : ", ", data[i]);
    }
    printf("]\n");
}

static float l2_norm(const float * data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (double)data[i] * data[i];
    }
    return (float)sqrt(sum);
}

int main(void) {
    const int n = 8;
    float x[8];
    float y[8];
    float z[8];

    // Initialize with a simple signal
    for (int i = 0; i < n; i++) {
        x[i] = (float)sin(2.0 * M_PI * i / n) + 0.5f;
    }

    memcpy(y, x, sizeof(x));
    
    printf("Testing FWHT with n=%d\n", n);
    print_vec("Input ", x, n);

    float norm_x = l2_norm(x, n);
    printf("L2 Norm (input): %.6f\n", norm_x);

    // First transform
    ggml_fwht_impl(y, n);
    print_vec("FWHT  ", y, n);
    
    float norm_y = l2_norm(y, n);
    printf("L2 Norm (FWHT):  %.6f\n", norm_y);

    // Check orthogonality (conservation of energy)
    if (fabsf(norm_x - norm_y) > 1e-5f) {
        printf("FAILED: L2 norm changed from %.6f to %.6f\n", norm_x, norm_y);
        return 1;
    }

    // Second transform (should return to original)
    memcpy(z, y, sizeof(y));
    ggml_fwht_impl(z, n);
    print_vec("IFWHT ", z, n);

    for (int i = 0; i < n; i++) {
        if (fabsf(x[i] - z[i]) > 1e-5f) {
            printf("FAILED: Reconstruction error at index %d: original=%.6f, reconstructed=%.6f\n", i, x[i], z[i]);
            return 1;
        }
    }

    printf("SUCCESS: FWHT is self-inverse and preserves norm.\n");
    return 0;
}