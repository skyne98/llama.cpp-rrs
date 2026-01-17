#include "rrs.h"
#include "quants.h"
#include "ggml-impl.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

void ggml_fwht_impl(float * data, int n) {
    if (n <= 1) return;

    for (int h = 1; h < n; h <<= 1) {
#if defined(__AVX2__)
        if (h >= 8) {
            for (int i = 0; i < n; i += h << 1) {
                for (int j = i; j < i + h; j += 8) {
                    const __m256 x = _mm256_loadu_ps(data + j);
                    const __m256 y = _mm256_loadu_ps(data + j + h);

                    _mm256_storeu_ps(data + j,     _mm256_add_ps(x, y));
                    _mm256_storeu_ps(data + j + h, _mm256_sub_ps(x, y));
                }
            }
            continue;
        }
#endif
        for (int i = 0; i < n; i += h << 1) {
            for (int j = i; j < i + h; j++) {
                const float x = data[j];
                const float y = data[j + h];

                data[j]     = x + y;
                data[j + h] = x - y;
            }
        }
    }

    const float scale = 1.0f / sqrtf((float)n);

    int i = 0;
#if defined(__AVX2__)
    const __m256 v_scale = _mm256_set1_ps(scale);
    for (; i <= n - 8; i += 8) {
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), v_scale));
    }
#endif
    for (; i < n; i++) {
        data[i] *= scale;
    }
}

static __thread float * rrs_scratch = NULL;
static __thread size_t rrs_scratch_size = 0;

void ggml_rrs_free_scratch(void) {
    if (rrs_scratch) { free(rrs_scratch); rrs_scratch = NULL; rrs_scratch_size = 0; }
}

void ggml_quantize_row_q4_K_rrs_act(const float * x, void * y, int64_t k) {
    if (k <= 0 || (k & (k - 1)) != 0) {
        quantize_row_q4_K(x, y, k);
        return;
    }
    if (rrs_scratch_size < (size_t)k) {
        if (rrs_scratch) free(rrs_scratch);
        rrs_scratch = (float *)malloc(k * sizeof(float));
        if (!rrs_scratch) { quantize_row_q4_K(x, y, k); return; }
        rrs_scratch_size = (size_t)k;
    }
    memcpy(rrs_scratch, x, k * sizeof(float));
    ggml_fwht_impl(rrs_scratch, (int)k);
    quantize_row_q4_K(rrs_scratch, y, k);
}

size_t quantize_q4_K_rrs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS, n_per_row);
    char * qrow = (char *)dst;
    
    const bool is_power_of_2 = (n_per_row > 0) && ((n_per_row & (n_per_row - 1)) == 0);
    
    for (int64_t row = 0; row < nrow; ++row) {
        if (is_power_of_2) {
            if (rrs_scratch_size < (size_t)n_per_row) {
                if (rrs_scratch) free(rrs_scratch);
                rrs_scratch = (float *)malloc(n_per_row * sizeof(float));
                if (!rrs_scratch) {
                    quantize_row_q4_K(src, qrow, n_per_row);
                    src += n_per_row;
                    qrow += row_size;
                    continue;
                }
                rrs_scratch_size = (size_t)n_per_row;
            }
            memcpy(rrs_scratch, src, n_per_row * sizeof(float));
            ggml_fwht_impl(rrs_scratch, (int)n_per_row);
            quantize_row_q4_K(rrs_scratch, qrow, n_per_row);
        } else {
            quantize_row_q4_K(src, qrow, n_per_row);
        }
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

extern void dequantize_row_q4_K(const void * vx, float * y, int64_t k);

void ggml_vec_dot_q4_K_rrs_q4_K_rrs(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const int nb = n / QK_K;
    const block_q4_K * x = (const block_q4_K *)vx;
    const block_q4_K * y = (const block_q4_K *)vy;

    float sumf = 0.0;
    float temp_x[QK_K];
    float temp_y[QK_K];

    for (int i = 0; i < nb; i++) {
        dequantize_row_q4_K(&x[i], temp_x, QK_K);
        dequantize_row_q4_K(&y[i], temp_y, QK_K);
        for (int j = 0; j < QK_K; j++) {
            sumf += temp_x[j] * temp_y[j];
        }
    }
    *s = sumf;
}