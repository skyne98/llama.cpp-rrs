#include "rrs.h"
#include "quants.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void ggml_fwht_impl(float * data, int n) {
    if (n <= 1) return;
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float x = data[j];
                float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        data[i] *= scale;
    }
}

static __thread float * rrs_scratch = NULL;
static __thread size_t rrs_scratch_size = 0;

void ggml_rrs_free_scratch(void) {
    if (rrs_scratch) { free(rrs_scratch); rrs_scratch = NULL; rrs_scratch_size = 0; }
}

void ggml_quantize_row_q4_0_rrs_act(const float * x, void * y, int64_t k) {
    if (k <= 0 || (k & (k - 1)) != 0) { quantize_row_q4_0(x, y, k); return; }
    if (rrs_scratch_size < (size_t)k) {
        if (rrs_scratch) free(rrs_scratch);
        rrs_scratch = (float *)malloc(k * sizeof(float));
        if (!rrs_scratch) { quantize_row_q4_0(x, y, k); return; }
        rrs_scratch_size = (size_t)k;
    }
    memcpy(rrs_scratch, x, k * sizeof(float));
    ggml_fwht_impl(rrs_scratch, (int)k);
    quantize_row_q4_0(rrs_scratch, y, k);
}

void ggml_vec_dot_q4_0_rrs_q4_0_rrs(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const int nb = n / 32;
    const block_q4_0 * x = (const block_q4_0 *)vx;
    const block_q4_0 * y = (const block_q4_0 *)vy;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d_x = ggml_fp16_to_fp32(x[i].d);
        float d_y = ggml_fp16_to_fp32(y[i].d);
        int sum_i = 0;
        for (int j = 0; j < 16; ++j) {
            const int8_t x0 = (x[i].qs[j] & 0x0F) - 8;
            const int8_t x1 = (x[i].qs[j] >> 4) - 8;
            const int8_t y0 = (y[i].qs[j] & 0x0F) - 8;
            const int8_t y1 = (y[i].qs[j] >> 4) - 8;
            sum_i += x0 * y0 + x1 * y1;
        }
        sumf += sum_i * d_x * d_y;
    }
    *s = sumf;
}
