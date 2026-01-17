#pragma once
#include "ggml.h"
#ifdef __cplusplus
extern "C" {
#endif
void ggml_fwht_impl(float * data, int n);
void ggml_quantize_row_q4_K_rrs_act(const float * x, void * y, int64_t k);
GGML_API size_t quantize_q4_K_rrs(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
void ggml_vec_dot_q4_K_rrs_q4_K_rrs(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_rrs_free_scratch(void);
#ifdef __cplusplus
}
#endif
