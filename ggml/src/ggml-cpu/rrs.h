#pragma once
#include "ggml.h"
#ifdef __cplusplus
extern "C" {
#endif
void ggml_fwht_impl(float * data, int n);
void ggml_quantize_row_q4_0_rrs_act(const float * x, void * y, int64_t k);
void ggml_vec_dot_q4_0_rrs_q4_0_rrs(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_rrs_free_scratch(void);
#ifdef __cplusplus
}
#endif
