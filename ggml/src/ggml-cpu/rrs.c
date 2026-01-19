#include "rrs.h"
#include "quants.h"
#include "ggml-impl.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#define RRS_AVX512_VNNI 1
#define RRS_AVX512 1
#elif defined(__AVX512F__)
#include <immintrin.h>
#define RRS_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define RRS_AVX2 1
#endif

#define RRS_ALIGN 64

static inline void * rrs_aligned_alloc(size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, RRS_ALIGN);
#else
    void * ptr = NULL;
    if (posix_memalign(&ptr, RRS_ALIGN, size) != 0) return NULL;
    return ptr;
#endif
}

static inline void rrs_aligned_free(void * ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void ggml_fwht_impl(float * data, int n) {
    if (n <= 1) return;

    const float scale = 1.0f / sqrtf((float)n);
    const int last_h = n >> 1;

    for (int h = 1; h < n; h <<= 1) {
        const int is_last = (h == last_h);
        const int stride = h << 1;

#if defined(RRS_AVX512)
        if (h >= 16) {
            if (is_last) {
                const __m512 v_scale = _mm512_set1_ps(scale);
                for (int i = 0; i < n; i += stride) {
                    for (int j = i; j < i + h; j += 16) {
                        const __m512 x = _mm512_load_ps(data + j);
                        const __m512 y = _mm512_load_ps(data + j + h);
                        _mm512_stream_ps(data + j,     _mm512_mul_ps(_mm512_add_ps(x, y), v_scale));
                        _mm512_stream_ps(data + j + h, _mm512_mul_ps(_mm512_sub_ps(x, y), v_scale));
                    }
                }
                _mm_sfence();
            } else {
                for (int i = 0; i < n; i += stride) {
                    for (int j = i; j < i + h; j += 16) {
                        const __m512 x = _mm512_load_ps(data + j);
                        const __m512 y = _mm512_load_ps(data + j + h);
                        _mm512_store_ps(data + j,     _mm512_add_ps(x, y));
                        _mm512_store_ps(data + j + h, _mm512_sub_ps(x, y));
                    }
                }
            }
            continue;
        }
#endif
#if defined(RRS_AVX2) || defined(RRS_AVX512)
        if (h >= 8) {
            if (is_last) {
                const __m256 v_scale = _mm256_set1_ps(scale);
                for (int i = 0; i < n; i += stride) {
                    for (int j = i; j < i + h; j += 8) {
                        const __m256 x = _mm256_load_ps(data + j);
                        const __m256 y = _mm256_load_ps(data + j + h);
                        _mm256_stream_ps(data + j,     _mm256_mul_ps(_mm256_add_ps(x, y), v_scale));
                        _mm256_stream_ps(data + j + h, _mm256_mul_ps(_mm256_sub_ps(x, y), v_scale));
                    }
                }
                _mm_sfence();
            } else {
                for (int i = 0; i < n; i += stride) {
                    for (int j = i; j < i + h; j += 8) {
                        const __m256 x = _mm256_load_ps(data + j);
                        const __m256 y = _mm256_load_ps(data + j + h);
                        _mm256_store_ps(data + j,     _mm256_add_ps(x, y));
                        _mm256_store_ps(data + j + h, _mm256_sub_ps(x, y));
                    }
                }
            }
            continue;
        }
#endif
        if (is_last) {
            for (int i = 0; i < n; i += stride) {
                for (int j = i; j < i + h; j++) {
                    const float x = data[j];
                    const float y = data[j + h];
                    data[j]     = (x + y) * scale;
                    data[j + h] = (x - y) * scale;
                }
            }
        } else {
            for (int i = 0; i < n; i += stride) {
                for (int j = i; j < i + h; j++) {
                    const float x = data[j];
                    const float y = data[j + h];
                    data[j]     = x + y;
                    data[j + h] = x - y;
                }
            }
        }
    }
}

static __thread float * rrs_scratch = NULL;
static __thread size_t rrs_scratch_size = 0;

void ggml_rrs_free_scratch(void) {
    if (rrs_scratch) {
        rrs_aligned_free(rrs_scratch);
        rrs_scratch = NULL;
        rrs_scratch_size = 0;
    }
}

static inline float * rrs_ensure_scratch(size_t k) {
    if (rrs_scratch_size < k) {
        if (rrs_scratch) rrs_aligned_free(rrs_scratch);
        const size_t alloc_size = ((k * sizeof(float) + RRS_ALIGN - 1) / RRS_ALIGN) * RRS_ALIGN;
        rrs_scratch = (float *)rrs_aligned_alloc(alloc_size);
        rrs_scratch_size = rrs_scratch ? k : 0;
    }
    return rrs_scratch;
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// Fast activation quantization optimized for FWHT-transformed data
// FWHT produces symmetric distributions centered at 0, so we use symmetric quantization
// with optimal clipping to handle outliers efficiently

// Symmetric quantization for FWHT data: quantize to [-7, 7] then shift to [0, 15]
// This is more efficient for symmetric data than asymmetric min/max
static float make_symmetric_quants_fast(int n, int nmax, const float * GGML_RESTRICT x,
        const float * GGML_RESTRICT weights, uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min) {
    // Find optimal clipping threshold using percentile-based approach
    // For FWHT data, clip at ~2.5 sigma to minimize MSE
    float sum_w = 0, sum_x2 = 0;
    float abs_max = 0;
    
    for (int i = 0; i < n; ++i) {
        sum_w += weights[i];
        sum_x2 += weights[i] * x[i] * x[i];
        float ax = fabsf(x[i]);
        if (ax > abs_max) abs_max = ax;
    }
    
    if (abs_max == 0) {
        for (int i = 0; i < n; ++i) L[i] = nmax / 2;  // midpoint for zero
        *the_min = 0;
        return 0.f;
    }
    
    // Estimate std dev and use optimal clipping ratio for 4-bit
    // Optimal clip for Gaussian at 4-bit is ~2.55 sigma
    float variance = sum_x2 / sum_w;
    float sigma = sqrtf(variance);
    float clip = 2.55f * sigma;
    if (clip > abs_max) clip = abs_max;
    if (clip < abs_max * 0.5f) clip = abs_max * 0.5f;  // Don't clip too aggressively
    
    // Try multiple clipping ratios and pick best
    float best_error = 1e30f;
    float best_scale = clip / (nmax / 2);
    uint8_t Lbest[32];
    
    for (int trial = 0; trial < 5; ++trial) {
        float try_clip = clip * (0.8f + 0.1f * trial);
        float try_scale = try_clip / (nmax / 2);
        float try_iscale = (nmax / 2) / try_clip;
        
        float cur_error = 0;
        for (int i = 0; i < n; ++i) {
            // Symmetric quantize: map [-clip, clip] to [0, nmax]
            int l = (int)(x[i] * try_iscale + (nmax / 2) + 0.5f);
            l = l < 0 ? 0 : (l > nmax ? nmax : l);
            Lbest[i] = (uint8_t)l;
            float reconstructed = (l - nmax / 2) * try_scale;
            float diff = reconstructed - x[i];
            cur_error += weights[i] * diff * diff;
        }
        
        if (cur_error < best_error) {
            best_error = cur_error;
            best_scale = try_scale;
            for (int i = 0; i < n; ++i) L[i] = Lbest[i];
        }
    }
    
    // For symmetric quant, min = -scale * (nmax/2), so the_min = scale * (nmax/2)
    *the_min = best_scale * (nmax / 2);
    return best_scale;
}

// MSE-minimizing quantization for a group of 32 values with importance weights
// Returns optimal scale, sets *the_min to optimal min (negated)
static float make_qkx2_quants_fast(int n, int nmax, const float * GGML_RESTRICT x,
        const float * GGML_RESTRICT weights, uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min) {
    float min_val = x[0], max_val = x[0];
    float sum_w = weights[0], sum_x = weights[0] * x[0];
    
    for (int i = 1; i < n; ++i) {
        if (x[i] < min_val) min_val = x[i];
        if (x[i] > max_val) max_val = x[i];
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if (min_val > 0) min_val = 0;
    if (max_val == min_val) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min_val;
        return 0.f;
    }
    
    float iscale = nmax / (max_val - min_val);
    float scale = 1 / iscale;
    float best_error = 0;
    
    // Initial quantization
    for (int i = 0; i < n; ++i) {
        int l = (int)(iscale * (x[i] - min_val) + 0.5f);
        L[i] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        float diff = scale * L[i] + min_val - x[i];
        best_error += weights[i] * diff * diff;
    }
    
    // Iterative refinement (3 steps for speed vs quality tradeoff)
    uint8_t Laux[32];
    for (int is = 0; is <= 6; ++is) {
        float try_iscale = (-0.6f + 0.2f * is + nmax) / (max_val - min_val);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        
        for (int i = 0; i < n; ++i) {
            int l = (int)(try_iscale * (x[i] - min_val) + 0.5f);
            l = l < 0 ? 0 : (l > nmax ? nmax : l);
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                cur_error += weights[i] * diff * diff;
            }
            
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min_val = this_min;
            }
        }
    }
    
    *the_min = -min_val;
    return scale;
}

// Quantization optimized for FWHT-transformed activations
// Uses symmetric quantization with optimal clipping for better quality
static void quantize_row_q4_K_fwht(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    block_q4_K * GGML_RESTRICT y = (block_q4_K *)vy;

    uint8_t L[QK_K];
    float weights[32];
    float mins[8], scales[8];

    for (int i = 0; i < nb; i++) {
        const float * xb = x + i * QK_K;
        float max_scale = 0, max_min = 0;
        
        // Process 8 groups of 32 elements each with symmetric quantization
        for (int j = 0; j < 8; j++) {
            const float * xg = xb + j * 32;
            
            // Compute importance weights: av_x + |x[i]|
            float sum_x2 = 0;
            for (int l = 0; l < 32; l++) sum_x2 += xg[l] * xg[l];
            float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(xg[l]);
            
            // Use MSE-minimizing asymmetric quantization
            scales[j] = make_qkx2_quants_fast(32, 15, xg, weights, L + j * 32, &mins[j]);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        
        // Encode scales/mins into block header
        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        
        for (int j = 0; j < 8; j++) {
            uint8_t ls = (uint8_t)(inv_scale * scales[j] + 0.5f);
            uint8_t lm = (uint8_t)(inv_min * mins[j] + 0.5f);
            if (ls > 63) ls = 63;
            if (lm > 63) lm = 63;
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j] |= ((lm >> 4) << 6);
            }
        }
        
        y[i].d = GGML_FP32_TO_FP16(max_scale / 63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min / 63.f);
        
        // Re-quantize with final encoded scales for accuracy
        for (int j = 0; j < 8; j++) {
            const float * xg = xb + j * 32;
            uint8_t sc, mn;
            get_scale_min_k4(j, y[i].scales, &sc, &mn);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * mn;
            if (d == 0) {
                for (int l = 0; l < 32; l++) L[j * 32 + l] = 0;
                continue;
            }
            const float id = 1.0f / d;
            for (int l = 0; l < 32; l++) {
                int q4 = (int)((xg[l] + dm) * id + 0.5f);
                L[j * 32 + l] = (uint8_t)(q4 < 0 ? 0 : (q4 > 15 ? 15 : q4));
            }
        }
        
        // Pack into qs: each 64 values -> 32 bytes (low nibble + high nibble)
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; l++) {
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            q += 32;
        }
    }
}

// Standard asymmetric quantization (for non-FWHT data)
static void quantize_row_q4_K_fast(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    block_q4_K * GGML_RESTRICT y = (block_q4_K *)vy;

    uint8_t L[QK_K];
    float weights[32];
    float mins[8], scales[8];

    for (int i = 0; i < nb; i++) {
        const float * xb = x + i * QK_K;
        float max_scale = 0, max_min = 0;
        
        for (int j = 0; j < 8; j++) {
            const float * xg = xb + j * 32;
            float sum_x2 = 0;
            for (int l = 0; l < 32; l++) sum_x2 += xg[l] * xg[l];
            float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(xg[l]);
            
            scales[j] = make_qkx2_quants_fast(32, 15, xg, weights, L + j * 32, &mins[j]);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        
        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        
        for (int j = 0; j < 8; j++) {
            uint8_t ls = (uint8_t)(inv_scale * scales[j] + 0.5f);
            uint8_t lm = (uint8_t)(inv_min * mins[j] + 0.5f);
            if (ls > 63) ls = 63;
            if (lm > 63) lm = 63;
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j] |= ((lm >> 4) << 6);
            }
        }
        
        y[i].d = GGML_FP32_TO_FP16(max_scale / 63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min / 63.f);
        
        for (int j = 0; j < 8; j++) {
            const float * xg = xb + j * 32;
            uint8_t sc, mn;
            get_scale_min_k4(j, y[i].scales, &sc, &mn);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * mn;
            if (d == 0) {
                for (int l = 0; l < 32; l++) L[j * 32 + l] = 0;
                continue;
            }
            const float id = 1.0f / d;
            for (int l = 0; l < 32; l++) {
                int q4 = (int)((xg[l] + dm) * id + 0.5f);
                L[j * 32 + l] = (uint8_t)(q4 < 0 ? 0 : (q4 > 15 ? 15 : q4));
            }
        }
        
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; l++) {
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            q += 32;
        }
    }
}

void ggml_quantize_row_q4_K_rrs_act(const float * x, void * y, int64_t k) {
    if (k <= 0) return;

    float * scratch = rrs_ensure_scratch((size_t)k);
    if (!scratch) {
        quantize_row_q4_K(x, y, k);
        return;
    }
    memcpy(scratch, x, k * sizeof(float));

    const int step = (int)(k & -k);
    for (int i = 0; i < k; i += step) {
        ggml_fwht_impl(scratch + i, step);
    }

    quantize_row_q4_K_fwht(scratch, y, k);
}

GGML_API size_t quantize_q4_K_rrs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K_RRS, n_per_row);
    char * qrow = (char *)dst;

    const int step = (int)(n_per_row & -n_per_row);

    for (int64_t row = 0; row < nrow; ++row) {
        float * scratch = rrs_ensure_scratch((size_t)n_per_row);
        if (!scratch) {
            quantize_row_q4_K(src, qrow, n_per_row);
        } else {
            memcpy(scratch, src, n_per_row * sizeof(float));
            for (int i = 0; i < n_per_row; i += step) {
                ggml_fwht_impl(scratch + i, step);
            }
            // Use standard high-quality quantizer for weights
            quantize_row_q4_K(scratch, qrow, n_per_row);
        }
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

extern void dequantize_row_q4_K(const void * vx, float * y, int64_t k);



#if defined(RRS_AVX512)

static inline int32_t hsum_i32_16(__m512i v) {
    __m256i lo = _mm512_castsi512_si256(v);
    __m256i hi = _mm512_extracti32x8_epi32(v, 1);
    __m256i sum256 = _mm256_add_epi32(lo, hi);
    __m128i lo128 = _mm256_castsi256_si128(sum256);
    __m128i hi128 = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum128);
}

static inline void unpack_scales_mins_k4(const uint8_t * scales, uint8_t * sc, uint8_t * mn) {
    sc[0] = scales[0] & 63; mn[0] = scales[4] & 63;
    sc[1] = scales[1] & 63; mn[1] = scales[5] & 63;
    sc[2] = scales[2] & 63; mn[2] = scales[6] & 63;
    sc[3] = scales[3] & 63; mn[3] = scales[7] & 63;
    sc[4] = (scales[8] & 0xF) | ((scales[0] >> 6) << 4);
    mn[4] = (scales[8] >> 4)  | ((scales[4] >> 6) << 4);
    sc[5] = (scales[9] & 0xF) | ((scales[1] >> 6) << 4);
    mn[5] = (scales[9] >> 4)  | ((scales[5] >> 6) << 4);
    sc[6] = (scales[10] & 0xF) | ((scales[2] >> 6) << 4);
    mn[6] = (scales[10] >> 4)  | ((scales[6] >> 6) << 4);
    sc[7] = (scales[11] & 0xF) | ((scales[3] >> 6) << 4);
    mn[7] = (scales[11] >> 4)  | ((scales[7] >> 6) << 4);
}

void ggml_vec_dot_q4_K_rrs_q4_K_rrs(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;

    const int nb = n / QK_K;
    const block_q4_K * x = (const block_q4_K *)vx;
    const block_q4_K * y = (const block_q4_K *)vy;

    float sumf = 0.0f;

#if defined(RRS_AVX512_VNNI)
    // VNNI-optimized path: process all 4 chunks (128 bytes) per block
    // Use VNNI for dot products, vectorize scale application
    
    const __m512i zeros = _mm512_setzero_si512();
    
    for (int i = 0; i < nb; i++) {
        _mm_prefetch((const char *)&x[i + 1], _MM_HINT_T0);
        _mm_prefetch((const char *)&y[i + 1], _MM_HINT_T0);

        const float dx   = GGML_FP16_TO_FP32(x[i].d);
        const float minx = GGML_FP16_TO_FP32(x[i].dmin);
        const float dy   = GGML_FP16_TO_FP32(y[i].d);
        const float miny = GGML_FP16_TO_FP32(y[i].dmin);

        uint8_t scx[8], mnx[8], scy[8], mny[8];
        unpack_scales_mins_k4(x[i].scales, scx, mnx);
        unpack_scales_mins_k4(y[i].scales, scy, mny);

        const uint8_t * qx = x[i].qs;
        const uint8_t * qy = y[i].qs;

        // Load all 128 bytes (4 chunks of 32 bytes each) at once
        const __m512i vqx_01 = _mm512_loadu_si512((const __m512i *)qx);        // chunks 0,1
        const __m512i vqx_23 = _mm512_loadu_si512((const __m512i *)(qx + 64)); // chunks 2,3
        const __m512i vqy_01 = _mm512_loadu_si512((const __m512i *)qy);
        const __m512i vqy_23 = _mm512_loadu_si512((const __m512i *)(qy + 64));
        
        // Unpack all chunks: lo nibbles (subgroups 0,2,4,6) and hi nibbles (subgroups 1,3,5,7)
        const __m512i low_mask = _mm512_set1_epi8(0x0F);
        const __m512i vqx_lo_01 = _mm512_and_si512(vqx_01, low_mask);
        const __m512i vqx_hi_01 = _mm512_and_si512(_mm512_srli_epi16(vqx_01, 4), low_mask);
        const __m512i vqx_lo_23 = _mm512_and_si512(vqx_23, low_mask);
        const __m512i vqx_hi_23 = _mm512_and_si512(_mm512_srli_epi16(vqx_23, 4), low_mask);
        
        const __m512i vqy_lo_01 = _mm512_and_si512(vqy_01, low_mask);
        const __m512i vqy_hi_01 = _mm512_and_si512(_mm512_srli_epi16(vqy_01, 4), low_mask);
        const __m512i vqy_lo_23 = _mm512_and_si512(vqy_23, low_mask);
        const __m512i vqy_hi_23 = _mm512_and_si512(_mm512_srli_epi16(vqy_23, 4), low_mask);
        
        // VNNI dot products: dpbusd produces 16 int32 results from 64 bytes
        // Each 32-byte half produces 8 int32 results that need summing
        __m512i dot_lo_01 = _mm512_dpbusd_epi32(zeros, vqx_lo_01, vqy_lo_01);
        __m512i dot_hi_01 = _mm512_dpbusd_epi32(zeros, vqx_hi_01, vqy_hi_01);
        __m512i dot_lo_23 = _mm512_dpbusd_epi32(zeros, vqx_lo_23, vqy_lo_23);
        __m512i dot_hi_23 = _mm512_dpbusd_epi32(zeros, vqx_hi_23, vqy_hi_23);
        
        // Sum bytes for min offset calculation using SAD
        __m512i sad_qx_lo_01 = _mm512_sad_epu8(vqx_lo_01, zeros);
        __m512i sad_qx_hi_01 = _mm512_sad_epu8(vqx_hi_01, zeros);
        __m512i sad_qx_lo_23 = _mm512_sad_epu8(vqx_lo_23, zeros);
        __m512i sad_qx_hi_23 = _mm512_sad_epu8(vqx_hi_23, zeros);
        
        __m512i sad_qy_lo_01 = _mm512_sad_epu8(vqy_lo_01, zeros);
        __m512i sad_qy_hi_01 = _mm512_sad_epu8(vqy_hi_01, zeros);
        __m512i sad_qy_lo_23 = _mm512_sad_epu8(vqy_lo_23, zeros);
        __m512i sad_qy_hi_23 = _mm512_sad_epu8(vqy_hi_23, zeros);
        
        // Extract per-subgroup results using horizontal adds
        // dot_lo_01 has 16 int32s: [8 for subgroup 0][8 for subgroup 2]
        // We need to sum each group of 8
        
        // For dot products: sum groups of 8 int32s
        // Use permute to bring halves together, then add
        __m256i dot_lo_01_lo = _mm512_castsi512_si256(dot_lo_01);
        __m256i dot_lo_01_hi = _mm512_extracti32x8_epi32(dot_lo_01, 1);
        __m256i dot_hi_01_lo = _mm512_castsi512_si256(dot_hi_01);
        __m256i dot_hi_01_hi = _mm512_extracti32x8_epi32(dot_hi_01, 1);
        __m256i dot_lo_23_lo = _mm512_castsi512_si256(dot_lo_23);
        __m256i dot_lo_23_hi = _mm512_extracti32x8_epi32(dot_lo_23, 1);
        __m256i dot_hi_23_lo = _mm512_castsi512_si256(dot_hi_23);
        __m256i dot_hi_23_hi = _mm512_extracti32x8_epi32(dot_hi_23, 1);
        
        // Horizontal sum each 256-bit chunk to single int32
        // Pack results: [sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7]
        int32_t dot_qq[8];
        
        // Subgroup 0 (lo nibbles of chunk 0)
        __m128i t = _mm_add_epi32(_mm256_castsi256_si128(dot_lo_01_lo), _mm256_extracti128_si256(dot_lo_01_lo, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[0] = _mm_cvtsi128_si32(t);
        
        // Subgroup 1 (hi nibbles of chunk 0)
        t = _mm_add_epi32(_mm256_castsi256_si128(dot_hi_01_lo), _mm256_extracti128_si256(dot_hi_01_lo, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[1] = _mm_cvtsi128_si32(t);
        
        // Subgroup 2 (lo nibbles of chunk 1)
        t = _mm_add_epi32(_mm256_castsi256_si128(dot_lo_01_hi), _mm256_extracti128_si256(dot_lo_01_hi, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[2] = _mm_cvtsi128_si32(t);
        
        // Subgroup 3 (hi nibbles of chunk 1)
        t = _mm_add_epi32(_mm256_castsi256_si128(dot_hi_01_hi), _mm256_extracti128_si256(dot_hi_01_hi, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[3] = _mm_cvtsi128_si32(t);
        
        // Subgroup 4 (lo nibbles of chunk 2)
        t = _mm_add_epi32(_mm256_castsi256_si128(dot_lo_23_lo), _mm256_extracti128_si256(dot_lo_23_lo, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[4] = _mm_cvtsi128_si32(t);
        
        // Subgroup 5 (hi nibbles of chunk 2)
        t = _mm_add_epi32(_mm256_castsi256_si128(dot_hi_23_lo), _mm256_extracti128_si256(dot_hi_23_lo, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[5] = _mm_cvtsi128_si32(t);
        
        // Subgroup 6 (lo nibbles of chunk 3)
        t = _mm_add_epi32(_mm256_castsi256_si128(dot_lo_23_hi), _mm256_extracti128_si256(dot_lo_23_hi, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[6] = _mm_cvtsi128_si32(t);
        
        // Subgroup 7 (hi nibbles of chunk 3)
        t = _mm_add_epi32(_mm256_castsi256_si128(dot_hi_23_hi), _mm256_extracti128_si256(dot_hi_23_hi, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1,0,3,2)));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2,3,0,1)));
        dot_qq[7] = _mm_cvtsi128_si32(t);
        
        // Extract SAD results (sum of bytes per subgroup)
        // SAD produces 8 uint64 results per 512-bit register (one per 64-bit lane)
        // Each 256-bit half has 4 results that need summing to get subgroup sum
        int32_t sum_qx[8], sum_qy[8];
        
        __m256i sad_lo = _mm512_castsi512_si256(sad_qx_lo_01);
        __m256i sad_hi = _mm512_extracti32x8_epi32(sad_qx_lo_01, 1);
        __m128i sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[0] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[2] = _mm_cvtsi128_si32(sv);
        
        sad_lo = _mm512_castsi512_si256(sad_qx_hi_01);
        sad_hi = _mm512_extracti32x8_epi32(sad_qx_hi_01, 1);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[1] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[3] = _mm_cvtsi128_si32(sv);
        
        sad_lo = _mm512_castsi512_si256(sad_qx_lo_23);
        sad_hi = _mm512_extracti32x8_epi32(sad_qx_lo_23, 1);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[4] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[6] = _mm_cvtsi128_si32(sv);
        
        sad_lo = _mm512_castsi512_si256(sad_qx_hi_23);
        sad_hi = _mm512_extracti32x8_epi32(sad_qx_hi_23, 1);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[5] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qx[7] = _mm_cvtsi128_si32(sv);
        
        // Same for qy
        sad_lo = _mm512_castsi512_si256(sad_qy_lo_01);
        sad_hi = _mm512_extracti32x8_epi32(sad_qy_lo_01, 1);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[0] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[2] = _mm_cvtsi128_si32(sv);
        
        sad_lo = _mm512_castsi512_si256(sad_qy_hi_01);
        sad_hi = _mm512_extracti32x8_epi32(sad_qy_hi_01, 1);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[1] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[3] = _mm_cvtsi128_si32(sv);
        
        sad_lo = _mm512_castsi512_si256(sad_qy_lo_23);
        sad_hi = _mm512_extracti32x8_epi32(sad_qy_lo_23, 1);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[4] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[6] = _mm_cvtsi128_si32(sv);
        
        sad_lo = _mm512_castsi512_si256(sad_qy_hi_23);
        sad_hi = _mm512_extracti32x8_epi32(sad_qy_hi_23, 1);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_lo), _mm256_extracti128_si256(sad_lo, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[5] = _mm_cvtsi128_si32(sv);
        sv = _mm_add_epi64(_mm256_castsi256_si128(sad_hi), _mm256_extracti128_si256(sad_hi, 1));
        sv = _mm_add_epi64(sv, _mm_shuffle_epi32(sv, _MM_SHUFFLE(1,0,3,2)));
        sum_qy[7] = _mm_cvtsi128_si32(sv);

        // Compute acc_mm
        int32_t acc_mm = 0;
        for (int j = 0; j < 8; j++) {
            acc_mm += (int32_t)mnx[j] * mny[j];
        }

        // Apply scales in scalar (small loop, well predicted)
        int32_t sum_qq_acc = 0, sum_qx_acc = 0, sum_qy_acc = 0;
        for (int j = 0; j < 8; j++) {
            sum_qq_acc += dot_qq[j] * (int32_t)scx[j] * scy[j];
            sum_qx_acc += sum_qx[j] * (int32_t)scx[j] * mny[j];
            sum_qy_acc += sum_qy[j] * (int32_t)scy[j] * mnx[j];
        }

        sumf += (dx * dy) * (float)sum_qq_acc
              - (dx * miny) * (float)sum_qx_acc
              - (dy * minx) * (float)sum_qy_acc
              + (minx * miny) * 32.0f * (float)acc_mm;
    }

#else
    // Non-VNNI AVX-512 path
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    const __m512i ones_16  = _mm512_set1_epi16(1);
    const __m512i zeros    = _mm512_setzero_si512();

    int i = 0;
    for (; i + 1 < nb; i += 2) {
        _mm_prefetch((const char *)&x[i + 2], _MM_HINT_T0);
        _mm_prefetch((const char *)&y[i + 2], _MM_HINT_T0);
        _mm_prefetch((const char *)&x[i + 3], _MM_HINT_T0);
        _mm_prefetch((const char *)&y[i + 3], _MM_HINT_T0);

        const float dx0   = GGML_FP16_TO_FP32(x[i].d);
        const float minx0 = GGML_FP16_TO_FP32(x[i].dmin);
        const float dy0   = GGML_FP16_TO_FP32(y[i].d);
        const float miny0 = GGML_FP16_TO_FP32(y[i].dmin);

        const float dx1   = GGML_FP16_TO_FP32(x[i+1].d);
        const float minx1 = GGML_FP16_TO_FP32(x[i+1].dmin);
        const float dy1   = GGML_FP16_TO_FP32(y[i+1].d);
        const float miny1 = GGML_FP16_TO_FP32(y[i+1].dmin);

        uint8_t scx0[8], mnx0[8], scy0[8], mny0[8];
        uint8_t scx1[8], mnx1[8], scy1[8], mny1[8];
        unpack_scales_mins_k4(x[i].scales, scx0, mnx0);
        unpack_scales_mins_k4(y[i].scales, scy0, mny0);
        unpack_scales_mins_k4(x[i+1].scales, scx1, mnx1);
        unpack_scales_mins_k4(y[i+1].scales, scy1, mny1);

        const uint8_t * qx0 = x[i].qs;
        const uint8_t * qy0 = y[i].qs;
        const uint8_t * qx1 = x[i+1].qs;
        const uint8_t * qy1 = y[i+1].qs;

        __m512i acc_qq0 = zeros, acc_qx0 = zeros, acc_qy0 = zeros;
        __m512i acc_qq1 = zeros, acc_qx1 = zeros, acc_qy1 = zeros;
        int32_t acc_mm0 = 0, acc_mm1 = 0;

        for (int is = 0; is < 8; is += 2) {
            const int32_t scx0_scy0_1 = (int32_t)scx0[is] * scy0[is];
            const int32_t scx0_scy0_2 = (int32_t)scx0[is+1] * scy0[is+1];
            const int32_t scx0_my0_1  = (int32_t)scx0[is] * mny0[is];
            const int32_t scx0_my0_2  = (int32_t)scx0[is+1] * mny0[is+1];
            const int32_t scy0_mx0_1  = (int32_t)scy0[is] * mnx0[is];
            const int32_t scy0_mx0_2  = (int32_t)scy0[is+1] * mnx0[is+1];

            const int32_t scx1_scy1_1 = (int32_t)scx1[is] * scy1[is];
            const int32_t scx1_scy1_2 = (int32_t)scx1[is+1] * scy1[is+1];
            const int32_t scx1_my1_1  = (int32_t)scx1[is] * mny1[is];
            const int32_t scx1_my1_2  = (int32_t)scx1[is+1] * mny1[is+1];
            const int32_t scy1_mx1_1  = (int32_t)scy1[is] * mnx1[is];
            const int32_t scy1_mx1_2  = (int32_t)scy1[is+1] * mnx1[is+1];

            const __m256i vqx0_256 = _mm256_loadu_si256((const __m256i *)qx0);
            const __m256i vqy0_256 = _mm256_loadu_si256((const __m256i *)qy0);
            const __m256i vqx1_256 = _mm256_loadu_si256((const __m256i *)qx1);
            const __m256i vqy1_256 = _mm256_loadu_si256((const __m256i *)qy1);

            const __m512i vqx0 = _mm512_cvtepu8_epi16(vqx0_256);
            const __m512i vqy0 = _mm512_cvtepu8_epi16(vqy0_256);
            const __m512i vqx1 = _mm512_cvtepu8_epi16(vqx1_256);
            const __m512i vqy1 = _mm512_cvtepu8_epi16(vqy1_256);

            const __m512i vqx0_lo = _mm512_and_si512(vqx0, low_mask);
            const __m512i vqx0_hi = _mm512_and_si512(_mm512_srli_epi16(vqx0, 4), low_mask);
            const __m512i vqy0_lo = _mm512_and_si512(vqy0, low_mask);
            const __m512i vqy0_hi = _mm512_and_si512(_mm512_srli_epi16(vqy0, 4), low_mask);

            const __m512i vqx1_lo = _mm512_and_si512(vqx1, low_mask);
            const __m512i vqx1_hi = _mm512_and_si512(_mm512_srli_epi16(vqx1, 4), low_mask);
            const __m512i vqy1_lo = _mm512_and_si512(vqy1, low_mask);
            const __m512i vqy1_hi = _mm512_and_si512(_mm512_srli_epi16(vqy1, 4), low_mask);

            const __m512i dot0_lo = _mm512_madd_epi16(_mm512_mullo_epi16(vqx0_lo, vqy0_lo), ones_16);
            const __m512i dot0_hi = _mm512_madd_epi16(_mm512_mullo_epi16(vqx0_hi, vqy0_hi), ones_16);
            acc_qq0 = _mm512_add_epi32(acc_qq0, _mm512_mullo_epi32(dot0_lo, _mm512_set1_epi32(scx0_scy0_1)));
            acc_qq0 = _mm512_add_epi32(acc_qq0, _mm512_mullo_epi32(dot0_hi, _mm512_set1_epi32(scx0_scy0_2)));

            const __m512i dot1_lo = _mm512_madd_epi16(_mm512_mullo_epi16(vqx1_lo, vqy1_lo), ones_16);
            const __m512i dot1_hi = _mm512_madd_epi16(_mm512_mullo_epi16(vqx1_hi, vqy1_hi), ones_16);
            acc_qq1 = _mm512_add_epi32(acc_qq1, _mm512_mullo_epi32(dot1_lo, _mm512_set1_epi32(scx1_scy1_1)));
            acc_qq1 = _mm512_add_epi32(acc_qq1, _mm512_mullo_epi32(dot1_hi, _mm512_set1_epi32(scx1_scy1_2)));

            const __m512i sum_qx0_lo = _mm512_sad_epu8(vqx0_lo, zeros);
            const __m512i sum_qx0_hi = _mm512_sad_epu8(vqx0_hi, zeros);
            acc_qx0 = _mm512_add_epi32(acc_qx0, _mm512_mullo_epi32(sum_qx0_lo, _mm512_set1_epi32(scx0_my0_1)));
            acc_qx0 = _mm512_add_epi32(acc_qx0, _mm512_mullo_epi32(sum_qx0_hi, _mm512_set1_epi32(scx0_my0_2)));

            const __m512i sum_qx1_lo = _mm512_sad_epu8(vqx1_lo, zeros);
            const __m512i sum_qx1_hi = _mm512_sad_epu8(vqx1_hi, zeros);
            acc_qx1 = _mm512_add_epi32(acc_qx1, _mm512_mullo_epi32(sum_qx1_lo, _mm512_set1_epi32(scx1_my1_1)));
            acc_qx1 = _mm512_add_epi32(acc_qx1, _mm512_mullo_epi32(sum_qx1_hi, _mm512_set1_epi32(scx1_my1_2)));

            const __m512i sum_qy0_lo = _mm512_sad_epu8(vqy0_lo, zeros);
            const __m512i sum_qy0_hi = _mm512_sad_epu8(vqy0_hi, zeros);
            acc_qy0 = _mm512_add_epi32(acc_qy0, _mm512_mullo_epi32(sum_qy0_lo, _mm512_set1_epi32(scy0_mx0_1)));
            acc_qy0 = _mm512_add_epi32(acc_qy0, _mm512_mullo_epi32(sum_qy0_hi, _mm512_set1_epi32(scy0_mx0_2)));

            const __m512i sum_qy1_lo = _mm512_sad_epu8(vqy1_lo, zeros);
            const __m512i sum_qy1_hi = _mm512_sad_epu8(vqy1_hi, zeros);
            acc_qy1 = _mm512_add_epi32(acc_qy1, _mm512_mullo_epi32(sum_qy1_lo, _mm512_set1_epi32(scy1_mx1_1)));
            acc_qy1 = _mm512_add_epi32(acc_qy1, _mm512_mullo_epi32(sum_qy1_hi, _mm512_set1_epi32(scy1_mx1_2)));

            acc_mm0 += (int32_t)mnx0[is] * mny0[is] + (int32_t)mnx0[is+1] * mny0[is+1];
            acc_mm1 += (int32_t)mnx1[is] * mny1[is] + (int32_t)mnx1[is+1] * mny1[is+1];

            qx0 += 32; qy0 += 32;
            qx1 += 32; qy1 += 32;
        }

        const int32_t sum_qq0 = hsum_i32_16(acc_qq0);
        const int32_t sum_qx0 = hsum_i32_16(acc_qx0);
        const int32_t sum_qy0 = hsum_i32_16(acc_qy0);

        const int32_t sum_qq1 = hsum_i32_16(acc_qq1);
        const int32_t sum_qx1 = hsum_i32_16(acc_qx1);
        const int32_t sum_qy1 = hsum_i32_16(acc_qy1);

        sumf += (dx0 * dy0) * (float)sum_qq0
              - (dx0 * miny0) * (float)sum_qx0
              - (dy0 * minx0) * (float)sum_qy0
              + (minx0 * miny0) * 32.0f * (float)acc_mm0;

        sumf += (dx1 * dy1) * (float)sum_qq1
              - (dx1 * miny1) * (float)sum_qx1
              - (dy1 * minx1) * (float)sum_qy1
              + (minx1 * miny1) * 32.0f * (float)acc_mm1;
    }

    for (; i < nb; i++) {
        const float dx   = GGML_FP16_TO_FP32(x[i].d);
        const float minx = GGML_FP16_TO_FP32(x[i].dmin);
        const float dy   = GGML_FP16_TO_FP32(y[i].d);
        const float miny = GGML_FP16_TO_FP32(y[i].dmin);

        uint8_t scx[8], mnx[8], scy[8], mny[8];
        unpack_scales_mins_k4(x[i].scales, scx, mnx);
        unpack_scales_mins_k4(y[i].scales, scy, mny);

        const uint8_t * qx = x[i].qs;
        const uint8_t * qy = y[i].qs;

        __m512i acc_qq = zeros, acc_qx = zeros, acc_qy = zeros;
        int32_t acc_mm = 0;

        for (int is = 0; is < 8; is += 2) {
            const int32_t scx_scy_1 = (int32_t)scx[is] * scy[is];
            const int32_t scx_scy_2 = (int32_t)scx[is+1] * scy[is+1];
            const int32_t scx_my_1  = (int32_t)scx[is] * mny[is];
            const int32_t scx_my_2  = (int32_t)scx[is+1] * mny[is+1];
            const int32_t scy_mx_1  = (int32_t)scy[is] * mnx[is];
            const int32_t scy_mx_2  = (int32_t)scy[is+1] * mnx[is+1];

            const __m256i vqx_256 = _mm256_loadu_si256((const __m256i *)qx);
            const __m256i vqy_256 = _mm256_loadu_si256((const __m256i *)qy);

            const __m512i vqx = _mm512_cvtepu8_epi16(vqx_256);
            const __m512i vqy = _mm512_cvtepu8_epi16(vqy_256);

            const __m512i vqx_lo = _mm512_and_si512(vqx, low_mask);
            const __m512i vqx_hi = _mm512_and_si512(_mm512_srli_epi16(vqx, 4), low_mask);
            const __m512i vqy_lo = _mm512_and_si512(vqy, low_mask);
            const __m512i vqy_hi = _mm512_and_si512(_mm512_srli_epi16(vqy, 4), low_mask);

            const __m512i dot_lo = _mm512_madd_epi16(_mm512_mullo_epi16(vqx_lo, vqy_lo), ones_16);
            const __m512i dot_hi = _mm512_madd_epi16(_mm512_mullo_epi16(vqx_hi, vqy_hi), ones_16);
            acc_qq = _mm512_add_epi32(acc_qq, _mm512_mullo_epi32(dot_lo, _mm512_set1_epi32(scx_scy_1)));
            acc_qq = _mm512_add_epi32(acc_qq, _mm512_mullo_epi32(dot_hi, _mm512_set1_epi32(scx_scy_2)));

            const __m512i sum_qx_lo = _mm512_sad_epu8(vqx_lo, zeros);
            const __m512i sum_qx_hi = _mm512_sad_epu8(vqx_hi, zeros);
            acc_qx = _mm512_add_epi32(acc_qx, _mm512_mullo_epi32(sum_qx_lo, _mm512_set1_epi32(scx_my_1)));
            acc_qx = _mm512_add_epi32(acc_qx, _mm512_mullo_epi32(sum_qx_hi, _mm512_set1_epi32(scx_my_2)));

            const __m512i sum_qy_lo = _mm512_sad_epu8(vqy_lo, zeros);
            const __m512i sum_qy_hi = _mm512_sad_epu8(vqy_hi, zeros);
            acc_qy = _mm512_add_epi32(acc_qy, _mm512_mullo_epi32(sum_qy_lo, _mm512_set1_epi32(scy_mx_1)));
            acc_qy = _mm512_add_epi32(acc_qy, _mm512_mullo_epi32(sum_qy_hi, _mm512_set1_epi32(scy_mx_2)));

            acc_mm += (int32_t)mnx[is] * mny[is] + (int32_t)mnx[is+1] * mny[is+1];

            qx += 32; qy += 32;
        }

        const int32_t sum_qq = hsum_i32_16(acc_qq);
        const int32_t sum_qx = hsum_i32_16(acc_qx);
        const int32_t sum_qy = hsum_i32_16(acc_qy);

        sumf += (dx * dy) * (float)sum_qq
              - (dx * miny) * (float)sum_qx
              - (dy * minx) * (float)sum_qy
              + (minx * miny) * 32.0f * (float)acc_mm;
    }
#endif

    *s = sumf;
}

#elif defined(RRS_AVX2)

static inline int32_t hsum_i32_8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

static inline void unpack_scales_mins_k4(const uint8_t * scales, uint8_t * sc, uint8_t * mn) {
    for (int j = 0; j < 8; j++) {
        get_scale_min_k4(j, scales, &sc[j], &mn[j]);
    }
}

void ggml_vec_dot_q4_K_rrs_q4_K_rrs(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;

    const int nb = n / QK_K;
    const block_q4_K * x = (const block_q4_K *)vx;
    const block_q4_K * y = (const block_q4_K *)vy;

    float sumf = 0.0f;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones_16  = _mm256_set1_epi16(1);
    const __m256i zeros    = _mm256_setzero_si256();

    int i = 0;
    for (; i + 1 < nb; i += 2) {
        _mm_prefetch((const char *)&x[i + 2], _MM_HINT_T0);
        _mm_prefetch((const char *)&y[i + 2], _MM_HINT_T0);
        _mm_prefetch((const char *)&x[i + 3], _MM_HINT_T0);
        _mm_prefetch((const char *)&y[i + 3], _MM_HINT_T0);

        const float dx0   = GGML_FP16_TO_FP32(x[i].d);
        const float minx0 = GGML_FP16_TO_FP32(x[i].dmin);
        const float dy0   = GGML_FP16_TO_FP32(y[i].d);
        const float miny0 = GGML_FP16_TO_FP32(y[i].dmin);

        const float dx1   = GGML_FP16_TO_FP32(x[i+1].d);
        const float minx1 = GGML_FP16_TO_FP32(x[i+1].dmin);
        const float dy1   = GGML_FP16_TO_FP32(y[i+1].d);
        const float miny1 = GGML_FP16_TO_FP32(y[i+1].dmin);

        uint8_t scx0[8], mnx0[8], scy0[8], mny0[8];
        uint8_t scx1[8], mnx1[8], scy1[8], mny1[8];
        unpack_scales_mins_k4(x[i].scales, scx0, mnx0);
        unpack_scales_mins_k4(y[i].scales, scy0, mny0);
        unpack_scales_mins_k4(x[i+1].scales, scx1, mnx1);
        unpack_scales_mins_k4(y[i+1].scales, scy1, mny1);

        const uint8_t * qx0 = x[i].qs;
        const uint8_t * qy0 = y[i].qs;
        const uint8_t * qx1 = x[i+1].qs;
        const uint8_t * qy1 = y[i+1].qs;

        __m256i acc_qq0 = zeros, acc_qx0 = zeros, acc_qy0 = zeros;
        __m256i acc_qq1 = zeros, acc_qx1 = zeros, acc_qy1 = zeros;
        int32_t acc_mm0 = 0, acc_mm1 = 0;

        for (int is = 0; is < 8; is += 2) {
            const int32_t scx0_scy0_1 = (int32_t)scx0[is] * scy0[is];
            const int32_t scx0_scy0_2 = (int32_t)scx0[is+1] * scy0[is+1];
            const int32_t scx0_my0_1  = (int32_t)scx0[is] * mny0[is];
            const int32_t scx0_my0_2  = (int32_t)scx0[is+1] * mny0[is+1];
            const int32_t scy0_mx0_1  = (int32_t)scy0[is] * mnx0[is];
            const int32_t scy0_mx0_2  = (int32_t)scy0[is+1] * mnx0[is+1];

            const int32_t scx1_scy1_1 = (int32_t)scx1[is] * scy1[is];
            const int32_t scx1_scy1_2 = (int32_t)scx1[is+1] * scy1[is+1];
            const int32_t scx1_my1_1  = (int32_t)scx1[is] * mny1[is];
            const int32_t scx1_my1_2  = (int32_t)scx1[is+1] * mny1[is+1];
            const int32_t scy1_mx1_1  = (int32_t)scy1[is] * mnx1[is];
            const int32_t scy1_mx1_2  = (int32_t)scy1[is+1] * mnx1[is+1];

            const __m256i vqx0 = _mm256_loadu_si256((const __m256i *)qx0);
            const __m256i vqy0 = _mm256_loadu_si256((const __m256i *)qy0);
            const __m256i vqx1 = _mm256_loadu_si256((const __m256i *)qx1);
            const __m256i vqy1 = _mm256_loadu_si256((const __m256i *)qy1);

            const __m256i vqx0_lo = _mm256_and_si256(vqx0, low_mask);
            const __m256i vqx0_hi = _mm256_and_si256(_mm256_srli_epi16(vqx0, 4), low_mask);
            const __m256i vqy0_lo = _mm256_and_si256(vqy0, low_mask);
            const __m256i vqy0_hi = _mm256_and_si256(_mm256_srli_epi16(vqy0, 4), low_mask);

            const __m256i vqx1_lo = _mm256_and_si256(vqx1, low_mask);
            const __m256i vqx1_hi = _mm256_and_si256(_mm256_srli_epi16(vqx1, 4), low_mask);
            const __m256i vqy1_lo = _mm256_and_si256(vqy1, low_mask);
            const __m256i vqy1_hi = _mm256_and_si256(_mm256_srli_epi16(vqy1, 4), low_mask);

            const __m256i dot0_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(vqx0_lo, vqy0_lo), ones_16);
            const __m256i dot0_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(vqx0_hi, vqy0_hi), ones_16);
            acc_qq0 = _mm256_add_epi32(acc_qq0, _mm256_mullo_epi32(dot0_lo, _mm256_set1_epi32(scx0_scy0_1)));
            acc_qq0 = _mm256_add_epi32(acc_qq0, _mm256_mullo_epi32(dot0_hi, _mm256_set1_epi32(scx0_scy0_2)));

            const __m256i dot1_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(vqx1_lo, vqy1_lo), ones_16);
            const __m256i dot1_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(vqx1_hi, vqy1_hi), ones_16);
            acc_qq1 = _mm256_add_epi32(acc_qq1, _mm256_mullo_epi32(dot1_lo, _mm256_set1_epi32(scx1_scy1_1)));
            acc_qq1 = _mm256_add_epi32(acc_qq1, _mm256_mullo_epi32(dot1_hi, _mm256_set1_epi32(scx1_scy1_2)));

            const __m256i sum_qx0_lo = _mm256_sad_epu8(vqx0_lo, zeros);
            const __m256i sum_qx0_hi = _mm256_sad_epu8(vqx0_hi, zeros);
            acc_qx0 = _mm256_add_epi32(acc_qx0, _mm256_mullo_epi32(sum_qx0_lo, _mm256_set1_epi32(scx0_my0_1)));
            acc_qx0 = _mm256_add_epi32(acc_qx0, _mm256_mullo_epi32(sum_qx0_hi, _mm256_set1_epi32(scx0_my0_2)));

            const __m256i sum_qx1_lo = _mm256_sad_epu8(vqx1_lo, zeros);
            const __m256i sum_qx1_hi = _mm256_sad_epu8(vqx1_hi, zeros);
            acc_qx1 = _mm256_add_epi32(acc_qx1, _mm256_mullo_epi32(sum_qx1_lo, _mm256_set1_epi32(scx1_my1_1)));
            acc_qx1 = _mm256_add_epi32(acc_qx1, _mm256_mullo_epi32(sum_qx1_hi, _mm256_set1_epi32(scx1_my1_2)));

            const __m256i sum_qy0_lo = _mm256_sad_epu8(vqy0_lo, zeros);
            const __m256i sum_qy0_hi = _mm256_sad_epu8(vqy0_hi, zeros);
            acc_qy0 = _mm256_add_epi32(acc_qy0, _mm256_mullo_epi32(sum_qy0_lo, _mm256_set1_epi32(scy0_mx0_1)));
            acc_qy0 = _mm256_add_epi32(acc_qy0, _mm256_mullo_epi32(sum_qy0_hi, _mm256_set1_epi32(scy0_mx0_2)));

            const __m256i sum_qy1_lo = _mm256_sad_epu8(vqy1_lo, zeros);
            const __m256i sum_qy1_hi = _mm256_sad_epu8(vqy1_hi, zeros);
            acc_qy1 = _mm256_add_epi32(acc_qy1, _mm256_mullo_epi32(sum_qy1_lo, _mm256_set1_epi32(scy1_mx1_1)));
            acc_qy1 = _mm256_add_epi32(acc_qy1, _mm256_mullo_epi32(sum_qy1_hi, _mm256_set1_epi32(scy1_mx1_2)));

            acc_mm0 += (int32_t)mnx0[is] * mny0[is] + (int32_t)mnx0[is+1] * mny0[is+1];
            acc_mm1 += (int32_t)mnx1[is] * mny1[is] + (int32_t)mnx1[is+1] * mny1[is+1];

            qx0 += 32; qy0 += 32;
            qx1 += 32; qy1 += 32;
        }

        const int32_t sum_qq0 = hsum_i32_8(acc_qq0);
        const int32_t sum_qx0 = hsum_i32_8(acc_qx0);
        const int32_t sum_qy0 = hsum_i32_8(acc_qy0);

        const int32_t sum_qq1 = hsum_i32_8(acc_qq1);
        const int32_t sum_qx1 = hsum_i32_8(acc_qx1);
        const int32_t sum_qy1 = hsum_i32_8(acc_qy1);

        sumf += (dx0 * dy0) * (float)sum_qq0
              - (dx0 * miny0) * (float)sum_qx0
              - (dy0 * minx0) * (float)sum_qy0
              + (minx0 * miny0) * 32.0f * (float)acc_mm0;

        sumf += (dx1 * dy1) * (float)sum_qq1
              - (dx1 * miny1) * (float)sum_qx1
              - (dy1 * minx1) * (float)sum_qy1
              + (minx1 * miny1) * 32.0f * (float)acc_mm1;
    }

    for (; i < nb; i++) {
        const float dx   = GGML_FP16_TO_FP32(x[i].d);
        const float minx = GGML_FP16_TO_FP32(x[i].dmin);
        const float dy   = GGML_FP16_TO_FP32(y[i].d);
        const float miny = GGML_FP16_TO_FP32(y[i].dmin);

        uint8_t scx[8], mnx[8], scy[8], mny[8];
        unpack_scales_mins_k4(x[i].scales, scx, mnx);
        unpack_scales_mins_k4(y[i].scales, scy, mny);

        const uint8_t * qx = x[i].qs;
        const uint8_t * qy = y[i].qs;

        __m256i acc_qq = zeros, acc_qx = zeros, acc_qy = zeros;
        int32_t acc_mm = 0;

        for (int is = 0; is < 8; is += 2) {
            const int32_t scx_scy_1 = (int32_t)scx[is] * scy[is];
            const int32_t scx_scy_2 = (int32_t)scx[is+1] * scy[is+1];
            const int32_t scx_my_1  = (int32_t)scx[is] * mny[is];
            const int32_t scx_my_2  = (int32_t)scx[is+1] * mny[is+1];
            const int32_t scy_mx_1  = (int32_t)scy[is] * mnx[is];
            const int32_t scy_mx_2  = (int32_t)scy[is+1] * mnx[is+1];

            const __m256i vqx = _mm256_loadu_si256((const __m256i *)qx);
            const __m256i vqy = _mm256_loadu_si256((const __m256i *)qy);

            const __m256i vqx_lo = _mm256_and_si256(vqx, low_mask);
            const __m256i vqx_hi = _mm256_and_si256(_mm256_srli_epi16(vqx, 4), low_mask);
            const __m256i vqy_lo = _mm256_and_si256(vqy, low_mask);
            const __m256i vqy_hi = _mm256_and_si256(_mm256_srli_epi16(vqy, 4), low_mask);

            const __m256i dot_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(vqx_lo, vqy_lo), ones_16);
            const __m256i dot_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(vqx_hi, vqy_hi), ones_16);
            acc_qq = _mm256_add_epi32(acc_qq, _mm256_mullo_epi32(dot_lo, _mm256_set1_epi32(scx_scy_1)));
            acc_qq = _mm256_add_epi32(acc_qq, _mm256_mullo_epi32(dot_hi, _mm256_set1_epi32(scx_scy_2)));

            const __m256i sum_qx_lo = _mm256_sad_epu8(vqx_lo, zeros);
            const __m256i sum_qx_hi = _mm256_sad_epu8(vqx_hi, zeros);
            acc_qx = _mm256_add_epi32(acc_qx, _mm256_mullo_epi32(sum_qx_lo, _mm256_set1_epi32(scx_my_1)));
            acc_qx = _mm256_add_epi32(acc_qx, _mm256_mullo_epi32(sum_qx_hi, _mm256_set1_epi32(scx_my_2)));

            const __m256i sum_qy_lo = _mm256_sad_epu8(vqy_lo, zeros);
            const __m256i sum_qy_hi = _mm256_sad_epu8(vqy_hi, zeros);
            acc_qy = _mm256_add_epi32(acc_qy, _mm256_mullo_epi32(sum_qy_lo, _mm256_set1_epi32(scy_mx_1)));
            acc_qy = _mm256_add_epi32(acc_qy, _mm256_mullo_epi32(sum_qy_hi, _mm256_set1_epi32(scy_mx_2)));

            acc_mm += (int32_t)mnx[is] * mny[is] + (int32_t)mnx[is+1] * mny[is+1];

            qx += 32; qy += 32;
        }

        const int32_t sum_qq = hsum_i32_8(acc_qq);
        const int32_t sum_qx = hsum_i32_8(acc_qx);
        const int32_t sum_qy = hsum_i32_8(acc_qy);

        sumf += (dx * dy) * (float)sum_qq
              - (dx * miny) * (float)sum_qx
              - (dy * minx) * (float)sum_qy
              + (minx * miny) * 32.0f * (float)acc_mm;
    }

    *s = sumf;
}

#else

void ggml_vec_dot_q4_K_rrs_q4_K_rrs(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;

    const int nb = n / QK_K;
    const block_q4_K * x = (const block_q4_K *)vx;
    const block_q4_K * y = (const block_q4_K *)vy;

    float sumf = 0.0f;
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

#endif