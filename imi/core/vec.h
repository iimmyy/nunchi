// imi's small math: norms, residual adds, the conv, cutting an input to 8-bit. Each runs over one 2048-wide vector
// (7680 for the FFN), so on its own it is tiny, but every thread repeats it before every table read.
#pragma once
#include <cmath>
#include <cstdint>
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define IMI_VEC_AVX2 1
#endif

#ifdef IMI_VEC_AVX2
inline float imi_hsum(__m256 a) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r)); r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
#endif

// 1 / root mean square, the RMS norm's scale (n a multiple of 32)
inline float imi_rms_scale(const float * x, int n, float eps) {
#ifdef IMI_VEC_AVX2
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i += 32) {
        const __m256 v0 = _mm256_loadu_ps(x + i), v1 = _mm256_loadu_ps(x + i + 8), v2 = _mm256_loadu_ps(x + i + 16), v3 = _mm256_loadu_ps(x + i + 24);
        a0 = _mm256_fmadd_ps(v0, v0, a0); a1 = _mm256_fmadd_ps(v1, v1, a1); a2 = _mm256_fmadd_ps(v2, v2, a2); a3 = _mm256_fmadd_ps(v3, v3, a3);
    }
    const float s = imi_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    return 1.0f / sqrtf(s / (float) n + eps);
#else
    double s = 0;
    for (int i = 0; i < n; i++) s += (double) x[i] * x[i];
    return 1.0f / sqrtf((float) (s / n) + eps);
#endif
}
// o = norm(x) * w
inline void imi_norm_mul(const float * x, const float * w, float * o, int n, float eps) {
    const float sc = imi_rms_scale(x, n, eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * sc * w[i];
}
// x += norm(y) * w   (the residual add after a post-norm)
inline void imi_norm_mul_add(const float * y, const float * w, float * x, int n, float eps) {
    const float sc = imi_rms_scale(y, n, eps);
    for (int i = 0; i < n; i++) x[i] += y[i] * sc * w[i];
}

// cut 32 values to 8-bit: q = round(x / d), d = max|x| / 127; returns d, writes the sum of q to *sum
inline float imi_cut32(const float * x, int8_t * q, int * sum) {
#ifdef IMI_VEC_AVX2
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 v[4];
    __m256 mx = _mm256_setzero_ps();
    for (int k = 0; k < 4; k++) { v[k] = _mm256_loadu_ps(x + 8 * k); mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, v[k])); }
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m)); m = _mm_max_ss(m, _mm_movehdup_ps(m));
    const float amax = _mm_cvtss_f32(m), d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
    const __m256 vid = _mm256_set1_ps(id);
    __m256i i[4];
    for (int k = 0; k < 4; k++) i[k] = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[k], vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m256i s = _mm256_add_epi32(_mm256_add_epi32(i[0], i[1]), _mm256_add_epi32(i[2], i[3]));
    __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4e)); s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xb1));
    *sum = _mm_cvtsi128_si32(s4);
    // pack 32 x int32 -> 32 x int8 in order (packs work per 128-bit half, so fix the order with one permute)
    const __m256i p01 = _mm256_packs_epi32(i[0], i[1]), p23 = _mm256_packs_epi32(i[2], i[3]);
    const __m256i p = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p01, p23), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256((__m256i *) q, p);
    return d;
#else
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
    const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
    int s = 0;
    for (int i = 0; i < 32; i++) { const int c = (int) lrintf(x[i] * id); q[i] = (int8_t) c; s += c; }
    *sum = s;
    return d;
#endif
}
