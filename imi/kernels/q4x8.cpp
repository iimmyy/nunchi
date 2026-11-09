// Our 4-bit kernel. Q4_0 stores weight = (code - 8) * d per 32 weights. The input is quantized to int8 per 32
// (scale dx), so  dot = sum_j d_j * dx_j * (sum_i code_i * xq_i - 8 * sum_i xq_i).  The second sum depends only on
// the input, so it is computed once per block and shared by every row: the inner loop never fixes signs.
// 8 rows per group: their 8 integer sums fold into one register, so scaling and the correction are one multiply and
// two FMAs for 8 rows. Floor box (Haswell, 2 cores): 10.5 GB/s on one thread vs ggml's best 8.5 (+24%).
#include "q4x8.h"
#include "ggml-cpu.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace {
struct blk_q4_0 { ggml_fp16_t d; uint8_t qs[16]; };

struct qin { std::vector<int8_t> q; std::vector<float> d, corr; };

void quant_in(const float * x, int n, qin & o) {
    const int nb = n / 32;
    o.q.resize(n); o.d.resize(nb); o.corr.resize(nb);
    for (int j = 0; j < nb; j++) {
        float amax = 0; for (int i = 0; i < 32; i++) amax = std::max(amax, fabsf(x[j * 32 + i]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        int s = 0;
        for (int i = 0; i < 32; i++) { const int v = (int) roundf(x[j * 32 + i] * id); o.q[j * 32 + i] = (int8_t) v; s += v; }
        o.d[j] = d; o.corr[j] = 8.0f * d * (float) s;
    }
}

#ifdef __AVX2__
void rows8(const imi_q4x8 & W, const qin & X, float * y, int g0, int g1) {
    const __m256i lo = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    const int nb = W.nb;
    for (int g = g0; g < g1; g++) {
        __m256 acc = _mm256_setzero_ps(), cor = _mm256_setzero_ps();
        const uint8_t * w = &W.codes[(size_t) g * nb * 128];
        const float * sc = &W.scales[(size_t) g * nb * 8];
        for (int j = 0; j < nb; j++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *) &X.q[j * 32]);
            const uint8_t * c = w + (size_t) j * 128;
            __m256i p[8];
            for (int k = 0; k < 4; k++) {
                const __m256i v = _mm256_loadu_si256((const __m256i *) (c + 32 * k));
                p[2 * k]     = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(v, lo), x), ones);
                p[2 * k + 1] = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v, 4), lo), x), ones);
            }
            const __m256i h01 = _mm256_hadd_epi32(p[0], p[1]), h23 = _mm256_hadd_epi32(p[2], p[3]);
            const __m256i h45 = _mm256_hadd_epi32(p[4], p[5]), h67 = _mm256_hadd_epi32(p[6], p[7]);
            const __m256i h0123 = _mm256_hadd_epi32(h01, h23), h4567 = _mm256_hadd_epi32(h45, h67);
            const __m256i s = _mm256_add_epi32(_mm256_permute2x128_si256(h0123, h4567, 0x20), _mm256_permute2x128_si256(h0123, h4567, 0x31));
            const __m256 d8 = _mm256_loadu_ps(sc + (size_t) j * 8);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(d8, _mm256_set1_ps(X.d[j])), acc);
            cor = _mm256_fmadd_ps(d8, _mm256_set1_ps(X.corr[j]), cor);
        }
        _mm256_storeu_ps(y + 8 * g, _mm256_sub_ps(acc, cor));
    }
}
#endif

void op(ggml_tensor * dst, int ith, int nth, void * ud) {
#ifdef __AVX2__
    imi_q4x8_call & call = *(imi_q4x8_call *) ud;
    const imi_q4x8 & W = *call.w;
    thread_local qin X;
    quant_in((const float *) dst->src[0]->data, W.cols, X);   // each thread, ~1 us: no extra stop between threads
    const int ng = W.rows / 8, chunk = 4;                     // 32 rows per chunk, taken by whoever is free
    for (int g = call.next.fetch_add(chunk); g < ng; g = call.next.fetch_add(chunk)) {
        rows8(W, X, (float *) dst->data, g, std::min(ng, g + chunk));
    }
    (void) ith; (void) nth;
#else
    (void) dst; (void) ith; (void) nth; (void) ud;
#endif
}
} // namespace

bool imi_q4x8_supported(const ggml_tensor * w) {
#ifdef __AVX2__
    return w->type == GGML_TYPE_Q4_0 && ggml_n_dims(w) == 2 && w->ne[1] % 8 == 0 && w->ne[0] % 32 == 0;
#else
    (void) w; return false;   // NEON version: next (Mac, iPhone)
#endif
}

imi_q4x8 * imi_q4x8_from(const ggml_tensor * t) {
    auto * o = new imi_q4x8;
    o->cols = (int) t->ne[0]; o->rows = (int) t->ne[1]; o->nb = o->cols / 32;
    const int ng = o->rows / 8;
    o->codes.resize((size_t) ng * o->nb * 128); o->scales.resize((size_t) ng * o->nb * 8);
    const blk_q4_0 * w = (const blk_q4_0 *) t->data;
    auto code = [&](int r, int j, int i) { const blk_q4_0 & b = w[(size_t) r * o->nb + j]; return i < 16 ? (b.qs[i] & 15) : (b.qs[i - 16] >> 4); };
    for (int g = 0; g < ng; g++) for (int j = 0; j < o->nb; j++) {
        uint8_t * c = &o->codes[((size_t) g * o->nb + j) * 128];
        for (int k = 0; k < 4; k++) for (int i = 0; i < 32; i++)
            c[k * 32 + i] = (uint8_t) (code(8 * g + 2 * k, j, i) | (code(8 * g + 2 * k + 1, j, i) << 4));
        for (int r = 0; r < 8; r++) o->scales[((size_t) g * o->nb + j) * 8 + r] = ggml_fp16_to_fp32(w[(size_t) (8 * g + r) * o->nb + j].d);
    }
    return o;
}

ggml_tensor * imi_q4x8_mul(ggml_context * ctx, imi_q4x8_call * call, ggml_tensor * x) {
    ggml_tensor * args[1] = { x };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, call->w->rows, 1, 1, 1, args, 1, op, GGML_N_TASKS_MAX, call);
}
