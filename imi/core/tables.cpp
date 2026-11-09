#include "tables.h"
#include "vec.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define IMI_AVX2 1
#endif

float imi_h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t b;
    if (e == 0) {
        if (m == 0) b = s;
        else { float f = (float) m / 1024.0f / 16384.0f; memcpy(&b, &f, 4); b |= s; }  // subnormal: m * 2^-24
    } else if (e == 31) b = s | 0x7f800000 | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &b, 4); return f;
}
uint16_t imi_f2h(float f) {
#ifdef IMI_AVX2
    return (uint16_t) _cvtss_sh(f, 0);
#elif defined(__aarch64__)
    __fp16 h = (__fp16) f; uint16_t r; memcpy(&r, &h, 2); return r;
#else
    return ggml_fp32_to_fp16(f);
#endif
}

void * imi_big_alloc(size_t bytes) {
    const size_t a = 2u << 20, n = (bytes + a - 1) / a * a;
    void * p = nullptr;
    if (posix_memalign(&p, a, n) != 0) return nullptr;
#ifdef MADV_HUGEPAGE
    madvise(p, n, MADV_HUGEPAGE);
#endif
    return p;
}
void imi_big_free(void * p) { free(p); }

namespace {
struct blk_q4_0 { uint16_t d; uint8_t qs[16]; };
struct blk_q4_1 { uint16_t d, m; uint8_t qs[16]; };
struct blk_q6_k { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(blk_q4_0) == 18 && sizeof(blk_q4_1) == 20 && sizeof(blk_q6_k) == 210, "block sizes");
}

bool imi_q4::build(const std::vector<std::pair<const ggml_tensor *, int>> & order) {
    if (order.empty() || order.size() % 8) return false;
    const ggml_tensor * t0 = order[0].first;
    if (t0->type != GGML_TYPE_Q4_0 && t0->type != GGML_TYPE_Q4_1) return false;
    for (auto & o : order) if (o.first->type != t0->type || o.first->ne[0] != t0->ne[0]) return false;
    rows = (int) order.size(); cols = (int) t0->ne[0]; nb = cols / 32; has_m = t0->type == GGML_TYPE_Q4_1;
    const size_t ng = rows / 8, nc = ng * nb * 128, ns = ng * nb * 8 * sizeof(uint16_t);
    bytes = nc + ns * (has_m ? 2 : 1);
    mem = imi_big_alloc(bytes);
    if (!mem) return false;
    codes = (uint8_t *) mem; d = (uint16_t *) (codes + nc); m = has_m ? d + ng * nb * 8 : nullptr;
    auto blk = [&](int r, int j, uint16_t & dd, uint16_t & mm) -> const uint8_t * {
        const ggml_tensor * t = order[r].first;
        const uint8_t * row = (const uint8_t *) t->data + (size_t) order[r].second * t->nb[1];
        if (has_m) { const blk_q4_1 & b = ((const blk_q4_1 *) row)[j]; dd = b.d; mm = b.m; return b.qs; }
        const blk_q4_0 & b = ((const blk_q4_0 *) row)[j]; dd = b.d; mm = 0; return b.qs;
    };
    for (size_t g = 0; g < ng; g++) for (int j = 0; j < nb; j++) {
        uint8_t * c = codes + (g * nb + j) * 128;
        for (int r = 0; r < 8; r++) {
            uint16_t dd, mm;
            const uint8_t * qs = blk((int) (8 * g + r), j, dd, mm);
            d[(g * nb + j) * 8 + r] = dd;
            if (has_m) m[(g * nb + j) * 8 + r] = mm;
            // Q4_0/Q4_1 byte i: column i (low 4 bits) and column 16 + i (high 4 bits)
            for (int tq = 0; tq < 4; tq++) for (int i = 0; i < 4; i++)
                c[32 * tq + 4 * r + i] = (uint8_t) ((qs[4 * tq + i] & 15) | (qs[4 * tq + i] & 0xF0));
        }
    }
    return true;
}
void imi_q4::release() { if (mem) imi_big_free(mem); mem = nullptr; }

void imi_q8::from(const float * x, int nn) {
    n = nn;
    const int nbk = n / 32;
    if ((int) q.size() < n) { q.resize(n); d.resize(nbk); s.resize(nbk); }
    for (int j = 0; j < nbk; j++) {
        int sum;
        d[j] = imi_cut32(x + 32 * j, &q[32 * j], &sum);
        s[j] = d[j] * (float) sum;
    }
}

#ifdef IMI_AVX2
// Our 4-bit layout puts 8 rows side by side, 4 columns each per 4 bytes. One maddubs against the input's 4 columns
// (broadcast) gives every row's partial sum in its own lanes, so the 8 rows never need folding together: the 16-bit
// sums just add up across the block (at most 16 products of 15 * 127 per lane) and one widening add finishes it.
static inline __m256i rows8_sum(const uint8_t * c, const int8_t * xq) {
    const __m256i lo = _mm256_set1_epi8(15);
    __m256i a = _mm256_setzero_si256();
    for (int t = 0; t < 4; t++) {
        int32_t xa, xb; memcpy(&xa, xq + 4 * t, 4); memcpy(&xb, xq + 16 + 4 * t, 4);
        const __m256i v = _mm256_loadu_si256((const __m256i *) (c + 32 * t));
        a = _mm256_add_epi16(a, _mm256_maddubs_epi16(_mm256_and_si256(v, lo), _mm256_set1_epi32(xa)));
        a = _mm256_add_epi16(a, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v, 4), lo), _mm256_set1_epi32(xb)));
    }
    return _mm256_madd_epi16(a, _mm256_set1_epi16(1));
}

template <bool HAS_M>
static inline void group_avx2(const imi_q4 & W, const imi_q8 & X, int g, float * out) {
    const int nb = W.nb;
    const uint8_t * w = W.codes + (size_t) g * nb * 128;
    const uint16_t * dp = W.d + (size_t) g * nb * 8;
    const uint16_t * mp = HAS_M ? W.m + (size_t) g * nb * 8 : nullptr;
    __m256 acc = _mm256_setzero_ps(), cor = _mm256_setzero_ps();
    for (int j = 0; j < nb; j++) {
        const __m256i s = rows8_sum(w + (size_t) j * 128, &X.q[32 * j]);
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (dp + 8 * j)));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(d8, _mm256_set1_ps(X.d[j])), acc);
        if (HAS_M) cor = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (mp + 8 * j))), _mm256_set1_ps(X.s[j]), cor);
        else       cor = _mm256_fmadd_ps(d8, _mm256_set1_ps(X.s[j]), cor);
    }
    // Q4_0: sum d*dx*(code . xq) - 8 * d * dx*sum(xq);  Q4_1: sum d*dx*(code . xq) + m * dx*sum(xq)
    _mm256_storeu_ps(out, HAS_M ? _mm256_add_ps(acc, cor) : _mm256_fnmadd_ps(_mm256_set1_ps(8.0f), cor, acc));
}

template <bool HAS_M>
static inline void group2_avx2(const imi_q4 & W, const imi_q8 & X, const imi_q8 & X2, int g, float * out, float * out2) {
    const int nb = W.nb;
    const uint8_t * w = W.codes + (size_t) g * nb * 128;
    const uint16_t * dp = W.d + (size_t) g * nb * 8;
    const uint16_t * mp = HAS_M ? W.m + (size_t) g * nb * 8 : nullptr;
    __m256 acc = _mm256_setzero_ps(), cor = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), cor2 = _mm256_setzero_ps();
    for (int j = 0; j < nb; j++) {
        const __m256i s = rows8_sum(w + (size_t) j * 128, &X.q[32 * j]), s2 = rows8_sum(w + (size_t) j * 128, &X2.q[32 * j]);
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (dp + 8 * j)));
        acc  = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s),  _mm256_mul_ps(d8, _mm256_set1_ps(X.d[j])), acc);
        acc2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s2), _mm256_mul_ps(d8, _mm256_set1_ps(X2.d[j])), acc2);
        const __m256 mm = HAS_M ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (mp + 8 * j))) : d8;
        cor  = _mm256_fmadd_ps(mm, _mm256_set1_ps(X.s[j]), cor);
        cor2 = _mm256_fmadd_ps(mm, _mm256_set1_ps(X2.s[j]), cor2);
    }
    _mm256_storeu_ps(out,  HAS_M ? _mm256_add_ps(acc, cor)   : _mm256_fnmadd_ps(_mm256_set1_ps(8.0f), cor, acc));
    _mm256_storeu_ps(out2, HAS_M ? _mm256_add_ps(acc2, cor2) : _mm256_fnmadd_ps(_mm256_set1_ps(8.0f), cor2, acc2));
}
#endif

// code of row r (0..7), column k (0..31) of a block in our layout
static inline int code_at(const uint8_t * c, int r, int k) {
    return k < 16 ? (c[32 * (k / 4) + 4 * r + k % 4] & 15) : (c[32 * ((k - 16) / 4) + 4 * r + (k - 16) % 4] >> 4);
}

void imi_q4_group2(const imi_q4 & W, const imi_q8 & X, const imi_q8 & X2, int g, float * out, float * out2) {
#ifdef IMI_AVX2
    if (W.has_m) group2_avx2<true>(W, X, X2, g, out, out2); else group2_avx2<false>(W, X, X2, g, out, out2);
#else
    imi_q4_group(W, X, g, out); imi_q4_group(W, X2, g, out2);
#endif
}

void imi_q4_group(const imi_q4 & W, const imi_q8 & X, int g, float * out) {
#ifdef IMI_AVX2
    if (W.has_m) group_avx2<true>(W, X, g, out); else group_avx2<false>(W, X, g, out);
#else
    const int nb = W.nb;
    for (int r = 0; r < 8; r++) {
        float acc = 0;
        for (int j = 0; j < nb; j++) {
            const uint8_t * c = W.codes + ((size_t) g * nb + j) * 128;
            int s = 0;
            for (int k = 0; k < 32; k++) s += code_at(c, r, k) * X.q[32 * j + k];
            const float dd = imi_h2f(W.d[((size_t) g * nb + j) * 8 + r]);
            acc += dd * X.d[j] * (float) s;
            if (W.has_m) acc += imi_h2f(W.m[((size_t) g * nb + j) * 8 + r]) * X.s[j];
            else         acc -= 8.0f * dd * X.s[j];
        }
        out[r] = acc;
    }
#endif
}

void imi_q8k::from(const float * x, int nn) {
    n = nn;
    const int nbk = n / 256;
    q.resize(n); d.resize(nbk); bsums.resize((size_t) nbk * 16);
    for (int j = 0; j < nbk; j++) {
        const float * v = x + 256 * j;
        float amax = 0;
        for (int i = 0; i < 256; i++) amax = std::max(amax, fabsf(v[i]));
        const float dd = amax / 127.0f, id = dd ? 1.0f / dd : 0.0f;
        for (int i = 0; i < 256; i++) q[256 * j + i] = (int8_t) lrintf(v[i] * id);
        for (int s = 0; s < 16; s++) { int t = 0; for (int i = 0; i < 16; i++) t += q[256 * j + 16 * s + i]; bsums[16 * j + s] = (int16_t) t; }
        d[j] = dd;
    }
}

// Q6_K value i of a 256 block: low 4 bits from ql, high 2 from qh, minus 32, times d * scale[i / 16]
void imi_q6k_dequant_row(const void * row, float * y, int nblocks) {
    const blk_q6_k * b = (const blk_q6_k *) row;
    for (int i = 0; i < nblocks; i++) {
        const float d = imi_h2f(b[i].d);
        const uint8_t * ql = b[i].ql; const uint8_t * qh = b[i].qh; const int8_t * sc = b[i].scales;
        float * o = y + 256 * i;
        for (int n = 0; n < 2; n++) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                o[l]      = d * sc[is + 0] * (float) ((int) ((ql[l] & 15)      | (((qh[l] >> 0) & 3) << 4)) - 32);
                o[l + 32] = d * sc[is + 2] * (float) ((int) ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32);
                o[l + 64] = d * sc[is + 4] * (float) ((int) ((ql[l] >> 4)      | (((qh[l] >> 4) & 3) << 4)) - 32);
                o[l + 96] = d * sc[is + 6] * (float) ((int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            o += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

float imi_q6k_row(const void * row, const imi_q8k & X, int nblocks) {
    const blk_q6_k * b = (const blk_q6_k *) row;
#ifdef IMI_AVX2
    const __m256i m4 = _mm256_set1_epi8(15), m30 = _mm256_set1_epi8(0x30);
    // broadcast scale 2k to the low half and 2k+1 to the high half (16-bit lanes)
    static const __m256i bc[4] = {
        _mm256_setr_epi8(0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3),
        _mm256_setr_epi8(4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5, 6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7),
        _mm256_setr_epi8(8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9, 10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11),
        _mm256_setr_epi8(12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13, 14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15),
    };
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; i++) {
        const uint8_t * ql = b[i].ql; const uint8_t * qh = b[i].qh; const int8_t * sc = b[i].scales;
        const int8_t * xq = X.q.data() + 256 * i;
        const __m256i corr = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) sc)),
                                               _mm256_loadu_si256((const __m256i *) (X.bsums.data() + 16 * i)));
        __m256i isum = _mm256_setzero_si256();
        for (int n = 0; n < 2; n++) {
            const __m256i a = _mm256_loadu_si256((const __m256i *) ql), a2 = _mm256_loadu_si256((const __m256i *) (ql + 32));
            const __m256i h = _mm256_loadu_si256((const __m256i *) qh);
            const __m256i q[4] = {
                _mm256_or_si256(_mm256_and_si256(a, m4),                       _mm256_and_si256(_mm256_slli_epi16(h, 4), m30)),
                _mm256_or_si256(_mm256_and_si256(a2, m4),                      _mm256_and_si256(_mm256_slli_epi16(h, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a, 4), m4),  _mm256_and_si256(h, m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a2, 4), m4), _mm256_and_si256(_mm256_srli_epi16(h, 2), m30)),
            };
            const __m128i s8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *) (sc + 8 * n)));
            const __m256i s2 = _mm256_set_m128i(s8, s8);
            for (int k = 0; k < 4; k++) {
                const __m256i p = _mm256_maddubs_epi16(q[k], _mm256_loadu_si256((const __m256i *) (xq + 128 * n + 32 * k)));
                isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p, _mm256_shuffle_epi8(s2, bc[k])));
            }
            ql += 64; qh += 32;
        }
        isum = _mm256_sub_epi32(isum, _mm256_slli_epi32(corr, 5));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(b[i].d) * X.d[i]), _mm256_cvtepi32_ps(isum), acc);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
#else
    float y[256], acc = 0;
    for (int i = 0; i < nblocks; i++) {
        imi_q6k_dequant_row(b + i, y, 1);
        float s = 0;
        for (int k = 0; k < 256; k++) s += y[k] * (float) X.q[256 * i + k];
        acc += s * X.d[i];
    }
    return acc;
#endif
}

// ---- 8-bit tables, many inputs ----
namespace { struct blk_q8_0 { uint16_t d; int8_t qs[32]; }; static_assert(sizeof(blk_q8_0) == 34, "q8_0"); }

void imi_q8_0_dequant_row(const void * row, float * y, int cols) {
    const blk_q8_0 * b = (const blk_q8_0 *) row;
    for (int j = 0; j < cols / 32; j++) { const float d = imi_h2f(b[j].d); for (int i = 0; i < 32; i++) y[32 * j + i] = d * b[j].qs[i]; }
}

bool imi_q8g::build(const std::vector<std::pair<const ggml_tensor *, int>> & order, int plane) {
    if (order.empty()) return false;
    const ggml_tensor * t0 = order[0].first;
    for (auto & o : order) if (o.first->type != GGML_TYPE_Q8_0 || o.first->ne[0] != t0->ne[0]) return false;
    rows = (int) order.size(); rows_pad = (rows + 7) / 8 * 8; cols = (int) t0->ne[0]; nb = cols / 32;
    const size_t ng = rows_pad / 8, nc = ng * cols * 8, ns = ng * nb * 8 * sizeof(uint16_t);
    bytes = nc + ns;
    mem = imi_big_alloc(bytes);
    if (!mem) return false;
    memset(mem, 0, bytes);
    codes = (int8_t *) mem; d = (uint16_t *) ((uint8_t *) mem + nc);
    for (int r = 0; r < rows; r++) {
        const ggml_tensor * t = order[r].first;
        const blk_q8_0 * b = (const blk_q8_0 *) ((const uint8_t *) t->data + (size_t) order[r].second * t->nb[1] + (size_t) plane * t->nb[2]);
        const size_t g = r / 8, rr = r % 8;
        for (int j = 0; j < nb; j++) {
            d[(g * nb + j) * 8 + rr] = b[j].d;
            for (int i = 0; i < 32; i++) codes[(g * cols + 32 * j + i) * 8 + rr] = b[j].qs[i];
        }
    }
    return true;
}
void imi_q8g::release() { if (mem) imi_big_free(mem); mem = nullptr; }

#ifdef IMI_AVX2
// the 12 running sums live in 12 named registers: as an array, the compiler wrote all 12 back to memory after every
// multiply-add, and one store per cycle was the limit
static void q8g_avx2_12(const imi_q8g & W, const float * __restrict xt, int g, float * __restrict out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0;
    const int8_t * w = W.codes + (size_t) g * W.cols * 8;
    const uint16_t * dp = W.d + (size_t) g * W.nb * 8;
    for (int j = 0; j < W.nb; j++) {
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (dp + 8 * j)));
        const int8_t * wj = w + (size_t) j * 256;
        const float * xj = xt + (size_t) j * 32 * 12;
        for (int kk = 0; kk < 32; kk++) {
            const __m256 wv = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (wj + kk * 8)))), d8);
            const float * xk = xj + kk * 12;
            a0  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 0), a0);   a1  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 1), a1);
            a2  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 2), a2);   a3  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 3), a3);
            a4  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 4), a4);   a5  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 5), a5);
            a6  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 6), a6);   a7  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 7), a7);
            a8  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 8), a8);   a9  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 9), a9);
            a10 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 10), a10); a11 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 11), a11);
        }
    }
    _mm256_storeu_ps(out + 0, a0);  _mm256_storeu_ps(out + 8, a1);  _mm256_storeu_ps(out + 16, a2); _mm256_storeu_ps(out + 24, a3);
    _mm256_storeu_ps(out + 32, a4); _mm256_storeu_ps(out + 40, a5); _mm256_storeu_ps(out + 48, a6); _mm256_storeu_ps(out + 56, a7);
    _mm256_storeu_ps(out + 64, a8); _mm256_storeu_ps(out + 72, a9); _mm256_storeu_ps(out + 80, a10); _mm256_storeu_ps(out + 88, a11);
}
static void q8g_avx2_6(const imi_q8g & W, const float * __restrict xt, int g, float * __restrict out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
    const int8_t * w = W.codes + (size_t) g * W.cols * 8;
    const uint16_t * dp = W.d + (size_t) g * W.nb * 8;
    for (int j = 0; j < W.nb; j++) {
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (dp + 8 * j)));
        const int8_t * wj = w + (size_t) j * 256;
        const float * xj = xt + (size_t) j * 32 * 6;
        for (int kk = 0; kk < 32; kk++) {
            const __m256 wv = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (wj + kk * 8)))), d8);
            const float * xk = xj + kk * 6;
            a0 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 0), a0); a1 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 1), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 2), a2); a3 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 3), a3);
            a4 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 4), a4); a5 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 5), a5);
        }
    }
    _mm256_storeu_ps(out + 0, a0);  _mm256_storeu_ps(out + 8, a1);  _mm256_storeu_ps(out + 16, a2);
    _mm256_storeu_ps(out + 24, a3); _mm256_storeu_ps(out + 32, a4); _mm256_storeu_ps(out + 40, a5);
}
static void q8g_avx2_1(const imi_q8g & W, const float * __restrict x, int g, float * __restrict out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;   // 4 chains so the adds overlap
    const int8_t * w = W.codes + (size_t) g * W.cols * 8;
    const uint16_t * dp = W.d + (size_t) g * W.nb * 8;
    for (int j = 0; j < W.nb; j++) {
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (dp + 8 * j)));
        const int8_t * wj = w + (size_t) j * 256;
        const float * xj = x + 32 * j;
        for (int kk = 0; kk < 32; kk += 4) {
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (wj + kk * 8)))),      _mm256_mul_ps(d8, _mm256_broadcast_ss(xj + kk)), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (wj + kk * 8 + 8)))),  _mm256_mul_ps(d8, _mm256_broadcast_ss(xj + kk + 1)), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (wj + kk * 8 + 16)))), _mm256_mul_ps(d8, _mm256_broadcast_ss(xj + kk + 2)), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (wj + kk * 8 + 24)))), _mm256_mul_ps(d8, _mm256_broadcast_ss(xj + kk + 3)), a3);
        }
    }
    _mm256_storeu_ps(out, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}
#endif

void imi_q8g_group(const imi_q8g & W, const float * xt, int C, int g, float * out) {
#ifdef IMI_AVX2
    if (C == 12) { q8g_avx2_12(W, xt, g, out); return; }
    if (C == 6)  { q8g_avx2_6(W, xt, g, out); return; }
    if (C == 1)  { q8g_avx2_1(W, xt, g, out); return; }
#endif
    std::vector<float> acc((size_t) C * 8, 0.0f);
    const int8_t * w = W.codes + (size_t) g * W.cols * 8;
    for (int j = 0; j < W.nb; j++) for (int r = 0; r < 8; r++) {
        const float dd = imi_h2f(W.d[((size_t) g * W.nb + j) * 8 + r]);
        for (int c = 0; c < C; c++) {
            float s = 0;
            for (int kk = 0; kk < 32; kk++) { const int k = 32 * j + kk; s += (float) w[(size_t) k * 8 + r] * xt[(size_t) k * C + c]; }
            acc[(size_t) c * 8 + r] += dd * s;
        }
    }
    memcpy(out, acc.data(), acc.size() * sizeof(float));
}

// ---- 4-bit tables, many inputs (reading a prompt) ----
void imi_q8b::shape(int n_in, int n_cols) {
    n = n_in; cols = n_cols; nb = cols / 32;
    if (q.size() < (size_t) n * cols) q.resize((size_t) n * cols);
    if (d.size() < (size_t) n * nb) { d.resize((size_t) n * nb); s.resize((size_t) n * nb); }
}
void imi_q8b::set(int i, const float * x) {
    for (int j = 0; j < nb; j++) {
        int sum;
        const float dd = imi_cut32(x + 32 * j, &q[((size_t) j * n + i) * 32], &sum);
        d[(size_t) j * n + i] = dd; s[(size_t) j * n + i] = dd * (float) sum;
    }
}

#ifdef IMI_AVX2
template <bool HAS_M>
static void group_batch_avx2(const imi_q4 & W, const imi_q8b & X, int g, float * out) {
    const int n = X.n, nb = W.nb;
    const __m256i lo = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 acc[64], cor[64];
    for (int c = 0; c < n; c++) { acc[c] = _mm256_setzero_ps(); cor[c] = _mm256_setzero_ps(); }
    const uint8_t * w = W.codes + (size_t) g * nb * 128;
    const uint16_t * dp = W.d + (size_t) g * nb * 8;
    const uint16_t * mp = HAS_M ? W.m + (size_t) g * nb * 8 : nullptr;
    for (int j = 0; j < nb; j++) {
        // the 8 rows' codes of this block, unpacked once for every input
        __m256i nl[4], nh[4];
        for (int t = 0; t < 4; t++) {
            const __m256i v = _mm256_loadu_si256((const __m256i *) (w + (size_t) j * 128 + 32 * t));
            nl[t] = _mm256_and_si256(v, lo); nh[t] = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo);
        }
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (dp + 8 * j)));
        const __m256 m8 = HAS_M ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (mp + 8 * j))) : d8;
        const int8_t * xq = X.q.data() + (size_t) j * n * 32;
        const float * xd = X.d.data() + (size_t) j * n, * xs = X.s.data() + (size_t) j * n;
        for (int c = 0; c < n; c++) {
            const int8_t * x = xq + 32 * c;
            __m256i a = _mm256_setzero_si256();
            for (int t = 0; t < 4; t++) {
                int32_t xa, xb; memcpy(&xa, x + 4 * t, 4); memcpy(&xb, x + 16 + 4 * t, 4);
                a = _mm256_add_epi16(a, _mm256_maddubs_epi16(nl[t], _mm256_set1_epi32(xa)));
                a = _mm256_add_epi16(a, _mm256_maddubs_epi16(nh[t], _mm256_set1_epi32(xb)));
            }
            const __m256i s = _mm256_madd_epi16(a, ones);
            acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(d8, _mm256_set1_ps(xd[c])), acc[c]);
            cor[c] = _mm256_fmadd_ps(m8, _mm256_set1_ps(xs[c]), cor[c]);
        }
    }
    for (int c = 0; c < n; c++)
        _mm256_storeu_ps(out + 8 * c, HAS_M ? _mm256_add_ps(acc[c], cor[c]) : _mm256_fnmadd_ps(_mm256_set1_ps(8.0f), cor[c], acc[c]));
}
#endif

void imi_q4_group_batch(const imi_q4 & W, const imi_q8b & X, int g, float * out) {
#ifdef IMI_AVX2
    if (W.has_m) group_batch_avx2<true>(W, X, g, out); else group_batch_avx2<false>(W, X, g, out);
#else
    const int nb = W.nb;
    for (int c = 0; c < X.n; c++) for (int r = 0; r < 8; r++) {
        float acc = 0;
        for (int j = 0; j < nb; j++) {
            const uint8_t * cc = W.codes + ((size_t) g * nb + j) * 128;
            const int8_t * xq = X.q.data() + ((size_t) j * X.n + c) * 32;
            int s = 0;
            for (int k = 0; k < 32; k++) s += code_at(cc, r, k) * xq[k];
            const float dd = imi_h2f(W.d[((size_t) g * nb + j) * 8 + r]);
            acc += dd * X.d[(size_t) j * X.n + c] * (float) s;
            if (W.has_m) acc += imi_h2f(W.m[((size_t) g * nb + j) * 8 + r]) * X.s[(size_t) j * X.n + c];
            else         acc -= 8.0f * dd * X.s[(size_t) j * X.n + c];
        }
        out[8 * c + r] = acc;
    }
#endif
}

// ---- float tables, 12 inputs ----
namespace {
template <typename T> void wg_fill(imi_wg & W, int rows, int cols, T get) {
    W.cols = cols; W.rows = rows; W.rows_pad = (rows + 7) / 8 * 8;
    const size_t esz = W.f32 ? 4 : 2;
    W.bytes = (size_t) W.rows_pad * cols * esz;
    W.w = imi_big_alloc(W.bytes);
    memset(W.w, 0, W.bytes);
    for (int r = 0; r < rows; r++) for (int k = 0; k < cols; k++) {
        const size_t at = ((size_t) (r / 8) * cols + k) * 8 + r % 8;
        if (W.f32) ((float *) W.w)[at] = get(r, k); else ((uint16_t *) W.w)[at] = get(r, k);
    }
}
}
bool imi_wg::build(const std::vector<std::pair<const ggml_tensor *, int>> & rows_in) {
    if (rows_in.empty()) return false;
    const ggml_tensor * t0 = rows_in[0].first;
    const int c = (int) t0->ne[0];
    for (const auto & p : rows_in) if (p.first->type != t0->type || p.first->ne[0] != c) return false;
    auto row = [&](int i) { return (const char *) rows_in[i].first->data + (size_t) rows_in[i].second * rows_in[i].first->nb[1]; };
    if (t0->type == GGML_TYPE_F16) {
        f32 = false;
        wg_fill(*this, (int) rows_in.size(), c, [&](int i, int k) { return ((const uint16_t *) row(i))[k]; });
    } else if (t0->type == GGML_TYPE_F32) {
        f32 = true;
        wg_fill(*this, (int) rows_in.size(), c, [&](int i, int k) { return ((const float *) row(i))[k]; });
    } else return false;
    return w != nullptr;
}
bool imi_wg::build(const ggml_tensor * t) {
    if (t->ne[2] != 1 || t->ne[3] != 1) return false;
    std::vector<std::pair<const ggml_tensor *, int>> r;
    for (int i = 0; i < (int) t->ne[1]; i++) r.push_back({ t, i });
    return build(r);
}
bool imi_wg::build(const float * m, int r, int c) {
    f32 = true;
    wg_fill(*this, r, c, [&](int i, int k) { return m[(size_t) i * c + k]; });
    return w != nullptr;
}
void imi_wg::release() { if (w) imi_big_free(w); w = nullptr; }

void imi_wg_group12(const imi_wg & W, const float * xt, int g, float * out) {
#ifdef IMI_AVX2
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0;
#define IMI_WG_STEP(LOAD)                                                                                                       \
    for (int k = 0; k < W.cols; k++) {                                                                                          \
        const __m256 wv = LOAD;                                                                                                 \
        const float * xk = xt + (size_t) k * 12;                                                                                \
        a0  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 0), a0);   a1  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 1), a1);  \
        a2  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 2), a2);   a3  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 3), a3);  \
        a4  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 4), a4);   a5  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 5), a5);  \
        a6  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 6), a6);   a7  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 7), a7);  \
        a8  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 8), a8);   a9  = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 9), a9);  \
        a10 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 10), a10); a11 = _mm256_fmadd_ps(wv, _mm256_broadcast_ss(xk + 11), a11); \
    }
    if (W.f32) {
        const float * w = (const float *) W.w + (size_t) g * W.cols * 8;
        IMI_WG_STEP(_mm256_loadu_ps(w + (size_t) k * 8))
    } else {
        const uint16_t * w = (const uint16_t *) W.w + (size_t) g * W.cols * 8;
        IMI_WG_STEP(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (w + (size_t) k * 8))))
    }
#undef IMI_WG_STEP
    _mm256_storeu_ps(out + 0, a0);  _mm256_storeu_ps(out + 8, a1);  _mm256_storeu_ps(out + 16, a2); _mm256_storeu_ps(out + 24, a3);
    _mm256_storeu_ps(out + 32, a4); _mm256_storeu_ps(out + 40, a5); _mm256_storeu_ps(out + 48, a6); _mm256_storeu_ps(out + 56, a7);
    _mm256_storeu_ps(out + 64, a8); _mm256_storeu_ps(out + 72, a9); _mm256_storeu_ps(out + 80, a10); _mm256_storeu_ps(out + 88, a11);
#else
    for (int c = 0; c < 12; c++) for (int r = 0; r < 8; r++) {
        float s = 0;
        for (int k = 0; k < W.cols; k++) {
            const size_t at = ((size_t) g * W.cols + k) * 8 + r;
            s += (W.f32 ? ((const float *) W.w)[at] : imi_h2f(((const uint16_t *) W.w)[at])) * xt[(size_t) k * 12 + c];
        }
        out[8 * c + r] = s;
    }
#endif
}
