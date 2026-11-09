// imi kernel test: Seoul's 4-bit tables (Q4_0) times one input, our layout vs ggml, on real weights.
//
// Our layout ("q4x2"): rows go in pairs. For each pair and each 32-weight block: 32 bytes, byte i = row A's code i in
// the low 4 bits and row B's code i in the high 4 bits; the two blocks' scales are kept as f32 side by side.
// Codes stay 0..15 (Q4_0 stores weight = (code - 8) * d). The input is quantized to int8 per 32 (scale dx), so
//   dot(row, x) = sum_j d_j * dx_j * (sum_i code_i * xq_i  -  8 * sum_i xq_i)
// The second sum depends only on the input: computed once per block and shared by every row, so the kernel never
// fixes signs per weight. maddubs takes the codes (unsigned) times xq (signed) directly.
//
//   q4x2_bench MODEL.gguf [threads]   checks against ggml's own dot (same int8 input) and times both
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"

#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct blk_q4_0 { ggml_fp16_t d; uint8_t qs[16]; };   // ggml's Q4_0: low nibbles = weights 0..15, high = 16..31

struct q4x2 {                 // one table in our layout
    int rows = 0, cols = 0, nb = 0;
    std::vector<uint8_t> codes;   // [rows/2][nb][32]
    std::vector<float>   scales;  // [rows/2][nb][2]
};

static q4x2 to_q4x2(const ggml_tensor * t) {
    q4x2 o; o.cols = (int) t->ne[0]; o.rows = (int) t->ne[1]; o.nb = o.cols / 32;
    o.codes.resize((size_t) o.rows / 2 * o.nb * 32);
    o.scales.resize((size_t) o.rows / 2 * o.nb * 2);
    const blk_q4_0 * w = (const blk_q4_0 *) t->data;
    for (int p = 0; p < o.rows / 2; p++) {
        for (int j = 0; j < o.nb; j++) {
            const blk_q4_0 & a = w[(size_t) (2 * p) * o.nb + j], & b = w[(size_t) (2 * p + 1) * o.nb + j];
            uint8_t * c = &o.codes[((size_t) p * o.nb + j) * 32];
            for (int i = 0; i < 32; i++) {
                const int ca = i < 16 ? (a.qs[i] & 15) : (a.qs[i - 16] >> 4);
                const int cb = i < 16 ? (b.qs[i] & 15) : (b.qs[i - 16] >> 4);
                c[i] = (uint8_t) (ca | (cb << 4));
            }
            o.scales[((size_t) p * o.nb + j) * 2 + 0] = ggml_fp16_to_fp32(a.d);
            o.scales[((size_t) p * o.nb + j) * 2 + 1] = ggml_fp16_to_fp32(b.d);
        }
    }
    return o;
}

struct qin { std::vector<int8_t> q; std::vector<float> d, corr; };  // corr_j = 8 * dx_j * sum(xq in block j)

static void quant_in(const float * x, int n, qin & o) {
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

static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// rows [p0, p1) pairs; 2 pairs (4 rows) in flight per loop so the core has independent work while maddubs waits
static void q4x2_rows(const q4x2 & W, const qin & X, float * y, int p0, int p1) {
    const __m256i lo = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    const int nb = W.nb;
    int p = p0;
    for (; p + 1 < p1; p += 2) {
        __m256 a0 = _mm256_setzero_ps(), b0 = a0, a1 = a0, b1 = a0;
        float c0a = 0, c0b = 0, c1a = 0, c1b = 0;
        const uint8_t * w0 = &W.codes[(size_t) p * nb * 32], * w1 = w0 + (size_t) nb * 32;
        const float * s0 = &W.scales[(size_t) p * nb * 2], * s1 = s0 + (size_t) nb * 2;
        for (int j = 0; j < nb; j++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *) &X.q[j * 32]);
            const __m256i v0 = _mm256_loadu_si256((const __m256i *) (w0 + j * 32));
            const __m256i v1 = _mm256_loadu_si256((const __m256i *) (w1 + j * 32));
            const __m256 dx = _mm256_set1_ps(X.d[j]);
            const __m256i pa0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(v0, lo), x), ones);
            const __m256i pb0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v0, 4), lo), x), ones);
            const __m256i pa1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(v1, lo), x), ones);
            const __m256i pb1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v1, 4), lo), x), ones);
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pa0), _mm256_mul_ps(_mm256_set1_ps(s0[2 * j]), dx), a0);
            b0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pb0), _mm256_mul_ps(_mm256_set1_ps(s0[2 * j + 1]), dx), b0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pa1), _mm256_mul_ps(_mm256_set1_ps(s1[2 * j]), dx), a1);
            b1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pb1), _mm256_mul_ps(_mm256_set1_ps(s1[2 * j + 1]), dx), b1);
            const float cj = X.corr[j];
            c0a += s0[2 * j] * cj; c0b += s0[2 * j + 1] * cj; c1a += s1[2 * j] * cj; c1b += s1[2 * j + 1] * cj;
        }
        y[2 * p + 0] = hsum(a0) - c0a; y[2 * p + 1] = hsum(b0) - c0b;
        y[2 * p + 2] = hsum(a1) - c1a; y[2 * p + 3] = hsum(b1) - c1b;
    }
    for (; p < p1; p++) {
        __m256 a = _mm256_setzero_ps(), b = a; float ca = 0, cb = 0;
        const uint8_t * w = &W.codes[(size_t) p * nb * 32]; const float * s = &W.scales[(size_t) p * nb * 2];
        for (int j = 0; j < nb; j++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *) &X.q[j * 32]);
            const __m256i v = _mm256_loadu_si256((const __m256i *) (w + j * 32));
            const __m256 dx = _mm256_set1_ps(X.d[j]);
            a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(v, lo), x), ones)), _mm256_mul_ps(_mm256_set1_ps(s[2 * j]), dx), a);
            b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v, 4), lo), x), ones)), _mm256_mul_ps(_mm256_set1_ps(s[2 * j + 1]), dx), b);
            ca += s[2 * j] * X.corr[j]; cb += s[2 * j + 1] * X.corr[j];
        }
        y[2 * p] = hsum(a) - ca; y[2 * p + 1] = hsum(b) - cb;
    }
}


// v2 ("q4x8"): 8 rows per group. Per group and block: 4 x 32 bytes (pairs packed as in q4x2), then the 8 rows'
// scales as 8 f32. The 8 integer row sums are folded into one register (hadd tree), so the float scaling, the
// input scale and the -8 correction are one multiply and two FMAs for all 8 rows instead of per row.
struct q4x8 { int rows = 0, cols = 0, nb = 0; std::vector<uint8_t> codes; std::vector<float> scales; };

static q4x8 to_q4x8(const ggml_tensor * t) {
    q4x8 o; o.cols = (int) t->ne[0]; o.rows = (int) t->ne[1]; o.nb = o.cols / 32;
    const int ng = o.rows / 8;
    o.codes.resize((size_t) ng * o.nb * 128); o.scales.resize((size_t) ng * o.nb * 8);
    const blk_q4_0 * w = (const blk_q4_0 *) t->data;
    auto code = [&](int r, int j, int i) { const blk_q4_0 & b = w[(size_t) r * o.nb + j]; return i < 16 ? (b.qs[i] & 15) : (b.qs[i - 16] >> 4); };
    for (int g = 0; g < ng; g++) for (int j = 0; j < o.nb; j++) {
        uint8_t * c = &o.codes[((size_t) g * o.nb + j) * 128];
        for (int k = 0; k < 4; k++) for (int i = 0; i < 32; i++)
            c[k * 32 + i] = (uint8_t) (code(8 * g + 2 * k, j, i) | (code(8 * g + 2 * k + 1, j, i) << 4));
        for (int r = 0; r < 8; r++) o.scales[((size_t) g * o.nb + j) * 8 + r] = ggml_fp16_to_fp32(w[(size_t) (8 * g + r) * o.nb + j].d);
    }
    return o;
}

static void q4x8_rows(const q4x8 & W, const qin & X, float * y, int g0, int g1) {
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

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: q4x2_bench MODEL.gguf [threads]\n"); return 1; }
    const int T = argc > 2 ? atoi(argv[2]) : 4;
    ggml_cpu_init();
    ggml_context * ctx = nullptr;
    gguf_init_params gp = { false, &ctx };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) return 1;

    // every Q4_0 matrix table of the brain except the word table
    std::vector<const ggml_tensor *> ts;
    size_t bytes = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const ggml_tensor * t = ggml_get_tensor(ctx, gguf_get_tensor_name(g, i));
        if (t->type == GGML_TYPE_Q4_0 && ggml_n_dims(t) == 2 && t->ne[1] % 8 == 0 && !strstr(t->name, "token_embd")) { ts.push_back(t); bytes += ggml_nbytes(t); }
    }
    std::vector<q4x2> ws; for (auto * t : ts) ws.push_back(to_q4x2(t));

    // check: our kernel vs ggml's vec_dot on the same int8 input (ggml's own q8_0 quantizer)
    double worst = 0;
    for (size_t k = 0; k < ts.size(); k += 17) {
        const ggml_tensor * t = ts[k]; const int n = (int) t->ne[0];
        std::vector<float> x(n); for (int i = 0; i < n; i++) x[i] = sinf(0.37f * i) * (1.0f + (i % 13) * 0.1f);
        qin X; quant_in(x.data(), n, X);
        std::vector<uint8_t> xg(ggml_row_size(GGML_TYPE_Q8_0, n));
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), xg.data(), n);
        std::vector<float> y(t->ne[1]); q4x2_rows(ws[k], X, y.data(), 0, (int) t->ne[1] / 2);
        double num = 0, den = 0;
        for (int r = 0; r < t->ne[1]; r++) {
            float ref; ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot(n, &ref, 0, (const char *) t->data + r * t->nb[1], 0, xg.data(), 0, 1);
            num += (y[r] - ref) * (y[r] - ref); den += (double) ref * ref;
        }
        worst = std::max(worst, sqrt(num / den));
    }
    printf("check vs ggml's 4-bit dot (same int8 input): worst table error %.2e\n", worst);
    std::vector<q4x8> w8; for (auto * t : ts) w8.push_back(to_q4x8(t));
    double worst8 = 0;
    for (size_t k = 0; k < ts.size(); k += 17) {
        const ggml_tensor * t = ts[k]; const int n = (int) t->ne[0];
        std::vector<float> x(n); for (int i = 0; i < n; i++) x[i] = sinf(0.37f * i) * (1.0f + (i % 13) * 0.1f);
        qin X; quant_in(x.data(), n, X);
        std::vector<uint8_t> xg(ggml_row_size(GGML_TYPE_Q8_0, n));
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), xg.data(), n);
        std::vector<float> y(t->ne[1]); q4x8_rows(w8[k], X, y.data(), 0, (int) t->ne[1] / 8);
        double num = 0, den = 0;
        for (int r = 0; r < t->ne[1]; r++) {
            float ref; ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot(n, &ref, 0, (const char *) t->data + r * t->nb[1], 0, xg.data(), 0, 1);
            num += (y[r] - ref) * (y[r] - ref); den += (double) ref * ref;
        }
        worst8 = std::max(worst8, sqrt(num / den));
    }
    printf("check q4x8: worst table error %.2e\n", worst8);

    // time: one brain step's worth of tables, T threads splitting each table's rows
    std::vector<std::vector<float>> xs, ys; std::vector<qin> qs(ws.size());
    for (auto & w : ws) { std::vector<float> x(w.cols, 0.01f); xs.push_back(x); ys.emplace_back(w.rows); }
    for (size_t k = 0; k < ws.size(); k++) quant_in(xs[k].data(), ws[k].cols, qs[k]);
    double best = 1e9;
    for (int rep = 0; rep < 10; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (int i = 0; i < T; i++) th.emplace_back([&, i] {
            for (size_t k = 0; k < ws.size(); k++) {
                const int np = ws[k].rows / 2, a = np * i / T, b = np * (i + 1) / T;
                q4x2_rows(ws[k], qs[k], ys[k].data(), a, b);
            }
        });
        for (auto & t : th) t.join();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    printf("q4x2, %d threads: %zu tables, %.0f MB in %.1f ms = %.1f GB/s\n", T, ws.size(), bytes / 1e6, best, bytes / 1e6 / best);
    best = 1e9;
    for (int rep = 0; rep < 10; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (int i = 0; i < T; i++) th.emplace_back([&, i] {
            for (size_t k = 0; k < w8.size(); k++) {
                const int ng = w8[k].rows / 8, a = ng * i / T, b = ng * (i + 1) / T;
                q4x8_rows(w8[k], qs[k], ys[k].data(), a, b);
            }
        });
        for (auto & t : th) t.join();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    printf("q4x8, %d threads: %.0f MB in %.1f ms = %.1f GB/s\n", T, bytes / 1e6, best, bytes / 1e6 / best);
    return 0;
}
