// imi-kq-bench: research (mixed bits, 2026-11-10). Can the engine read tables at 2, 3, 5, 6 and 8 bits about as fast
// as their size says, on the room computer's kind of processor? Times one input through a big table (far bigger than
// the processor's caches, so the table streams from memory every pass) with ggml's own row kernels for each type, and
// imi's 8-row 4-bit kernel for comparison, and a THROWAWAY 8-row 2-bit kernel (q2x8 below: unknown 1 of the plan,
// can a 2-bit table read at memory speed on this processor?).
//   imi-kq-bench threads [passes]      (pin cores from outside, e.g. taskset -c 0-3)
#include "tables.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace q2x8 {   // THROWAWAY prototype: Q2_K numbers (bit for bit) in an 8-row layout like imi's 4-bit tables.
// Per group of 8 rows and 256-column block: codes [16 sub-blocks][32 B] (byte 4r+i of sub-block s holds row r's
// columns 16s+4k+i for k = 0..3, 2 bits each at bit 2k), then the packed scale|min<<4 bytes [8 pairs][8 rows][2]
// (sub-blocks 2p and 2p+1), then d[8] and dmin[8] as 16-bit floats: 672 B = 8 x 84 B, the same size as Q2_K.
struct blk { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
static_assert(sizeof(blk) == 84, "q2_K block");
constexpr size_t GB = 672;
struct tab { int rows = 0, cols = 0, nb = 0; uint8_t * mem = nullptr; size_t bytes = 0; };
static inline int code_of(const blk & b, int e) { return (b.qs[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3; }
static tab build(const uint8_t * q1, int R0, int copies, int C) {   // rows r of the copies = rows r % R0 of q1
    tab t; t.rows = R0 * copies; t.cols = C; t.nb = C / 256;
    const size_t ng = t.rows / 8, rb = (size_t) t.nb * sizeof(blk);
    t.bytes = ng * t.nb * GB; t.mem = (uint8_t *) imi_big_alloc(t.bytes);
    for (size_t g = 0; g < ng; g++) for (int j = 0; j < t.nb; j++) {
        uint8_t * o = t.mem + (g * t.nb + j) * GB;
        for (int r = 0; r < 8; r++) {
            const blk & b = ((const blk *) (q1 + (size_t) ((8 * g + r) % R0) * rb))[j];
            for (int s = 0; s < 16; s++) for (int i = 0; i < 4; i++) {
                int v = 0;
                for (int k = 0; k < 4; k++) v |= code_of(b, 16 * s + 4 * k + i) << (2 * k);
                o[32 * s + 4 * r + i] = (uint8_t) v;
            }
            for (int p = 0; p < 8; p++) { o[512 + 16 * p + 2 * r] = b.scales[2 * p]; o[512 + 16 * p + 2 * r + 1] = b.scales[2 * p + 1]; }
            memcpy(o + 640 + 2 * r, &b.d, 2); memcpy(o + 656 + 2 * r, &b.dmin, 2);
        }
    }
    return t;
}
static inline void group(const tab & W, const imi_q8k & X, int g, float * out) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m15 = _mm_set1_epi8(15);
    const __m256i dup[2] = { _mm256_setr_epi8(0,1,0,1,4,5,4,5,8,9,8,9,12,13,12,13, 0,1,0,1,4,5,4,5,8,9,8,9,12,13,12,13),
                             _mm256_setr_epi8(2,3,2,3,6,7,6,7,10,11,10,11,14,15,14,15, 2,3,2,3,6,7,6,7,10,11,10,11,14,15,14,15) };
    __m256 acc = _mm256_setzero_ps();
    for (int j = 0; j < W.nb; j++) {
        const uint8_t * o = W.mem + ((size_t) g * W.nb + j) * GB;
        const int8_t * xq = X.q.data() + 256 * j;
        const int16_t * bs = X.bsums.data() + 16 * j;
        __m256i isum = _mm256_setzero_si256(), msum = _mm256_setzero_si256();
        for (int p = 0; p < 8; p++) {
            const __m128i scb = _mm_loadu_si128((const __m128i *) (o + 512 + 16 * p));
            const __m256i sc16 = _mm256_cvtepu8_epi16(_mm_and_si128(scb, m15));                     // [row: s 2p, 2p+1]
            const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(scb, 4), m15));
            int32_t bp; memcpy(&bp, bs + 2 * p, 4);
            msum = _mm256_add_epi32(msum, _mm256_madd_epi16(mn16, _mm256_set1_epi32(bp)));
            for (int h = 0; h < 2; h++) {
                const int s = 2 * p + h;
                const __m256i v = _mm256_loadu_si256((const __m256i *) (o + 32 * s));
                __m256i p16 = _mm256_setzero_si256();
                for (int k = 0; k < 4; k++) {
                    int32_t xa; memcpy(&xa, xq + 16 * s + 4 * k, 4);
                    p16 = _mm256_add_epi16(p16, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v, 2 * k), m3), _mm256_set1_epi32(xa)));
                }
                isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p16, _mm256_shuffle_epi8(sc16, dup[h])));
            }
        }
        const __m256 dx = _mm256_set1_ps(X.d[j]);
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (o + 640)));
        const __m256 m8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (o + 656)));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_mul_ps(d8, dx), acc);
        acc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(msum), _mm256_mul_ps(m8, dx), acc);
    }
    _mm256_storeu_ps(out, acc);
}
// variant B: one shift per 32 bytes instead of three. Codes k = 0, 2 are read as bits 0-1 of v and of v >> 4 (x1),
// k = 1, 3 as bits 2-3 of the same (x4, still inside maddubs' 16-bit range); the x4 sums are kept apart and divided by
// 4 at the end of the block (exact: every product is a multiple of 4).
static inline void group_b(const tab & W, const imi_q8k & X, int g, float * out) {
    const __m256i m03 = _mm256_set1_epi8(0x03), m0c = _mm256_set1_epi8(0x0c);
    const __m128i m15 = _mm_set1_epi8(15);
    const __m256i dup[2] = { _mm256_setr_epi8(0,1,0,1,4,5,4,5,8,9,8,9,12,13,12,13, 0,1,0,1,4,5,4,5,8,9,8,9,12,13,12,13),
                             _mm256_setr_epi8(2,3,2,3,6,7,6,7,10,11,10,11,14,15,14,15, 2,3,2,3,6,7,6,7,10,11,10,11,14,15,14,15) };
    __m256 acc = _mm256_setzero_ps();
    for (int j = 0; j < W.nb; j++) {
        const uint8_t * o = W.mem + ((size_t) g * W.nb + j) * GB;
        const int8_t * xq = X.q.data() + 256 * j;
        const int16_t * bs = X.bsums.data() + 16 * j;
        __m256i i1 = _mm256_setzero_si256(), i4 = _mm256_setzero_si256(), msum = _mm256_setzero_si256();
        for (int p = 0; p < 8; p++) {
            const __m128i scb = _mm_loadu_si128((const __m128i *) (o + 512 + 16 * p));
            const __m256i sc16 = _mm256_cvtepu8_epi16(_mm_and_si128(scb, m15));
            const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(scb, 4), m15));
            int32_t bp; memcpy(&bp, bs + 2 * p, 4);
            msum = _mm256_add_epi32(msum, _mm256_madd_epi16(mn16, _mm256_set1_epi32(bp)));
            for (int h = 0; h < 2; h++) {
                const int s = 2 * p + h;
                const __m256i v = _mm256_loadu_si256((const __m256i *) (o + 32 * s)), v4 = _mm256_srli_epi16(v, 4);
                const int8_t * xs = xq + 16 * s;
                int32_t x0, x1, x2, x3; memcpy(&x0, xs, 4); memcpy(&x1, xs + 4, 4); memcpy(&x2, xs + 8, 4); memcpy(&x3, xs + 12, 4);
                const __m256i a1 = _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(v, m03), _mm256_set1_epi32(x0)),
                                                    _mm256_maddubs_epi16(_mm256_and_si256(v4, m03), _mm256_set1_epi32(x2)));
                const __m256i a4 = _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(v, m0c), _mm256_set1_epi32(x1)),
                                                    _mm256_maddubs_epi16(_mm256_and_si256(v4, m0c), _mm256_set1_epi32(x3)));
                const __m256i sc = _mm256_shuffle_epi8(sc16, dup[h]);
                i1 = _mm256_add_epi32(i1, _mm256_madd_epi16(a1, sc));
                i4 = _mm256_add_epi32(i4, _mm256_madd_epi16(a4, sc));
            }
        }
        const __m256i isum = _mm256_add_epi32(i1, _mm256_srai_epi32(i4, 2));
        const __m256 dx = _mm256_set1_ps(X.d[j]);
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (o + 640)));
        const __m256 m8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (o + 656)));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_mul_ps(d8, dx), acc);
        acc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(msum), _mm256_mul_ps(m8, dx), acc);
    }
    _mm256_storeu_ps(out, acc);
}
}   // namespace q2x8

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void run(int n_th, int n, const std::function<void(int, int)> & f) {   // rows [a, b) per thread
    std::vector<std::thread> th;
    for (int t = 0; t < n_th; t++) th.emplace_back([&, t] { f((int) ((long long) n * t / n_th), (int) ((long long) n * (t + 1) / n_th)); });
    for (auto & t : th) t.join();
}

int main(int argc, char ** argv) {
    const int n_th = argc > 1 ? atoi(argv[1]) : 3, passes = argc > 2 ? atoi(argv[2]) : 15;
    const int C = 2048, R0 = 7680, COPIES = 40, R = R0 * COPIES;   // 629M numbers, like one full read of the brain
    ggml_cpu_init();
    std::mt19937 rng(1110);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> w((size_t) R0 * C), x(C);
    for (auto & v : w) v = nd(rng);
    for (auto & v : x) v = nd(rng) * 50.0f;
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 };
    printf("threads %d, table %d x %d (%d copies of %d rows), %d passes, median ms\n", n_th, R, C, COPIES, R0, passes);
    std::vector<float> out(R), ref(R);
    for (ggml_type ty : types) {
        const size_t rb = ggml_row_size(ty, C), one = rb * R0;
        std::vector<uint8_t> q1(one);
        ggml_quantize_chunk(ty, w.data(), q1.data(), 0, R0, C, nullptr);
        uint8_t * big = (uint8_t *) imi_big_alloc(one * COPIES);
        for (int c = 0; c < COPIES; c++) memcpy(big + one * c, q1.data(), one);
        const auto * tc = ggml_get_type_traits_cpu(ty);
        std::vector<uint8_t> xq(ggml_row_size(tc->vec_dot_type, C));
        ggml_get_type_traits_cpu(tc->vec_dot_type)->from_float(x.data(), xq.data(), C);
        std::vector<double> ms;
        for (int p = 0; p < passes; p++) {
            const double t0 = now();
            run(n_th, R, [&](int a, int b) { for (int r = a; r < b; r++) tc->vec_dot(C, &out[r], 0, big + rb * r, 0, xq.data(), 0, 1); });
            ms.push_back(1e3 * (now() - t0));
        }
        std::sort(ms.begin(), ms.end());
        const double m = ms[ms.size() / 2], gb = (double) one * COPIES / 1e9;
        printf("ggml %-5s %5.3f bits  %7.1f MB  %7.2f ms  %5.1f GB/s  (%.2f ms per 100 MB)\n", ggml_type_name(ty),
               8.0 * rb / C, 1e3 * gb, m, gb / (m / 1e3), m / (10 * gb));
        if (ty == GGML_TYPE_Q4_0) {   // imi's own 8-row layout, same numbers
            ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, C, R0);
            t->data = q1.data();
            std::vector<std::pair<const ggml_tensor *, int>> order;
            for (int c = 0; c < COPIES; c++) for (int r = 0; r < R0; r++) order.push_back({ t, r });
            imi_q4 W; W.build(order);
            imi_q8 X; X.from(x.data(), C);
            std::vector<double> ms2;
            for (int p = 0; p < passes; p++) {
                const double t0 = now();
                run(n_th, R / 8, [&](int a, int b) { for (int g = a; g < b; g++) imi_q4_group(W, X, g, &ref[(size_t) g * 8]); });
                ms2.push_back(1e3 * (now() - t0));
            }
            std::sort(ms2.begin(), ms2.end());
            double md = 0; for (int r = 0; r < R; r++) md = std::max(md, (double) std::fabs(ref[r] - out[r]));
            const double m2 = ms2[ms2.size() / 2], gb2 = (double) W.bytes / 1e9;
            printf("imi  q4_0  (8-row)       %7.1f MB  %7.2f ms  %5.1f GB/s  (%.2f ms per 100 MB)  max diff vs ggml %.2e\n",
                   1e3 * gb2, m2, gb2 / (m2 / 1e3), m2 / (10 * gb2), md);
            W.release(); ggml_free(ctx);
        }
        if (ty == GGML_TYPE_Q2_K) {   // the throwaway 8-row 2-bit kernel, same numbers
            q2x8::tab W = q2x8::build(q1.data(), R0, COPIES, C);
            imi_q8k X; X.from(x.data(), C);
            std::vector<double> ms2;
            for (int p = 0; p < passes; p++) {
                const double t0 = now();
                run(n_th, R / 8, [&](int a, int b) { for (int g = a; g < b; g++) q2x8::group(W, X, g, &ref[(size_t) g * 8]); });
                ms2.push_back(1e3 * (now() - t0));
            }
            std::sort(ms2.begin(), ms2.end());
            std::vector<float> wd(C);   // exact answer for the first rows: the table's own numbers times the input
            double md = 0, mg = 0, ms = 0;
            for (int r = 0; r < R0; r++) {
                ggml_get_type_traits(ty)->to_float(q1.data() + rb * r, wd.data(), C);
                double e = 0; for (int k = 0; k < C; k++) e += (double) wd[k] * x[k];
                md = std::max(md, std::fabs(ref[r] - e)); mg = std::max(mg, std::fabs(out[r] - e)); ms = std::max(ms, std::fabs(e));
            }
            std::vector<double> ms3;
            std::vector<float> refb(R);
            for (int p = 0; p < passes; p++) {
                const double t0 = now();
                run(n_th, R / 8, [&](int a, int b) { for (int g = a; g < b; g++) q2x8::group_b(W, X, g, &refb[(size_t) g * 8]); });
                ms3.push_back(1e3 * (now() - t0));
            }
            std::sort(ms3.begin(), ms3.end());
            double dab = 0; for (int r = 0; r < R; r++) dab = std::max(dab, (double) std::fabs(refb[r] - ref[r]));
            printf("imi  q2_K  (8-row, variant B: masks) %7.2f ms  %5.1f GB/s  max diff vs variant A %.2e\n",
                   ms3[ms3.size() / 2], (double) W.bytes / 1e9 / (ms3[ms3.size() / 2] / 1e3), dab);
            const double m2 = ms2[ms2.size() / 2], gb2 = (double) W.bytes / 1e9;
            printf("imi  q2_K  (8-row, throwaway) %7.1f MB  %7.2f ms  %5.1f GB/s  (%.2f ms per 100 MB)  max error vs exact %.2e (ggml's %.2e; outputs up to %.2e)\n",
                   1e3 * gb2, m2, gb2 / (m2 / 1e3), m2 / (10 * gb2), md, mg, ms);
            imi_big_free(W.mem);
        }
        imi_big_free(big);
        fflush(stdout);
    }
    return 0;
}
