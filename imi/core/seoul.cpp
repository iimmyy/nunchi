#include "seoul.h"
#include "vec.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define IMI_AVX2 1
#endif

namespace {
inline void norm_mul(const float * x, const float * w, float * o, int n, float eps) { imi_norm_mul(x, w, o, n, eps); }
// NEOX rotation: pairs (i, i + n/2), angle pos * base^(-2i/n) / freq_factor[i]
void rope(float * v, int n, int pos, float base, const float * ff) {
    const float scale = powf(base, -2.0f / (float) n);
    float th = (float) pos;
    for (int i = 0; i < n / 2; i++) {
        const float a = th / (ff ? ff[i] : 1.0f), c = cosf(a), s = sinf(a);
        const float x0 = v[i], x1 = v[i + n / 2];
        v[i] = x0 * c - x1 * s; v[i + n / 2] = x0 * s + x1 * c;
        th *= scale;
    }
}
inline float dot_h(const float * q, const uint16_t * k, int n) {
#ifdef IMI_AVX2
    __m256 a = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (k + i))), a);
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r)); r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
#else
    float s = 0; for (int i = 0; i < n; i++) s += q[i] * imi_h2f(k[i]); return s;
#endif
}
inline void axpy_h(float p, const uint16_t * v, float * o, int n) {
#ifdef IMI_AVX2
    const __m256 pp = _mm256_set1_ps(p);
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(o + i, _mm256_fmadd_ps(pp, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (v + i))), _mm256_loadu_ps(o + i)));
#else
    for (int i = 0; i < n; i++) o[i] += p * imi_h2f(v[i]);
#endif
}
inline float silu(float x) { return x / (1.0f + expf(-x)); }
const float * f32(const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; }
}

bool imi_seoul::load(const std::string & path, int threads, int ctx_len, imi_team * shared) {
    if (!f.load(path, false)) return false;
    const ggml_tensor * te = f.get("token_embd.weight");
    if (te->type != GGML_TYPE_Q6_K) { fprintf(stderr, "imi-seoul: word table must be Q6_K\n"); return false; }
    tok = (const uint8_t *) te->data; tok_row = te->nb[1];
    n_embd = (int) te->ne[0]; n_vocab = (int) te->ne[1];
    out_norm   = f32(f.get("output_norm.weight"));
    rope_freqs = f32(f.get("rope_freqs.weight", false));
    eps           = f.key_f("lfm3.attention.layer_norm_rms_epsilon", 1e-5f);
    rope_base     = f.key_f("lfm3.rope.freq_base", 1e6f);
    rope_base_swa = f.key_f("lfm3.rope.freq_base_swa", 1e4f);
    n_swa         = (int) f.key_i("lfm3.attention.sliding_window", 0);
    d_conv        = (int) f.key_i("lfm3.shortconv.l_cache", 7) - 1;
    n_head        = (int) f.key_i("lfm3.attention.head_count", 16);
    head_dim      = (int) f.key_i("lfm3.attention.key_length", 128);
    const int n_layer = (int) f.key_i("lfm3.block_count", 0);
    std::vector<bool> swa(n_layer, false);
    const int64_t kswa = gguf_find_key(f.meta, "lfm3.attention.sliding_window_pattern");
    if (kswa >= 0) {
        const bool * a = (const bool *) gguf_get_arr_data(f.meta, kswa);
        for (int i = 0; i < n_layer && i < (int) gguf_get_arr_n(f.meta, kswa); i++) swa[i] = a[i];
    }
    auto rows = [](std::vector<std::pair<const ggml_tensor *, int>> & o, const ggml_tensor * t, int r0, int n) {
        for (int r = r0; r < r0 + n; r++) o.push_back({ t, r });
    };
    auto whole = [&](imi_q4 & dst, std::vector<const ggml_tensor *> parts) {
        std::vector<std::pair<const ggml_tensor *, int>> o;
        for (auto * p : parts) rows(o, p, 0, (int) p->ne[1]);
        return dst.build(o);
    };
    size_t bytes = 0;
    int n_attn = 0;
    layers.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        layer & L = layers[il];
        L.attn_norm      = f32(f.get(p + "attn_norm.weight"));
        L.post_attn_norm = f32(f.get(p + "post_attention_norm.weight"));
        L.ffn_norm       = f32(f.get(p + "ffn_norm.weight"));
        L.post_ffn_norm  = f32(f.get(p + "post_ffw_norm.weight"));
        bool ok = true;
        if (f.get(p + "shortconv.conv.weight", false)) {
            L.conv = true;
            {   // file: [channel][tap]; ours: [tap][channel]
                const ggml_tensor * cw = f.get(p + "shortconv.conv.weight");
                const int K = (int) cw->ne[0], C = (int) cw->ne[1];
                conv_w_t.emplace_back((size_t) K * C);
                for (int e = 0; e < C; e++) for (int k = 0; k < K; k++) conv_w_t.back()[(size_t) k * C + e] = ((const float *) cw->data)[(size_t) e * K + k];
                L.conv_w = conv_w_t.back().data();
            }
            ok &= whole(L.in_proj, { f.get(p + "shortconv.in_proj.weight") });
            ok &= whole(L.out_proj, { f.get(p + "shortconv.out_proj.weight") });
        } else {
            const ggml_tensor * k = f.get(p + "attn_k.weight");
            ok &= whole(L.qkvg, { f.get(p + "attn_q.weight"), k, f.get(p + "attn_v.weight"), f.get(p + "attn_gate.weight") });
            ok &= whole(L.o, { f.get(p + "attn_output.weight") });
            L.q_norm = f32(f.get(p + "attn_q_norm.weight"));
            L.k_norm = f32(f.get(p + "attn_k_norm.weight"));
            L.swa = swa[il];
            L.ai = n_attn++;
            n_head_kv = (int) (k->ne[1] / head_dim);
        }
        // gate and up interleaved by 8-row group: one worker reads a gate group and its up group and writes silu(g)*u
        const ggml_tensor * fg = f.get(p + "ffn_gate.weight"), * fu = f.get(p + "ffn_up.weight");
        n_ff = (int) fg->ne[1];
        std::vector<std::pair<const ggml_tensor *, int>> o;
        for (int t = 0; t < n_ff; t += 8) { rows(o, fg, t, 8); rows(o, fu, t, 8); }
        ok &= L.gate_up.build(o);
        ok &= whole(L.down, { f.get(p + "ffn_down.weight") });
        if (!ok) { fprintf(stderr, "imi-seoul: layer %d has a table our kernels do not read\n", il); return false; }
        for (imi_q4 * w : { &L.in_proj, &L.out_proj, &L.qkvg, &L.o, &L.gate_up, &L.down }) bytes += w->bytes;
    }
    n_ctx = ctx_len;
    const int kvd = n_head_kv * head_dim;
    k_cache.assign(n_attn, std::vector<uint16_t>((size_t) n_ctx * kvd));
    v_cache.assign(n_attn, std::vector<uint16_t>((size_t) n_ctx * kvd));
    conv_state.assign(n_layer, {});
    for (int il = 0; il < n_layer; il++) if (layers[il].conv) conv_state[il].assign((size_t) n_embd * d_conv, 0.0f);

    n_threads = threads;
    per.resize(n_threads);
    const int wide = std::max({ n_embd, n_ff, n_ctx });
    for (auto & T : per) { T.x.resize(n_embd); T.h.resize(wide); T.t.resize(wide); T.u.resize(wide); }
    bcx.resize(3 * n_embd); qkv.resize(n_head * head_dim + 2 * kvd + n_head); att.resize(n_head * head_dim);
    y.resize(n_embd); act.resize(n_ff); fo.resize(n_embd);
    hidden.resize(n_embd); logits.resize(n_vocab);
    ctr.reset(new std::atomic<int>[n_ctr]);
    if (shared) { team = shared; n_threads = shared->n; per.resize(n_threads); for (auto & T : per) { T.x.resize(n_embd); T.h.resize(wide); T.t.resize(wide); T.u.resize(wide); } }
    else team->start(n_threads);
    fprintf(stderr, "imi-seoul: %d layers (%d attention, %d heads / %d kv x %d, window %d), %d words, our layout %zu MB, %d threads\n",
            n_layer, n_attn, n_head, n_head_kv, head_dim, n_swa, n_vocab, bytes >> 20, n_threads);
    return true;
}

void imi_seoul::reset() {
    pos = 0;
    for (auto & s : conv_state) std::fill(s.begin(), s.end(), 0.0f);
}

void imi_seoul::step(int32_t token, const float * embd, bool want_logits) {
    if (pos >= n_ctx) { fprintf(stderr, "imi-seoul: context full (%d)\n", n_ctx); return; }
    for (int i = 0; i < n_ctr; i++) ctr[i].store(0, std::memory_order_relaxed);
    team->run([&](int tid) { run(tid, token, embd, want_logits); });
    pos++;
}

void imi_seoul::run(int tid, int32_t token, const float * embd, bool want_logits) {
    per_thread & T = per[tid];
    const int E = n_embd, D = head_dim, kvd = n_head_kv * D, grp = n_head / n_head_kv;
    float * x = T.x.data();
    if (token >= 0) imi_q6k_dequant_row(tok + (size_t) token * tok_row, x, E / 256);
    else memcpy(x, embd, (size_t) E * sizeof(float));

    int c = 0;
    const bool timing = tid == 0;
    auto table = [&](const imi_q4 & W, float * out) {
        std::atomic<int> & k = ctr[c++];
        const auto t0 = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        const int ng = W.rows / 8;
        for (int g = k.fetch_add(4, std::memory_order_relaxed); g < ng; g = k.fetch_add(4, std::memory_order_relaxed)) {
            const int e = std::min(ng, g + 4);
            for (int gg = g; gg < e; gg++) imi_q4_group(W, T.qi, gg, out + 8 * gg);
        }
        if (timing) t_tables += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    auto table_ffn = [&](const imi_q4 & W, float * out) {   // groups come in (gate, up) pairs; chunks start on a pair
        std::atomic<int> & k = ctr[c++];
        const auto t0 = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        const int ng = W.rows / 8;
        float a[8], b[8];
        for (int g = k.fetch_add(4, std::memory_order_relaxed); g < ng; g = k.fetch_add(4, std::memory_order_relaxed)) {
            const int e = std::min(ng, g + 4);
            for (int gg = g; gg < e; gg += 2) {
                imi_q4_group(W, T.qi, gg, a); imi_q4_group(W, T.qi, gg + 1, b);
                float * o = out + 8 * (gg / 2);
                for (int i = 0; i < 8; i++) o[i] = silu(a[i]) * b[i];
            }
        }
        if (timing) t_tables += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    auto sync = [&] {
        if (!timing) { team->sync(); return; }
        const auto t0 = std::chrono::steady_clock::now();
        team->sync();
        t_sync += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    for (size_t il = 0; il < layers.size(); il++) {
        const layer & L = layers[il];
        norm_mul(x, L.attn_norm, T.h.data(), E, eps);
        T.qi.from(T.h.data(), E);
        if (L.conv) {
            table(L.in_proj, bcx.data());
            sync();
            const float * b = bcx.data(), * cc = b + E, * xx = b + 2 * E, * st = conv_state[il].data(), * w = L.conv_w;
            float * u = T.u.data();
            for (int e = 0; e < E; e++) u[e] = b[e] * xx[e] * w[d_conv * E + e];
            for (int k = 0; k < d_conv; k++) { const float * sk = st + k * E, * wk = w + k * E; for (int e = 0; e < E; e++) u[e] += sk[e] * wk[e]; }
            for (int e = 0; e < E; e++) u[e] *= cc[e];
            T.qi.from(u, E);
            table(L.out_proj, y.data());
            sync();
            if (tid == 0) {   // the conv memory slides one step (every thread has read it by now)
                float * sw = conv_state[il].data();
                memmove(sw, sw + E, (size_t) (d_conv - 1) * E * sizeof(float));
                for (int e = 0; e < E; e++) sw[(d_conv - 1) * E + e] = b[e] * xx[e];
            }
        } else {
            table(L.qkvg, qkv.data());
            sync();
            const auto ta = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
            const float base = L.swa ? rope_base_swa : rope_base;
            const float * ff = L.swa ? nullptr : rope_freqs;
            uint16_t * kc = k_cache[L.ai].data(), * vc = v_cache[L.ai].data();
            const int s0 = (L.swa && n_swa > 0) ? std::max(0, pos - n_swa + 1) : 0, nk = pos - s0 + 1;
            const float scale = 1.0f / sqrtf((float) D);
            for (int g = tid; g < n_head_kv; g += n_threads) {      // each thread: whole key/value heads
                float kk[512];
                norm_mul(qkv.data() + n_head * D + g * D, L.k_norm, kk, D, eps);
                rope(kk, D, pos, base, ff);
                const float * vv = qkv.data() + n_head * D + kvd + g * D;
                for (int i = 0; i < D; i++) { kc[(size_t) pos * kvd + g * D + i] = imi_f2h(kk[i]); vc[(size_t) pos * kvd + g * D + i] = imi_f2h(vv[i]); }
                for (int hh = 0; hh < grp; hh++) {
                    const int h = g * grp + hh;
                    float q[512], o[512];
                    norm_mul(qkv.data() + h * D, L.q_norm, q, D, eps);
                    rope(q, D, pos, base, ff);
                    float * sc = T.t.data(), mx = -INFINITY;
                    for (int t = 0; t < nk; t++) { sc[t] = dot_h(q, kc + (size_t) (s0 + t) * kvd + g * D, D) * scale; mx = std::max(mx, sc[t]); }
                    double sum = 0;
                    for (int t = 0; t < nk; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                    const float inv = (float) (1.0 / sum);
                    std::fill(o, o + D, 0.0f);
                    for (int t = 0; t < nk; t++) axpy_h(sc[t] * inv, vc + (size_t) (s0 + t) * kvd + g * D, o, D);
                    const float gate = 1.0f / (1.0f + expf(-qkv[n_head * D + 2 * kvd + h]));
                    for (int i = 0; i < D; i++) att[h * D + i] = o[i] * gate;
                }
            }
            if (timing) t_attn += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
            sync();
            T.qi.from(att.data(), n_head * D);
            table(L.o, y.data());
            sync();
        }
        imi_norm_mul_add(y.data(), L.post_attn_norm, x, E, eps);

        norm_mul(x, L.ffn_norm, T.h.data(), E, eps);
        T.qi.from(T.h.data(), E);
        table_ffn(L.gate_up, act.data());
        sync();
        T.qi.from(act.data(), n_ff);
        table(L.down, fo.data());
        sync();
        imi_norm_mul_add(fo.data(), L.post_ffn_norm, x, E, eps);
    }
    float * hid = T.h.data();
    norm_mul(x, out_norm, hid, E, eps);
    if (tid == 0) memcpy(hidden.data(), hid, (size_t) E * sizeof(float));
    if (want_logits) {
        T.qk.from(hid, E);
        std::atomic<int> & k = ctr[c++];
        const int nb = E / 256, chunk = 64;
        for (int r = k.fetch_add(chunk, std::memory_order_relaxed); r < n_vocab; r = k.fetch_add(chunk, std::memory_order_relaxed)) {
            const int e = std::min(n_vocab, r + chunk);
            for (int i = r; i < e; i++) logits[i] = imi_q6k_row(tok + (size_t) i * tok_row, T.qk, nb);
        }
    }
}

void imi_seoul::step_batch(const int32_t * tokens, const float * embd, int n, bool want_logits, float * hidden_all) {
    if (bX.empty()) {
        const int kvd = n_head_kv * head_dim;
        bX.resize((size_t) NB * n_embd); bH.resize((size_t) NB * std::max(n_embd, n_ff)); bY.resize((size_t) NB * n_embd);
        bBig.resize((size_t) NB * std::max(3 * n_embd, n_head * head_dim + 2 * kvd + n_head)); bAct.resize((size_t) NB * n_ff);
    }
    for (int i0 = 0; i0 < n; i0 += NB) {
        const int m = std::min(NB, n - i0);
        if (pos + m > n_ctx) { fprintf(stderr, "imi-seoul: context full (%d)\n", n_ctx); return; }
        for (int i = 0; i < n_ctr; i++) ctr[i].store(0, std::memory_order_relaxed);
        team->run([&](int tid) {
            run_batch(tid, tokens ? tokens + i0 : nullptr, embd ? embd + (size_t) i0 * n_embd : nullptr, m,
                      hidden_all ? hidden_all + (size_t) i0 * n_embd : nullptr);
        });
        pos += m;
    }
    // the last position: its hidden state is already in `hidden`; its word scores if asked
    if (want_logits) {
        for (int i = 0; i < n_ctr; i++) ctr[i].store(0, std::memory_order_relaxed);
        team->run([&](int tid) {
            per_thread & T = per[tid];
            T.qk.from(hidden.data(), n_embd);
            const int nb = n_embd / 256, chunk = 64;
            for (int r = ctr[0].fetch_add(chunk); r < n_vocab; r = ctr[0].fetch_add(chunk))
                for (int i = r; i < std::min(n_vocab, r + chunk); i++) logits[i] = imi_q6k_row(tok + (size_t) i * tok_row, T.qk, nb);
        });
    }
}

void imi_seoul::run_batch(int tid, const int32_t * tokens, const float * embd, int n, float * hidden_all) {
    const int E = n_embd, D = head_dim, kvd = n_head_kv * D, grp = n_head / n_head_kv, nth = n_threads;
    const int p0 = tid * n / nth, p1 = (tid + 1) * n / nth;          // this thread's positions for the small math
    float * X = bX.data(), * H = bH.data(), * Y = bY.data(), * B = bBig.data(), * A = bAct.data();
    int c = 0;
    auto sync = [&] { team->sync(); };
    // one table times all n inputs in bq: rows split in chunks of groups
    auto table = [&](const imi_q4 & W, float * o, int ldo) {
        std::atomic<int> & k = ctr[c++];
        const int ng = W.rows / 8;
        float tmp[64 * 8];
        for (int g = k.fetch_add(1, std::memory_order_relaxed); g < ng; g = k.fetch_add(1, std::memory_order_relaxed)) {
            imi_q4_group_batch(W, bq, g, tmp);
            for (int p = 0; p < n; p++) memcpy(o + (size_t) p * ldo + 8 * g, tmp + 8 * p, 8 * sizeof(float));
        }
    };
    auto table_ffn = [&](const imi_q4 & W, float * o) {
        std::atomic<int> & k = ctr[c++];
        const int ng = W.rows / 8;
        float a[64 * 8], b[64 * 8];
        for (int g = 2 * k.fetch_add(1, std::memory_order_relaxed); g < ng; g = 2 * k.fetch_add(1, std::memory_order_relaxed)) {
            imi_q4_group_batch(W, bq, g, a); imi_q4_group_batch(W, bq, g + 1, b);
            for (int p = 0; p < n; p++) { float * op = o + (size_t) p * n_ff + 4 * g; for (int r = 0; r < 8; r++) op[r] = silu(a[8 * p + r]) * b[8 * p + r]; }
        }
    };
    if (tid == 0) bq.shape(n, E);
    for (int p = p0; p < p1; p++) {
        if (tokens) imi_q6k_dequant_row(tok + (size_t) tokens[p] * tok_row, X + (size_t) p * E, E / 256);
        else memcpy(X + (size_t) p * E, embd + (size_t) p * E, (size_t) E * sizeof(float));
    }
    sync();
    for (size_t il = 0; il < layers.size(); il++) {
        const layer & L = layers[il];
        if (tid == 0) bq.shape(n, E);
        sync();
        for (int p = p0; p < p1; p++) { imi_norm_mul(X + (size_t) p * E, L.attn_norm, H + (size_t) p * E, E, eps); bq.set(p, H + (size_t) p * E); }
        sync();
        if (L.conv) {
            table(L.in_proj, B, 3 * E);
            sync();
            // the conv runs along the positions; each thread takes a slice of channels (and their memory)
            const int e0 = tid * E / nth, e1 = (tid + 1) * E / nth, K = d_conv + 1;
            float * st = conv_state[il].data();
            const float * w = L.conv_w;
            for (int p = 0; p < n; p++) {
                const float * b = B + (size_t) p * 3 * E, * cc = b + E, * xx = b + 2 * E;
                float * u = H + (size_t) p * E;
                for (int e = e0; e < e1; e++) {
                    const float bx = b[e] * xx[e];
                    float s = bx * w[(size_t) d_conv * E + e];
                    for (int k = 0; k < d_conv; k++) s += st[(size_t) k * E + e] * w[(size_t) k * E + e];
                    u[e] = cc[e] * s;
                    for (int k = 0; k + 1 < d_conv; k++) st[(size_t) k * E + e] = st[(size_t) (k + 1) * E + e];
                    st[(size_t) (d_conv - 1) * E + e] = bx;
                }
            }
            (void) K;
            sync();
            for (int p = p0; p < p1; p++) bq.set(p, H + (size_t) p * E);
            sync();
            table(L.out_proj, Y, E);
            sync();
        } else {
            const int nr = n_head * D + 2 * kvd + n_head;
            table(L.qkvg, B, nr);
            sync();
            const float base = L.swa ? rope_base_swa : rope_base;
            const float * ff = L.swa ? nullptr : rope_freqs;
            uint16_t * kc = k_cache[L.ai].data(), * vc = v_cache[L.ai].data();
            for (int p = p0; p < p1; p++) {          // this batch's keys and values into the attention memory
                const float * r = B + (size_t) p * nr;
                for (int g = 0; g < n_head_kv; g++) {
                    float kk[512];
                    imi_norm_mul(r + n_head * D + g * D, L.k_norm, kk, D, eps);
                    rope(kk, D, pos + p, base, ff);
                    for (int i = 0; i < D; i++) { kc[(size_t) (pos + p) * kvd + g * D + i] = imi_f2h(kk[i]); vc[(size_t) (pos + p) * kvd + g * D + i] = imi_f2h(r[n_head * D + kvd + g * D + i]); }
                }
            }
            sync();
            const float scale = 1.0f / sqrtf((float) D);
            std::vector<float> sc(n_ctx);
            for (int u = tid; u < n * n_head; u += nth) {    // (position, head) pairs
                const int p = u / n_head, h = u % n_head, g = h / grp, at = pos + p;
                const int s0 = (L.swa && n_swa > 0) ? std::max(0, at - n_swa + 1) : 0, nk = at - s0 + 1;
                const float * r = B + (size_t) p * nr;
                float q[512], o[512];
                imi_norm_mul(r + h * D, L.q_norm, q, D, eps);
                rope(q, D, at, base, ff);
                float mx = -INFINITY;
                for (int t = 0; t < nk; t++) { sc[t] = dot_h(q, kc + (size_t) (s0 + t) * kvd + g * D, D) * scale; mx = std::max(mx, sc[t]); }
                double sum = 0;
                for (int t = 0; t < nk; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                const float inv = (float) (1.0 / sum);
                std::fill(o, o + D, 0.0f);
                for (int t = 0; t < nk; t++) axpy_h(sc[t] * inv, vc + (size_t) (s0 + t) * kvd + g * D, o, D);
                const float gate = 1.0f / (1.0f + expf(-r[n_head * D + 2 * kvd + h]));
                float * out = H + (size_t) p * (n_head * D) + h * D;
                for (int i = 0; i < D; i++) out[i] = o[i] * gate;
            }
            sync();
            for (int p = p0; p < p1; p++) bq.set(p, H + (size_t) p * (n_head * D));
            sync();
            table(L.o, Y, E);
            sync();
        }
        for (int p = p0; p < p1; p++) {
            imi_norm_mul_add(Y + (size_t) p * E, L.post_attn_norm, X + (size_t) p * E, E, eps);
            imi_norm_mul(X + (size_t) p * E, L.ffn_norm, H + (size_t) p * E, E, eps);
            bq.set(p, H + (size_t) p * E);
        }
        sync();
        table_ffn(L.gate_up, A);
        sync();
        if (tid == 0) bq.shape(n, n_ff);
        sync();
        for (int p = p0; p < p1; p++) bq.set(p, A + (size_t) p * n_ff);
        sync();
        table(L.down, Y, E);
        sync();
        for (int p = p0; p < p1; p++) imi_norm_mul_add(Y + (size_t) p * E, L.post_ffn_norm, X + (size_t) p * E, E, eps);
        sync();
    }
    for (int p = p0; p < p1; p++) {
        float * hp = H + (size_t) p * E;
        imi_norm_mul(X + (size_t) p * E, out_norm, hp, E, eps);
        if (hidden_all) memcpy(hidden_all + (size_t) p * E, hp, (size_t) E * sizeof(float));
        if (p == n - 1) memcpy(hidden.data(), hp, (size_t) E * sizeof(float));
    }
}

double imi_seoul::bench_tables(int ncol, int reps) {
    std::vector<const imi_q4 *> ws;
    for (auto & L : layers) for (const imi_q4 * w : { &L.in_proj, &L.out_proj, &L.qkvg, &L.o, &L.gate_up, &L.down }) if (w->rows) ws.push_back(w);
    std::vector<float> o1(32768), o2(32768);
    std::vector<imi_q8> x1(n_threads), x2(n_threads);
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < n_ctr; i++) ctr[i].store(0);
        const auto t0 = std::chrono::steady_clock::now();
        team->run([&](int tid) {
            std::vector<float> in(n_ff);
            for (size_t i = 0; i < in.size(); i++) in[i] = 0.01f * (float) ((i * 7 + tid) % 13) - 0.06f;
            for (size_t k = 0; k < ws.size(); k++) {
                const imi_q4 & W = *ws[k];
                x1[tid].from(in.data(), W.cols); if (ncol > 1) x2[tid].from(in.data() + 1, W.cols);
                std::atomic<int> & cc = ctr[k];
                const int ng = W.rows / 8;
                for (int g = cc.fetch_add(4); g < ng; g = cc.fetch_add(4))
                    for (int gg = g; gg < std::min(ng, g + 4); gg++) {
                        if (ncol > 1) imi_q4_group2(W, x1[tid], x2[tid], gg, o1.data() + 8 * gg, o2.data() + 8 * gg);
                        else imi_q4_group(W, x1[tid], gg, o1.data() + 8 * gg);
                    }
                team->sync();
            }
        });
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

imi_seoul::~imi_seoul() {
    for (auto & L : layers) for (imi_q4 * w : { &L.in_proj, &L.out_proj, &L.qkvg, &L.o, &L.gate_up, &L.down }) w->release();
}
