#include "ears.h"
#include "vec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
std::vector<float> copy_f32(const ggml_tensor * t) {
    const float * d = (const float *) t->data;
    return std::vector<float>(d, d + ggml_nelements(t));
}
// [ch][3][3] (kw fastest, kw along time) -> [kh*3+kw][ch]
std::vector<float> taps9(const ggml_tensor * t) {
    const int C = (int) t->ne[3];
    const float * d = (const float *) t->data;
    std::vector<float> o((size_t) 9 * C);
    for (int c = 0; c < C; c++) for (int j = 0; j < 9; j++) o[(size_t) j * C + c] = d[(size_t) c * 9 + j];
    return o;
}
void rope(float * v, int n, int pos, float base) {
    const float scale = powf(base, -2.0f / (float) n);
    float th = (float) pos;
    for (int i = 0; i < n / 2; i++) {
        const float c = cosf(th), s = sinf(th), x0 = v[i], x1 = v[i + n / 2];
        v[i] = x0 * c - x1 * s; v[i + n / 2] = x0 * s + x1 * c;
        th *= scale;
    }
}
inline float silu(float x) { return x / (1.0f + expf(-x)); }
inline float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
enum { ACT_NONE = 0, ACT_SILU = 1, ACT_GELU = 2 };
double secs(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}
}

bool imi_ears::load(const std::string & path, int threads, imi_team * shared) {
    imi_file f;
    if (!f.load(path, false)) return false;
    if (!f.get("a.pre_encode.out.weight", false)) { fprintf(stderr, "imi-ears: %s has no ears\n", path.c_str()); return false; }
    bool ok = true;
    // mel front end: the Hann window, an exact DFT as a table (cos and -sin rows, bin by bin), Slaney filters
    hann.resize(n_fft);
    for (int i = 0; i < n_fft; i++) hann[i] = (float) (0.5 * (1.0 - cos(2.0 * M_PI * i / n_fft)));
    n_bin = n_fft / 2 + 1;
    {
        std::vector<float> m((size_t) 2 * n_bin * n_fft);
        for (int b = 0; b < n_bin; b++) for (int k = 0; k < n_fft; k++) {
            const double a = 2.0 * M_PI * (double) ((size_t) b * k % n_fft) / n_fft;
            m[(size_t) (2 * b) * n_fft + k] = (float) cos(a); m[(size_t) (2 * b + 1) * n_fft + k] = (float) -sin(a);
        }
        ok &= dft.build(m.data(), 2 * n_bin, n_fft);
    }
    n_mel = (int) f.key_i("clip.audio.num_mel_bins", 128);
    {
        const double sr = 16000, lin = 3.0 / 200, min_log_mel = 15.0, step = log(6.4) / 27;
        auto hz2mel = [&](double hz) { return hz < 1000 ? hz * lin : min_log_mel + log(hz / 1000) / step; };
        auto mel2hz = [&](double m) { return m < min_log_mel ? m / lin : 1000 * exp((m - min_log_mel) * step); };
        const double lo = hz2mel(0), hi = hz2mel(sr / 2);
        std::vector<double> pts(n_mel + 2);
        for (int i = 0; i < n_mel + 2; i++) pts[i] = mel2hz(lo + (hi - lo) * i / (n_mel + 1));
        fb.assign((size_t) n_mel * n_bin, 0.0f);
        for (int m = 0; m < n_mel; m++) {
            const double fl = pts[m], fc = pts[m + 1], fr = pts[m + 2], en = 2.0 / (fr - fl);
            for (int b = 0; b < n_bin; b++) {
                const double hz = b * sr / n_fft;
                double w = 0;
                if (hz >= fl && hz <= fc) w = (hz - fl) / (fc - fl); else if (hz > fc && hz <= fr) w = (fr - hz) / (fr - fc);
                fb[(size_t) m * n_bin + b] = (float) (w * en);
            }
        }
    }
    // downsampler
    const ggml_tensor * c0 = f.get("a.conv1d.0.weight");
    n_ch = (int) c0->ne[3];
    c0_w = taps9(c0);                         c0_b = copy_f32(f.get("a.conv1d.0.bias"));
    c2_w = taps9(f.get("a.conv1d.2.weight")); c2_b = copy_f32(f.get("a.conv1d.2.bias"));
    c5_w = taps9(f.get("a.conv1d.5.weight")); c5_b = copy_f32(f.get("a.conv1d.5.bias"));
    ok &= c3.build((const float *) f.get("a.conv1d.3.weight")->data, n_ch, n_ch); c3_b = copy_f32(f.get("a.conv1d.3.bias"));
    ok &= c6.build((const float *) f.get("a.conv1d.6.weight")->data, n_ch, n_ch); c6_b = copy_f32(f.get("a.conv1d.6.bias"));
    ok &= pre.build(f.get("a.pre_encode.out.weight")); pre_b = copy_f32(f.get("a.pre_encode.out.bias"));
    n_embd = pre.rows;
    if (pre.cols != n_ch * n_mel / 8) { fprintf(stderr, "imi-ears: unexpected downsampler width %d\n", pre.cols); return false; }
    // layers
    window = (int) f.key_i("clip.audio.window_size", 32);
    n_head = (int) f.key_i("clip.audio.attention.head_count", 8);
    eps = f.key_f("clip.audio.attention.layer_norm_epsilon", 1e-5f);
    size_t bytes = dft.bytes + c3.bytes + c6.bytes + pre.bytes;
    auto rows = [](std::vector<std::pair<const ggml_tensor *, int>> & o, const ggml_tensor * t, int r0, int n) { for (int r = r0; r < r0 + n; r++) o.push_back({ t, r }); };
    for (int il = 0;; il++) {
        const std::string p = "a.blk." + std::to_string(il) + ".";
        if (!f.get(p + "ln1.weight", false)) break;
        layer L;
        L.ln1 = copy_f32(f.get(p + "ln1.weight")); L.ln2 = copy_f32(f.get(p + "ln2.weight"));
        std::vector<std::pair<const ggml_tensor *, int>> in;
        if (const ggml_tensor * dw = f.get(p + "conv_dw.weight", false)) {
            d_conv = (int) dw->ne[0];
            L.conv_w.resize((size_t) d_conv * n_embd);
            const float * d = (const float *) dw->data;
            for (int e = 0; e < n_embd; e++) for (int k = 0; k < d_conv; k++) L.conv_w[(size_t) k * n_embd + e] = d[(size_t) e * d_conv + k];
            const ggml_tensor * pw1 = f.get(p + "conv_pw1.weight");
            rows(in, pw1, 0, (int) pw1->ne[1]);
            ok &= L.out.build(f.get(p + "conv_pw2.weight"));
        } else {
            L.attn = true;
            L.q_norm = copy_f32(f.get(p + "attn_q_norm.weight")); L.k_norm = copy_f32(f.get(p + "attn_k_norm.weight"));
            head_dim = (int) L.q_norm.size();
            const ggml_tensor * q = f.get(p + "attn_q.weight"), * k = f.get(p + "attn_k.weight"), * v = f.get(p + "attn_v.weight");
            n_head_kv = (int) (k->ne[1] / head_dim);
            rows(in, q, 0, (int) q->ne[1]); rows(in, k, 0, (int) k->ne[1]); rows(in, v, 0, (int) v->ne[1]);
            ok &= L.out.build(f.get(p + "attn_out.weight"));
        }
        ok &= L.in.build(in);
        const ggml_tensor * fg = f.get(p + "ffn_gate.weight"), * fu = f.get(p + "ffn_up.weight");
        n_ff = (int) fg->ne[1];
        std::vector<std::pair<const ggml_tensor *, int>> gu;
        for (int t = 0; t < n_ff; t += 8) { rows(gu, fg, t, 8); rows(gu, fu, t, 8); }
        ok &= L.gate_up.build(gu);
        ok &= L.down.build(f.get(p + "ffn_down.weight"));
        for (imi_wg * w : { &L.in, &L.out, &L.gate_up, &L.down }) bytes += w->bytes;
        layers.push_back(std::move(L));
    }
    post_ln = copy_f32(f.get("a.post_ln.weight"));
    ln_w = copy_f32(f.get("mm.a.mlp.0.weight")); ln_b = copy_f32(f.get("mm.a.mlp.0.bias"));
    ok &= mlp1.build(f.get("mm.a.mlp.1.weight")); mlp1_b = copy_f32(f.get("mm.a.mlp.1.bias"));
    ok &= mlp3.build(f.get("mm.a.mlp.3.weight")); mlp3_b = copy_f32(f.get("mm.a.mlp.3.bias"));
    n_out = mlp3.rows;
    bytes += mlp1.bytes + mlp3.bytes;
    if (!ok || n_ff % 8 || n_embd % 32 || head_dim % 32) { fprintf(stderr, "imi-ears: a table has an unexpected type or shape\n"); return false; }
    n_threads = shared ? shared->n : threads;
    if (shared) team = shared; else team->start(n_threads);
    fprintf(stderr, "imi-ears: %zu layers, n_embd %d, ffn %d, %d heads (%d kv) x %d, window %d, %d mels, out %d, %zu MB, %d threads\n",
            layers.size(), n_embd, n_ff, n_head, n_head_kv, head_dim, window, n_mel, n_out, bytes >> 20, n_threads);
    return true;
}

void imi_ears::par(int n, const std::function<void(int)> & fn) {
    if (n <= 0) return;
    ctr.store(0, std::memory_order_relaxed);
    team->run([&](int) { for (int u = ctr.fetch_add(1, std::memory_order_relaxed); u < n; u = ctr.fetch_add(1, std::memory_order_relaxed)) fn(u); });
}

void imi_ears::pack(const float * a, int P, int K, int lda, float * o) {
    par((P + 11) / 12, [&](int ch) {
        float * d = o + (size_t) ch * K * 12;
        for (int j = 0; j < 12; j++) {
            const int p = ch * 12 + j;
            if (p < P) { const float * r = a + (size_t) p * lda; for (int k = 0; k < K; k++) d[(size_t) k * 12 + j] = r[k]; }
            else for (int k = 0; k < K; k++) d[(size_t) k * 12 + j] = 0.0f;
        }
    });
}

// units = (block of row groups) x (block of up to 16 position chunks): the block's weights stay in L2 across its chunks
void imi_ears::table(const imi_wg & W, const float * x, int P, float * out, int ldo, const float * bias, bool add) {
    const int ng = W.rows_pad / 8, gb = W.cols > 2048 ? 2 : 4, ngb = (ng + gb - 1) / gb;
    const int nch = (P + 11) / 12, cb = 16, ncb = (nch + cb - 1) / cb;
    par(ngb * ncb, [&](int u) {
        const int g0 = (u % ngb) * gb, c0 = (u / ngb) * cb;
        float tmp[96];
        for (int ch = c0; ch < std::min(nch, c0 + cb); ch++) {
            const float * xc = x + (size_t) ch * W.cols * 12;
            for (int g = g0; g < std::min(ng, g0 + gb); g++) {
                imi_wg_group12(W, xc, g, tmp);
                const int nr = std::min(8, W.rows - 8 * g);
                for (int j = 0; j < 12 && ch * 12 + j < P; j++) {
                    float * o = out + (size_t) (ch * 12 + j) * ldo + 8 * g;
                    for (int r = 0; r < nr; r++) {
                        const float v = tmp[8 * j + r] + (bias ? bias[8 * g + r] : 0.0f);
                        o[r] = add ? o[r] + v : v;
                    }
                }
            }
        }
    });
}

void imi_ears::table_glu(const imi_wg & W, const float * x, int P, float * out) {
    const int npair = W.rows_pad / 16, pb = 2, npb = (npair + pb - 1) / pb, half = W.rows / 2;
    const int nch = (P + 11) / 12, cb = 16, ncb = (nch + cb - 1) / cb;
    par(npb * ncb, [&](int u) {
        const int q0 = (u % npb) * pb, c0 = (u / npb) * cb;
        float a[96], b[96];
        for (int ch = c0; ch < std::min(nch, c0 + cb); ch++) {
            const float * xc = x + (size_t) ch * W.cols * 12;
            for (int q = q0; q < std::min(npair, q0 + pb); q++) {
                imi_wg_group12(W, xc, 2 * q, a); imi_wg_group12(W, xc, 2 * q + 1, b);
                for (int j = 0; j < 12 && ch * 12 + j < P; j++)
                    for (int r = 0; r < 8; r++) out[(size_t) (ch * 12 + j) * half + 8 * q + r] = silu(a[8 * j + r]) * b[8 * j + r];
            }
        }
    });
}

int imi_ears::encode(const float * pcm, int n, std::vector<float> & out) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    const int n_pad = n <= n_fft ? n_fft : n + (hop - (n - n_fft) % hop) % hop;
    const int n_fr = (n_pad - n_fft) / hop + 1;
    const int T2 = (n_fr + 1) / 2, F2 = n_mel / 2, T4 = (T2 + 1) / 2, F4 = F2 / 2, P = (T4 + 1) / 2, F8 = F4 / 2, C = n_ch, E = n_embd;
    auto ch12 = [](int v) { return (size_t) (v + 11) / 12 * 12; };
    xt.resize(std::max({ ch12(T4 * F4) * C, ch12(P * F8) * C, ch12(P) * pre.cols, ch12(P) * n_ff, ch12(P) * n_out }));
    buf.resize(std::max({ (size_t) T2 * F2 * C, (size_t) T4 * F4 * C, (size_t) P * layers[0].in.rows, (size_t) P * 3 * E, (size_t) P * n_ff, (size_t) P * n_out }));
    buf2.resize((size_t) T4 * F4 * C);
    xres.resize((size_t) P * E);

    // ---- mel: frames as a 12-at-a-time table read of the DFT; power; Slaney filters (double sums); log10
    last_mel.assign((size_t) n_fr * n_mel, 0.0f);
    float * mel = last_mel.data();
    par((n_fr + 11) / 12, [&](int ch) {
        std::vector<float> fx((size_t) n_fft * 12), pw((size_t) 12 * n_bin);
        for (int j = 0; j < 12; j++) {
            const int fr = ch * 12 + j;
            for (int k = 0; k < n_fft; k++) {
                const int i = fr * hop + k;
                fx[(size_t) k * 12 + j] = fr < n_fr && i < n ? hann[k] * pcm[i] : 0.0f;
            }
        }
        float tmp[96];
        for (int g = 0; g < dft.rows_pad / 8; g++) {
            imi_wg_group12(dft, fx.data(), g, tmp);
            for (int j = 0; j < 12; j++) for (int r = 0; r < 8; r += 2) {
                const int b = 4 * g + r / 2;
                if (b < n_bin) pw[(size_t) j * n_bin + b] = tmp[8 * j + r] * tmp[8 * j + r] + tmp[8 * j + r + 1] * tmp[8 * j + r + 1];
            }
        }
        for (int j = 0; j < 12; j++) {
            const int fr = ch * 12 + j;
            if (fr >= n_fr) break;
            for (int m = 0; m < n_mel; m++) {
                double s = 0;
                const float * w = fb.data() + (size_t) m * n_bin;
                for (int b = 0; b < n_bin; b++) s += (double) w[b] * pw[(size_t) j * n_bin + b];
                mel[(size_t) fr * n_mel + m] = (float) ((log10(std::max(s, 1e-12)) + 4.0) / 4.0);
            }
        }
    });
    t_mel += secs(t0); t0 = clk::now();

    // ---- downsampler. Layout [time][freq][ch]; 3x3 stride 2, zero outside (the stream's first left column is zero)
    float * a0 = buf.data();
    par(T2, [&](int t) {
        for (int fq = 0; fq < F2; fq++) {
            float * o = a0 + ((size_t) t * F2 + fq) * C;
            for (int c = 0; c < C; c++) o[c] = c0_b[c];
            for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                const int ti = 2 * t + kw - 1, fi = 2 * fq + kh - 1;
                if (ti < 0 || ti >= n_fr || fi < 0 || fi >= n_mel) continue;
                const float v = mel[(size_t) ti * n_mel + fi], * w = c0_w.data() + (size_t) (kh * 3 + kw) * C;
                for (int c = 0; c < C; c++) o[c] += v * w[c];
            }
            for (int c = 0; c < C; c++) o[c] = silu(o[c]);
        }
    });
    auto dw = [&](const float * in, int Ti, int Fi, const std::vector<float> & w, const std::vector<float> & b, float * o_) {
        const int To = (Ti + 1) / 2, Fo = Fi / 2;
        par(To, [&](int t) {
            for (int fq = 0; fq < Fo; fq++) {
                float * o = o_ + ((size_t) t * Fo + fq) * C;
                for (int c = 0; c < C; c++) o[c] = b[c];
                for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                    const int ti = 2 * t + kw - 1, fi = 2 * fq + kh - 1;
                    if (ti < 0 || ti >= Ti || fi < 0 || fi >= Fi) continue;
                    const float * v = in + ((size_t) ti * Fi + fi) * C, * ww = w.data() + (size_t) (kh * 3 + kw) * C;
                    for (int c = 0; c < C; c++) o[c] += v[c] * ww[c];
                }
            }
        });
    };
    auto silu_all = [&](float * a, size_t cnt) {
        const size_t blk = 1 << 14;
        par((int) ((cnt + blk - 1) / blk), [&](int u) { for (size_t i = (size_t) u * blk; i < std::min(cnt, (size_t) (u + 1) * blk); i++) a[i] = silu(a[i]); });
    };
    dw(a0, T2, F2, c2_w, c2_b, buf2.data());                        // [T4][F4][C]
    pack(buf2.data(), T4 * F4, C, C, xt.data());
    table(c3, xt.data(), T4 * F4, buf.data(), C, c3_b.data(), false);
    silu_all(buf.data(), (size_t) T4 * F4 * C);
    dw(buf.data(), T4, F4, c5_w, c5_b, buf2.data());                // [P][F8][C]
    pack(buf2.data(), P * F8, C, C, xt.data());
    table(c6, xt.data(), P * F8, buf.data(), C, c6_b.data(), false);
    silu_all(buf.data(), (size_t) P * F8 * C);
    // [P][F8][C] -> feature c * F8 + f, straight into 12-position chunks
    {
        const float * a3 = buf.data();
        const int K = C * F8;
        par((P + 11) / 12, [&](int ch) {
            float * d = xt.data() + (size_t) ch * K * 12;
            for (int j = 0; j < 12; j++) {
                const int p = ch * 12 + j;
                for (int c = 0; c < C; c++) for (int fq = 0; fq < F8; fq++)
                    d[(size_t) (c * F8 + fq) * 12 + j] = p < P ? a3[((size_t) p * F8 + fq) * C + c] : 0.0f;
            }
        });
    }
    float * x = xres.data();
    table(pre, xt.data(), P, x, E, pre_b.data(), false);
    t_down += secs(t0); t0 = clk::now();

    // ---- layers
    auto norm_pack = [&](const std::vector<float> & w) {
        par((P + 11) / 12, [&](int ch) {
            float * d = xt.data() + (size_t) ch * E * 12;
            std::vector<float> h(E);
            for (int j = 0; j < 12; j++) {
                const int p = ch * 12 + j;
                if (p < P) imi_norm_mul(x + (size_t) p * E, w.data(), h.data(), E, eps); else std::fill(h.begin(), h.end(), 0.0f);
                for (int k = 0; k < E; k++) d[(size_t) k * 12 + j] = h[k];
            }
        });
    };
    const int D = head_dim;
    for (const layer & L : layers) {
        norm_pack(L.ln1);
        const int ld = L.in.rows;
        table(L.in, xt.data(), P, buf.data(), ld, nullptr, false);
        if (!L.attn) {
            // b | c | x: b*x through the causal depthwise conv over the last d_conv steps, times c
            par((P + 11) / 12, [&](int ch) {
                float * d = xt.data() + (size_t) ch * E * 12;
                for (int j = 0; j < 12; j++) {
                    const int p = ch * 12 + j;
                    if (p >= P) { for (int e = 0; e < E; e++) d[(size_t) e * 12 + j] = 0.0f; continue; }
                    for (int e = 0; e < E; e++) {
                        float co = 0;
                        for (int k = 0; k < d_conv; k++) {
                            const int t = p - (d_conv - 1) + k;
                            if (t < 0) continue;
                            const float * r = buf.data() + (size_t) t * ld;
                            co += L.conv_w[(size_t) k * E + e] * (r[e] * r[2 * E + e]);
                        }
                        d[(size_t) e * 12 + j] = buf[(size_t) p * ld + E + e] * co;
                    }
                }
            });
        } else {
            const int Hq = n_head * D, Hk = n_head_kv * D, grp = n_head / n_head_kv;
            par(P, [&](int p) {
                float * r = buf.data() + (size_t) p * ld;
                for (int h = 0; h < n_head; h++) { imi_norm_mul(r + h * D, L.q_norm.data(), r + h * D, D, eps); rope(r + h * D, D, p, rope_theta); }
                for (int h = 0; h < n_head_kv; h++) { float * k = r + Hq + h * D; imi_norm_mul(k, L.k_norm.data(), k, D, eps); rope(k, D, p, rope_theta); }
            });
            const float scale = 1.0f / sqrtf((float) D);
            par((P + 11) / 12, [&](int ch) {
                float * d = xt.data() + (size_t) ch * Hq * 12;
                std::vector<float> sc(window), o(D);
                for (int j = 0; j < 12; j++) {
                    const int p = ch * 12 + j;
                    for (int hh = 0; hh < n_head; hh++) {
                        if (p >= P) { for (int i = 0; i < D; i++) d[(size_t) (hh * D + i) * 12 + j] = 0.0f; continue; }
                        const int kv = hh / grp, s0 = std::max(0, p - window + 1), nk = p - s0 + 1;
                        const float * q = buf.data() + (size_t) p * ld + hh * D;
                        float mx = -INFINITY;
                        for (int t = 0; t < nk; t++) {
                            const float * k = buf.data() + (size_t) (s0 + t) * ld + Hq + kv * D;
                            float s = 0;
                            for (int i = 0; i < D; i++) s += q[i] * k[i];
                            sc[t] = s * scale; mx = std::max(mx, sc[t]);
                        }
                        float sum = 0;
                        for (int t = 0; t < nk; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                        std::fill(o.begin(), o.end(), 0.0f);
                        for (int t = 0; t < nk; t++) {
                            const float * v = buf.data() + (size_t) (s0 + t) * ld + Hq + Hk + kv * D;
                            for (int i = 0; i < D; i++) o[i] += sc[t] * v[i];
                        }
                        for (int i = 0; i < D; i++) d[(size_t) (hh * D + i) * 12 + j] = o[i] / sum;
                    }
                }
            });
        }
        table(L.out, xt.data(), P, x, E, nullptr, true);
        norm_pack(L.ln2);
        table_glu(L.gate_up, xt.data(), P, buf.data());
        pack(buf.data(), P, n_ff, n_ff, xt.data());
        table(L.down, xt.data(), P, x, E, nullptr, true);
    }
    t_layers += secs(t0); t0 = clk::now();

    // ---- last norm, LayerNorm, 768 -> 2048, GELU (erf), 2048 -> 2048
    par((P + 11) / 12, [&](int ch) {
        float * d = xt.data() + (size_t) ch * E * 12;
        std::vector<float> h(E);
        for (int j = 0; j < 12; j++) {
            const int p = ch * 12 + j;
            if (p < P) {
                imi_norm_mul(x + (size_t) p * E, post_ln.data(), h.data(), E, eps);
                double mu = 0, var = 0;
                for (int k = 0; k < E; k++) mu += h[k];
                mu /= E;
                for (int k = 0; k < E; k++) var += (h[k] - mu) * (h[k] - mu);
                const float inv = (float) (1.0 / sqrt(var / E + 1e-5));
                for (int k = 0; k < E; k++) h[k] = (float) (h[k] - mu) * inv * ln_w[k] + ln_b[k];
            } else std::fill(h.begin(), h.end(), 0.0f);
            for (int k = 0; k < E; k++) d[(size_t) k * 12 + j] = h[k];
        }
    });
    table(mlp1, xt.data(), P, buf.data(), mlp1.rows, mlp1_b.data(), false);
    {
        const size_t cnt = (size_t) P * mlp1.rows, blk = 1 << 14;
        float * a = buf.data();
        par((int) ((cnt + blk - 1) / blk), [&](int u) { for (size_t i = (size_t) u * blk; i < std::min(cnt, (size_t) (u + 1) * blk); i++) a[i] = gelu_erf(a[i]); });
    }
    pack(buf.data(), P, mlp1.rows, mlp1.rows, xt.data());
    out.resize((size_t) P * n_out);
    table(mlp3, xt.data(), P, out.data(), n_out, mlp3_b.data(), false);
    t_proj += secs(t0);
    return P;
}

imi_ears::~imi_ears() {
    for (imi_wg * w : { &dft, &c3, &c6, &pre, &mlp1, &mlp3 }) w->release();
    for (auto & L : layers) for (imi_wg * w : { &L.in, &L.out, &L.gate_up, &L.down }) w->release();
}
