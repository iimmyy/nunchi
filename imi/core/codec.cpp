#include "codec.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
const float * f32(const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; }
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
void transpose(const float * a, int P, int K, int lda, float * t) {
    for (int p = 0; p < P; p++) for (int k = 0; k < K; k++) t[(size_t) k * P + p] = a[(size_t) p * lda + k];
}
using cf = std::complex<float>;
// inverse DFT, mixed radix: halves while the size is even, a direct sum at the odd base (1280 = 2^8 * 5)
void ifft_rec(const cf * in, int n, int stride, cf * out, const cf * tw, int tws) {
    if (n % 2) {
        for (int k = 0; k < n; k++) {
            cf s = 0;
            for (int j = 0; j < n; j++) s += in[(size_t) j * stride] * tw[(size_t) ((j * k) % n) * tws];
            out[k] = s;
        }
        return;
    }
    const int h = n / 2;
    ifft_rec(in, h, 2 * stride, out, tw, 2 * tws);
    ifft_rec(in + stride, h, 2 * stride, out + h, tw, 2 * tws);
    for (int k = 0; k < h; k++) { const cf e = out[k], o = out[k + h] * tw[(size_t) k * tws]; out[k] = e + o; out[k + h] = e - o; }
}
}

bool imi_codec::load(const std::string & path, int threads, imi_team * shared) {
    if (!f.load(path, false)) return false;
    const std::string g = "a.gen.wav.";
    const ggml_tensor * cb = f.keep(g + "quant.first.codebook.weight");
    lvl = f32(cb); n_lvl = (int) cb->ne[0];
    qo_w = f32(f.keep(g + "quant.first.out_proj.weight")); qo_b = f32(f.keep(g + "quant.first.out_proj.bias"));
    in_b = f32(f.keep(g + "tfm.in_proj.bias")); out_b = f32(f.keep(g + "tfm.out_proj.bias"));
    out_norm = f32(f.keep(g + "tfm.output_norm.weight"));
    auto whole = [&](imi_q8g & dst, std::vector<const ggml_tensor *> parts) {
        std::vector<std::pair<const ggml_tensor *, int>> o;
        for (auto * p : parts) for (int r = 0; r < p->ne[1]; r++) o.push_back({ p, r });
        return dst.build(o);
    };
    const ggml_tensor * ip = f.get(g + "tfm.in_proj.weight");
    n_embd = (int) ip->ne[1];
    bool ok = whole(in_proj, { ip }) && whole(out_proj, { f.get(g + "tfm.out_proj.weight") });
    n_bins = (int) f.get(g + "tfm.out_proj.weight")->ne[1] / 2;
    for (int il = 0;; il++) {
        const std::string p = g + "tfm.blk." + std::to_string(il) + ".";
        if (!f.get(p + "ln1.weight", false)) break;
        layer L;
        L.ln1 = f32(f.keep(p + "ln1.weight")); L.ln2 = f32(f.keep(p + "ln2.weight"));
        L.post1 = f32(f.keep(p + "attn_post_norm.weight")); L.post2 = f32(f.keep(p + "ffn_post_norm.weight"));
        if (const ggml_tensor * cw = f.get(p + "conv_dw.weight", false)) {
            L.conv = true;
            const int K = (int) cw->ne[0], C = (int) cw->ne[1];
            d_conv = K - 1;
            conv_w_t.emplace_back((size_t) K * C);
            for (int e = 0; e < C; e++) for (int k = 0; k < K; k++) conv_w_t.back()[(size_t) k * C + e] = f32(cw)[(size_t) e * K + k];
            ok &= whole(L.pw1, { f.get(p + "conv_pw1.weight") }) && whole(L.pw2, { f.get(p + "conv_pw2.weight") });
        } else {
            L.q_norm = f32(f.keep(p + "attn_q_norm.weight")); L.k_norm = f32(f.keep(p + "attn_k_norm.weight"));
            const ggml_tensor * q = f.get(p + "attn_q.weight"), * k = f.get(p + "attn_k.weight"), * gw = f.get(p + "attn_gate.weight");
            head_dim = (int) f.get(p + "attn_q_norm.weight")->ne[0];
            n_head = (int) (q->ne[1] / head_dim); n_head_kv = (int) (k->ne[1] / head_dim);
            ok &= whole(L.qkv, { q, k, f.get(p + "attn_v.weight") }) && whole(L.o, { f.get(p + "attn_out.weight") });
            L.gate_w.resize((size_t) gw->ne[0] * gw->ne[1]);
            for (size_t i = 0; i < L.gate_w.size(); i++)
                L.gate_w[i] = gw->type == GGML_TYPE_F16 ? imi_h2f(((const uint16_t *) gw->data)[i]) : f32(gw)[i];
        }
        const ggml_tensor * fg = f.get(p + "ffn_gate.weight"), * fu = f.get(p + "ffn_up.weight");
        n_ff = (int) fg->ne[1];
        std::vector<std::pair<const ggml_tensor *, int>> o;
        for (int t = 0; t < n_ff; t += 8) { for (int r = t; r < t + 8; r++) o.push_back({ fg, r }); for (int r = t; r < t + 8; r++) o.push_back({ fu, r }); }
        ok &= L.gate_up.build(o) && whole(L.down, { f.get(p + "ffn_down.weight") });
        layers.push_back(std::move(L));
    }
    // conv weights now sit in conv_w_t in layer order
    for (size_t il = 0, ic = 0; il < layers.size(); il++) if (layers[il].conv) layers[il].conv_w = conv_w_t[ic++].data();
    if (!ok) { fprintf(stderr, "imi-codec: a codec table is not 8-bit\n"); return false; }
    f.drop_data();   // every other table is in our layout (or copied, like the conv taps and gates) now

    n_threads = shared ? shared->n : threads;
    per.resize(n_threads);
    const int wide = std::max({ 3 * n_embd, n_ff, 2 * n_bins + 8 });
    for (auto & T : per) { T.x.resize((size_t) n_fs * n_embd); T.h.resize((size_t) n_fs * wide); T.xt.resize((size_t) n_fs * wide); }
    big.resize((size_t) n_fs * 3 * n_embd); y.resize((size_t) n_fs * n_embd); act.resize((size_t) n_fs * n_ff);
    spec.resize((size_t) n_fs * 2 * n_bins);
    conv_state.assign(layers.size(), {}); k_win.assign(layers.size(), {}); v_win.assign(layers.size(), {});
    ctr.reset(new std::atomic<int>[64]);
    hann.resize(n_fft); tw.resize(n_fft);
    for (int i = 0; i < n_fft; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float) M_PI * i / n_fft));   // periodic
        tw[i] = std::polar(1.0f, 2.0f * (float) M_PI * i / n_fft);
    }
    reset();
    if (shared) team = shared; else team->start(n_threads);
    fprintf(stderr, "imi-codec: %zu layers (n_embd %d, ffn %d, %d heads / %d kv x %d, window %d), %d STFT frames per frame, n_fft %d hop %d\n",
            layers.size(), n_embd, n_ff, n_head, n_head_kv, head_dim, window, n_fs, n_fft, hop);
    return true;
}

void imi_codec::reset() {
    pos = 0;
    const int kvd = n_head_kv * head_dim;
    for (size_t il = 0; il < layers.size(); il++) {
        if (layers[il].conv) conv_state[il].assign((size_t) d_conv * n_embd, 0.0f);
        else { k_win[il].assign((size_t) (window - 1) * kvd, 0.0f); v_win[il].assign((size_t) (window - 1) * kvd, 0.0f); }
    }
    ola.assign(n_fft, 0.0f); wsum.assign(n_fft, 0.0f);
    pad_left = (n_fft - hop) / 2;
}

void imi_codec::frame(const int32_t * codes, std::vector<float> & pcm) {
    for (int i = 0; i < 64; i++) ctr[i].store(0, std::memory_order_relaxed);
    team->run([&](int tid) { run(tid, codes); });
    pos += n_fs;
    for (int p = 0; p < n_fs; p++) istft(spec.data() + (size_t) p * 2 * n_bins, pcm);
}

void imi_codec::run(int tid, const int32_t * codes) {
    per_thread & T = per[tid];
    const int E = n_embd, P = n_fs, D = head_dim, kvd = n_head_kv * D, grp = n_head / n_head_kv, W1 = window - 1;
    float * x = T.x.data(), * h = T.h.data(), * xt = T.xt.data();
    int c = 0;
    auto table = [&](const imi_q8g & Wt, int C, const float * in, float * o, int ldo, const float * bias) {
        std::atomic<int> & k = ctr[c++];
        const int ng = Wt.rows_pad / 8;
        float tmp[12 * 8];
        for (int g = k.fetch_add(2, std::memory_order_relaxed); g < ng; g = k.fetch_add(2, std::memory_order_relaxed))
            for (int gg = g; gg < std::min(ng, g + 2); gg++) {
                imi_q8g_group(Wt, in, C, gg, tmp);
                for (int p = 0; p < C; p++) for (int r = 0; r < 8 && 8 * gg + r < Wt.rows; r++)
                    o[(size_t) p * ldo + 8 * gg + r] = tmp[8 * p + r] + (bias ? bias[8 * gg + r] : 0.0f);
            }
    };

    // the frame's vector (every thread makes its own: 48 -> 256 -> 512 is tiny)
    {
        float q[64], v[512];
        for (int i = 0; i < n_cb; i++) for (int l = 0; l < n_lvl; l++) q[i * n_lvl + l] = lvl[(size_t) codes[i] * n_lvl + l];
        const int nq = n_cb * n_lvl, n256 = in_proj.cols;
        for (int o = 0; o < n256; o++) { float s = qo_b[o]; for (int i = 0; i < nq; i++) s += qo_w[(size_t) o * nq + i] * q[i]; v[o] = s; }
        float tmp[8];
        for (int gg = 0; gg < in_proj.rows_pad / 8; gg++) {
            imi_q8g_group(in_proj, v, 1, gg, tmp);
            for (int r = 0; r < 8 && 8 * gg + r < E; r++) x[8 * gg + r] = tmp[r] + in_b[8 * gg + r];
        }
        for (int p = 1; p < P; p++) memcpy(x + (size_t) p * E, x, (size_t) E * sizeof(float));   // nearest upsampling
    }
    for (size_t il = 0; il < layers.size(); il++) {
        const layer & L = layers[il];
        for (int p = 0; p < P; p++) imi_norm_mul(x + (size_t) p * E, L.ln1, h + (size_t) p * E, E, eps);
        transpose(h, P, E, E, xt);
        if (L.conv) {
            table(L.pw1, P, xt, big.data(), 3 * E, nullptr);
            team->sync();
            // causal depthwise conv over [memory | this frame's b*x]; each thread does all channels itself
            float * bx = h;   // reuse: [P][E]
            for (int p = 0; p < P; p++) for (int e = 0; e < E; e++) bx[(size_t) p * E + e] = big[(size_t) p * 3 * E + e] * big[(size_t) p * 3 * E + 2 * E + e];
            const float * st = conv_state[il].data(), * w = L.conv_w;
            float * u = xt + (size_t) P * E;   // [P][E] scratch after the transposed input
            for (int p = 0; p < P; p++) {
                float * up = u + (size_t) p * E;
                for (int e = 0; e < E; e++) up[e] = 0.0f;
                for (int k = 0; k <= d_conv; k++) {
                    const int t = p + k;            // index into [memory (d_conv rows) | bx (P rows)]
                    const float * src = t < d_conv ? st + (size_t) t * E : bx + (size_t) (t - d_conv) * E;
                    const float * wk = w + (size_t) k * E;
                    for (int e = 0; e < E; e++) up[e] += src[e] * wk[e];
                }
                const float * cc = big.data() + (size_t) p * 3 * E + E;
                for (int e = 0; e < E; e++) up[e] *= cc[e];
            }
            transpose(u, P, E, E, xt);
            table(L.pw2, P, xt, y.data(), E, nullptr);
            team->sync();
            if (tid == 0) {   // memory = the last d_conv rows of [memory | bx], from thread 0's own bx
                std::vector<float> & s = conv_state[il];
                std::vector<float> all((size_t) (d_conv + P) * E);
                memcpy(all.data(), s.data(), (size_t) d_conv * E * sizeof(float));
                memcpy(all.data() + (size_t) d_conv * E, bx, (size_t) P * E * sizeof(float));
                memcpy(s.data(), all.data() + (size_t) P * E, (size_t) d_conv * E * sizeof(float));
            }
        } else {
            const int nq = n_head * D, nr = nq + 2 * kvd;
            table(L.qkv, P, xt, big.data(), nr, nullptr);
            team->sync();
            // keys/values over [window memory | this frame], normed and rotated (each thread makes its own copy)
            std::vector<float> K((size_t) (W1 + P) * kvd), V((size_t) (W1 + P) * kvd);
            memcpy(K.data(), k_win[il].data(), (size_t) W1 * kvd * sizeof(float));
            memcpy(V.data(), v_win[il].data(), (size_t) W1 * kvd * sizeof(float));
            for (int p = 0; p < P; p++) {
                const float * r = big.data() + (size_t) p * nr;
                for (int g = 0; g < n_head_kv; g++) {
                    float * kk = K.data() + (size_t) (W1 + p) * kvd + g * D;
                    imi_norm_mul(r + nq + g * D, L.k_norm, kk, D, eps);
                    rope(kk, D, pos + p, theta);
                }
                memcpy(V.data() + (size_t) (W1 + p) * kvd, r + nq + kvd, (size_t) kvd * sizeof(float));
            }
            for (int hh = tid; hh < n_head; hh += n_threads) {
                const int kv = hh / grp;
                for (int p = 0; p < P; p++) {
                    float q[128], sc[64];
                    imi_norm_mul(big.data() + (size_t) p * nr + hh * D, L.q_norm, q, D, eps);
                    rope(q, D, pos + p, theta);
                    // query p sees keys p .. p + W1 of the concatenation, if their position is >= 0
                    float mx = -INFINITY;
                    for (int j = p; j <= p + W1; j++) {
                        if (pos - W1 + j < 0) { sc[j - p] = -INFINITY; continue; }
                        float s = 0; const float * kk = K.data() + (size_t) j * kvd + kv * D;
                        for (int i = 0; i < D; i++) s += q[i] * kk[i];
                        sc[j - p] = s / sqrtf((float) D); mx = std::max(mx, sc[j - p]);
                    }
                    float sum = 0;
                    for (int j = 0; j <= W1; j++) { sc[j] = sc[j] == -INFINITY ? 0.0f : expf(sc[j] - mx); sum += sc[j]; }
                    float gs = 0; const float * gw = L.gate_w.data() + (size_t) hh * E, * hp = h + (size_t) p * E;
                    for (int e = 0; e < E; e++) gs += gw[e] * hp[e];
                    const float gate = 1.0f / (1.0f + expf(-gs));
                    float * o = act.data() + (size_t) p * nq + hh * D;
                    for (int i = 0; i < D; i++) {
                        float a = 0;
                        for (int j = 0; j <= W1; j++) a += sc[j] * V[(size_t) (p + j) * kvd + kv * D + i];
                        o[i] = a / sum * gate;
                    }
                }
            }
            team->sync();
            if (tid == 0) {   // window memory = the last W1 positions
                memcpy(k_win[il].data(), K.data() + (size_t) P * kvd, (size_t) W1 * kvd * sizeof(float));
                memcpy(v_win[il].data(), V.data() + (size_t) P * kvd, (size_t) W1 * kvd * sizeof(float));
            }
            transpose(act.data(), P, nq, nq, xt);
            table(L.o, P, xt, y.data(), E, nullptr);
            team->sync();
        }
        for (int p = 0; p < P; p++) imi_norm_mul_add(y.data() + (size_t) p * E, L.post1, x + (size_t) p * E, E, eps);
        for (int p = 0; p < P; p++) imi_norm_mul(x + (size_t) p * E, L.ln2, h + (size_t) p * E, E, eps);
        transpose(h, P, E, E, xt);
        {   // (gate, up) group pairs -> silu(gate) * up
            std::atomic<int> & k = ctr[c++];
            const int ng = L.gate_up.rows_pad / 8;
            float a[12 * 8], b[12 * 8];
            for (int g = k.fetch_add(2, std::memory_order_relaxed); g < ng; g = k.fetch_add(2, std::memory_order_relaxed)) {
                imi_q8g_group(L.gate_up, xt, P, g, a); imi_q8g_group(L.gate_up, xt, P, g + 1, b);
                for (int p = 0; p < P; p++) for (int r = 0; r < 8; r++) act[(size_t) p * n_ff + 4 * g + r] = silu(a[8 * p + r]) * b[8 * p + r];
            }
        }
        team->sync();
        transpose(act.data(), P, n_ff, n_ff, xt);
        table(L.down, P, xt, y.data(), E, nullptr);
        team->sync();
        for (int p = 0; p < P; p++) imi_norm_mul_add(y.data() + (size_t) p * E, L.post2, x + (size_t) p * E, E, eps);
    }
    for (int p = 0; p < P; p++) imi_norm_mul(x + (size_t) p * E, out_norm, h + (size_t) p * E, E, eps);
    transpose(h, P, E, E, xt);
    table(out_proj, P, xt, spec.data(), 2 * n_bins, out_b);
}

// one STFT frame: log magnitude + phase -> inverse FFT -> windowed overlap-add -> hop samples (less the start padding)
void imi_codec::istft(const float * lp, std::vector<float> & pcm) {
    std::vector<cf> X(n_fft), xo(n_fft);
    for (int k = 0; k < n_bins; k++) X[k] = std::polar(expf(lp[k]), lp[n_bins + k]);
    for (int k = 1; k < n_bins - 1; k++) X[n_fft - k] = std::conj(X[k]);
    ifft_rec(X.data(), n_fft, 1, xo.data(), tw.data(), 1);
    for (int j = 0; j < n_fft; j++) { ola[j] += xo[j].real() / n_fft * hann[j]; wsum[j] += hann[j] * hann[j]; }
    for (int i = 0; i < hop; i++) {
        const float s = wsum[i] > 1e-8f ? ola[i] / wsum[i] : ola[i];
        if (pad_left > 0) pad_left--; else pcm.push_back(s);
    }
    std::copy(ola.begin() + hop, ola.end(), ola.begin()); std::fill(ola.end() - hop, ola.end(), 0.0f);
    std::copy(wsum.begin() + hop, wsum.end(), wsum.begin()); std::fill(wsum.end() - hop, wsum.end(), 0.0f);
}

void imi_codec::flush(std::vector<float> & pcm) {
    // what the overlap still holds, less the end padding (the reference's centred inverse STFT)
    const int tail = n_fft - hop, trim = (n_fft - hop) / 2;
    for (int i = 0; i < tail - trim; i++) pcm.push_back(wsum[i] > 1e-8f ? ola[i] / wsum[i] : ola[i]);
    reset();
}

imi_codec::~imi_codec() {
    in_proj.release(); out_proj.release();
    for (auto & L : layers) for (imi_q8g * w : { &L.gate_up, &L.down, &L.pw1, &L.pw2, &L.qkv, &L.o }) w->release();
}
