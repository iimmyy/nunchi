#include "voice.h"
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
// [P][K] -> [K][P]
void transpose(const float * a, int P, int K, int lda, float * t) {
    for (int p = 0; p < P; p++) for (int k = 0; k < K; k++) t[(size_t) k * P + p] = a[(size_t) p * lda + k];
}
}

bool imi_voice::load(const std::string & path, int threads, imi_team * shared) {
    if (!f.load(path, false)) return false;
    const std::string g = "a.gen.code.";
    embd      = f.keep(g + "embd.weight");       // read while running: masked-pass inputs
    out_embd  = f.keep(g + "out_embd.weight");   // read while running: the brain's next input
    mask_embd = f32(f.keep(g + "mask_embd"));
    out_norm  = f32(f.keep(g + "output_norm.weight"));
    head_norm = f32(f.keep(g + "head_norm.weight"));
    proj_b    = f32(f.keep(g + "proj_in.bias"));
    const ggml_tensor * pw = f.get(g + "proj_in.weight");
    n_embd = (int) embd->ne[0]; n_vocab = (int) embd->ne[1]; n_cb = (int) embd->ne[2]; n_in = (int) pw->ne[0];
    if (embd->type != GGML_TYPE_Q8_0) { fprintf(stderr, "imi-voice: needs the 8-bit generator file (genq8b)\n"); return false; }
    auto rows = [](std::vector<std::pair<const ggml_tensor *, int>> & o, const ggml_tensor * t, int r0, int n) { for (int r = r0; r < r0 + n; r++) o.push_back({ t, r }); };
    auto whole = [&](imi_q8g & dst, std::vector<const ggml_tensor *> parts) {
        std::vector<std::pair<const ggml_tensor *, int>> o;
        for (auto * p : parts) rows(o, p, 0, (int) p->ne[1]);
        return dst.build(o);
    };
    bool ok = whole(proj_in, { pw });
    size_t bytes = proj_in.bytes;
    for (int il = 0;; il++) {
        const std::string p = g + "blk." + std::to_string(il) + ".";
        if (!f.get(p + "ln1.weight", false)) break;
        layer L;
        L.ln1 = f32(f.keep(p + "ln1.weight")); L.ln2 = f32(f.keep(p + "ln2.weight"));
        L.q_norm = f32(f.keep(p + "attn_q_norm.weight")); L.k_norm = f32(f.keep(p + "attn_k_norm.weight"));
        const ggml_tensor * q = f.get(p + "attn_q.weight"), * k = f.get(p + "attn_k.weight");
        head_dim = (int) f.get(p + "attn_q_norm.weight")->ne[0];
        n_head = (int) (q->ne[1] / head_dim); n_head_kv = (int) (k->ne[1] / head_dim);
        ok &= whole(L.qkvg, { q, k, f.get(p + "attn_v.weight"), f.get(p + "attn_gate.weight") });
        ok &= whole(L.o, { f.get(p + "attn_out.weight") });
        const ggml_tensor * fg = f.get(p + "ffn_gate.weight"), * fu = f.get(p + "ffn_up.weight");
        n_ff = (int) fg->ne[1];
        std::vector<std::pair<const ggml_tensor *, int>> o;
        for (int t = 0; t < n_ff; t += 8) { rows(o, fg, t, 8); rows(o, fu, t, 8); }
        ok &= L.gate_up.build(o);
        ok &= whole(L.down, { f.get(p + "ffn_down.weight") });
        for (imi_q8g * w : { &L.qkvg, &L.o, &L.gate_up, &L.down }) bytes += w->bytes;
        layers.push_back(L);
    }
    score.resize(n_cb);
    for (int c = 0; c < n_cb; c++) {
        std::vector<std::pair<const ggml_tensor *, int>> o;
        rows(o, embd, 0, n_vocab);
        ok &= score[c].build(o, c);
        bytes += score[c].bytes;
    }
    if (!ok) { fprintf(stderr, "imi-voice: a generator table is not 8-bit\n"); return false; }
    f.drop_data();   // every other table is in our layout now
    n_qkv = (n_head + 2 * n_head_kv) * head_dim + n_head;
    n_threads = shared ? shared->n : threads;
    per.resize(n_threads);
    const int wide = std::max({ n_embd, n_ff, n_in });
    for (auto & T : per) { T.x.resize((size_t) n_cb * n_embd); T.h.resize((size_t) n_cb * wide); T.xt.resize((size_t) n_cb * wide); }
    qkv.resize((size_t) n_cb * n_qkv); att.resize((size_t) n_cb * n_head * head_dim); y.resize((size_t) n_cb * n_embd);
    act.resize((size_t) n_cb * n_ff); z.resize((size_t) n_cb * n_embd); base.resize((size_t) n_cb * n_embd);
    ctr.reset(new std::atomic<int>[64]);
    if (shared) team = shared; else team->start(n_threads);
    fprintf(stderr, "imi-voice: %zu layers, n_embd %d, %d codebooks x %d codes, %d heads (%d kv) x %d, our layout %zu MB, %d threads\n",
            layers.size(), n_embd, n_cb, n_vocab, n_head, n_head_kv, head_dim, bytes >> 20, n_threads);
    return true;
}

void imi_voice::feedback(const int32_t * codes, float * out) const {
    std::vector<float> row(n_in);
    std::fill(out, out + n_in, 0.0f);
    for (int c = 0; c < n_cb; c++) {
        imi_q8_0_dequant_row((const uint8_t *) out_embd->data + (size_t) (codes[c] + c * n_vocab) * out_embd->nb[1], row.data(), n_in);
        for (int i = 0; i < n_in; i++) out[i] += row[i];
    }
}

void imi_voice::begin_frame(const float * h) {
    ctr[0].store(0);
    team->run([&](int tid) {
        (void) tid;
        const int ng = proj_in.rows_pad / 8;
        float o[8];
        for (int g = ctr[0].fetch_add(8); g < ng; g = ctr[0].fetch_add(8))
            for (int gg = g; gg < std::min(ng, g + 8); gg++) {
                imi_q8g_group(proj_in, h, 1, gg, o);
                for (int r = 0; r < 8; r++) base[8 * gg + r] = o[r] + proj_b[8 * gg + r];
            }
    });
}

void imi_voice::pass(const int32_t * codes, const std::vector<int> & need, std::vector<float> & out) {
    out.resize(need.size() * (size_t) n_vocab);
    for (int i = 0; i < 64; i++) ctr[i].store(0, std::memory_order_relaxed);
    team->run([&](int tid) { run_pass(tid, codes, need, out.data()); });
}

void imi_voice::run_pass(int tid, const int32_t * codes, const std::vector<int> & need, float * out) {
    per_thread & T = per[tid];
    const int E = n_embd, P = n_cb, D = head_dim, Hq = n_head * D, Hk = n_head_kv * D, grp = n_head / n_head_kv;
    float * x = T.x.data(), * h = T.h.data(), * xt = T.xt.data();
    for (int i = 0; i < P; i++) {
        float * row = x + (size_t) i * E;
        if (codes[i] >= 0) imi_q8_0_dequant_row((const uint8_t *) embd->data + (size_t) codes[i] * embd->nb[1] + (size_t) i * embd->nb[2], row, E);
        else memcpy(row, mask_embd, (size_t) E * sizeof(float));
        for (int e = 0; e < E; e++) row[e] += base[(size_t) i * E + e];
    }
    int c = 0;
    // one table times all P positions: rows split in chunks; out = [P][ldo]
    auto table = [&](const imi_q8g & W, float * o, int ldo) {
        std::atomic<int> & k = ctr[c++];
        const int ng = W.rows_pad / 8;
        float tmp[12 * 8];
        for (int g = k.fetch_add(2, std::memory_order_relaxed); g < ng; g = k.fetch_add(2, std::memory_order_relaxed))
            for (int gg = g; gg < std::min(ng, g + 2); gg++) {
                imi_q8g_group(W, xt, P, gg, tmp);
                for (int p = 0; p < P; p++) for (int r = 0; r < 8 && 8 * gg + r < W.rows; r++) o[(size_t) p * ldo + 8 * gg + r] = tmp[8 * p + r];
            }
    };
    auto table_ffn = [&](const imi_q8g & W, float * o) {   // (gate, up) group pairs -> silu(gate) * up
        std::atomic<int> & k = ctr[c++];
        const int ng = W.rows_pad / 8;
        float a[12 * 8], b[12 * 8];
        for (int g = k.fetch_add(2, std::memory_order_relaxed); g < ng; g = k.fetch_add(2, std::memory_order_relaxed)) {
            imi_q8g_group(W, xt, P, g, a); imi_q8g_group(W, xt, P, g + 1, b);
            for (int p = 0; p < P; p++) for (int r = 0; r < 8; r++) o[(size_t) p * n_ff + 4 * g + r] = silu(a[8 * p + r]) * b[8 * p + r];
        }
    };
    for (const layer & L : layers) {
        for (int p = 0; p < P; p++) imi_norm_mul(x + (size_t) p * E, L.ln1, h + (size_t) p * E, E, eps);
        transpose(h, P, E, E, xt);
        table(L.qkvg, qkv.data(), n_qkv);
        team->sync();
        for (int hh = tid; hh < n_head; hh += n_threads) {    // each thread: whole query heads
            const int kv = hh / grp;
            float kk[12][64], qq[12][64], sc[12];
            for (int p = 0; p < P; p++) {
                const float * r = qkv.data() + (size_t) p * n_qkv;
                imi_norm_mul(r + Hq + kv * D, L.k_norm, kk[p], D, eps); rope(kk[p], D, p, rope_theta);
                imi_norm_mul(r + hh * D, L.q_norm, qq[p], D, eps);       rope(qq[p], D, p, rope_theta);
            }
            const float scale = 1.0f / sqrtf((float) D);
            for (int p = 0; p < P; p++) {
                float mx = -INFINITY;
                for (int t = 0; t < P; t++) { float s = 0; for (int i = 0; i < D; i++) s += qq[p][i] * kk[t][i]; sc[t] = s * scale; mx = std::max(mx, sc[t]); }
                float sum = 0;
                for (int t = 0; t < P; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                const float gate = 1.0f / (1.0f + expf(-qkv[(size_t) p * n_qkv + Hq + 2 * Hk + hh]));
                float * o = att.data() + (size_t) p * Hq + hh * D;
                for (int i = 0; i < D; i++) {
                    float a = 0;
                    for (int t = 0; t < P; t++) a += sc[t] * qkv[(size_t) t * n_qkv + Hq + Hk + kv * D + i];
                    o[i] = a / sum * gate;
                }
            }
        }
        team->sync();
        transpose(att.data(), P, Hq, Hq, xt);
        table(L.o, y.data(), E);
        team->sync();
        for (size_t i = 0; i < (size_t) P * E; i++) x[i] += y[i];
        for (int p = 0; p < P; p++) imi_norm_mul(x + (size_t) p * E, L.ln2, h + (size_t) p * E, E, eps);
        transpose(h, P, E, E, xt);
        table_ffn(L.gate_up, act.data());
        team->sync();
        transpose(act.data(), P, n_ff, n_ff, xt);
        table(L.down, y.data(), E);
        team->sync();
        for (size_t i = 0; i < (size_t) P * E; i++) x[i] += y[i];
    }
    // final norms, then the needed codebooks' scores (each reads its own code table with its own position)
    for (int p = 0; p < P; p++) {
        imi_norm_mul(x + (size_t) p * E, out_norm, h + (size_t) p * E, E, eps);
        imi_norm_mul(h + (size_t) p * E, head_norm + (size_t) p * E, xt + (size_t) p * E, E, eps);
    }
    std::atomic<int> & k = ctr[c++];
    const int ng = score[0].rows_pad / 8, units = (int) need.size() * ng;
    float tmp[8];
    for (int u = k.fetch_add(4, std::memory_order_relaxed); u < units; u = k.fetch_add(4, std::memory_order_relaxed))
        for (int uu = u; uu < std::min(units, u + 4); uu++) {
            const int j = uu / ng, gg = uu % ng, cb = need[j];
            imi_q8g_group(score[cb], xt + (size_t) cb * E, 1, gg, tmp);
            for (int r = 0; r < 8 && 8 * gg + r < n_vocab; r++) out[(size_t) j * n_vocab + 8 * gg + r] = tmp[r];
        }
}

imi_voice::~imi_voice() {
    proj_in.release();
    for (auto & L : layers) for (imi_q8g * w : { &L.qkvg, &L.o, &L.gate_up, &L.down }) w->release();
    for (auto & s : score) s.release();
}
