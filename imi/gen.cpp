// imi.cpp voice generator (LFM3-Omni depthformer): one audio frame = 12 codes, picked in 3 masked passes.
//
// Our structure, written from the model's math (pre-norm blocks, GQA attention with q/k norm, NEOX rope, per-head
// sigmoid gate, SiLU-gated FFN, bidirectional over the 12 codebook positions):
//   - the input projection depends only on the brain's hidden state, so it runs once per frame, not once per pass
//   - a pass scores only the codebooks it is about to pick, not all 12
//
//   imi-gen MMPROJ.gguf DUMP.bin [threads]   checks every pass against llama.cpp's dump (LG_GEN_DUMP), then times it
#include "model.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct gen_layer {
    ggml_tensor *ln1, *ln2, *q, *k, *v, *o, *gate, *q_norm, *k_norm, *ffn_up, *ffn_gate, *ffn_down;
};

struct imi_gen {
    imi_file f;
    ggml_tensor *proj_in_w, *proj_in_b, *embd_w, *mask_embd, *out_norm, *head_norm, *out_embd;
    std::vector<gen_layer> layers;
    int n_embd = 0, n_vocab = 0, n_cb = 0, n_head = 32, n_head_kv = 8, head_dim = 0;
    float eps = 1e-5f, rope_theta = 1000000.0f;
    int n_threads = 4;

    ggml_threadpool * tp = nullptr;
    std::vector<uint8_t> arena, work;
    std::vector<float> base;      // proj_in(h) + b for the current frame, [n_cb][n_embd]
    std::vector<float> tok;       // one token row, dequantized

    bool load(const std::string & path, int threads) {
        if (!f.load(path)) return false;
        proj_in_w = f.get("a.gen.code.proj_in.weight");
        proj_in_b = f.get("a.gen.code.proj_in.bias");
        embd_w    = f.get("a.gen.code.embd.weight");
        mask_embd = f.get("a.gen.code.mask_embd");
        out_norm  = f.get("a.gen.code.output_norm.weight");
        head_norm = f.get("a.gen.code.head_norm.weight");
        out_embd  = f.get("a.gen.code.out_embd.weight");
        n_embd  = (int) embd_w->ne[0];
        n_vocab = (int) embd_w->ne[1];
        n_cb    = (int) embd_w->ne[2];
        for (int il = 0;; il++) {
            const std::string p = "a.gen.code.blk." + std::to_string(il) + ".";
            if (!f.get(p + "ln1.weight", false)) break;
            gen_layer L;
            L.ln1 = f.get(p + "ln1.weight");          L.ln2 = f.get(p + "ln2.weight");
            L.q = f.get(p + "attn_q.weight");         L.k = f.get(p + "attn_k.weight");
            L.v = f.get(p + "attn_v.weight");         L.o = f.get(p + "attn_out.weight");
            L.gate = f.get(p + "attn_gate.weight");
            L.q_norm = f.get(p + "attn_q_norm.weight"); L.k_norm = f.get(p + "attn_k_norm.weight");
            L.ffn_up = f.get(p + "ffn_up.weight");    L.ffn_gate = f.get(p + "ffn_gate.weight");
            L.ffn_down = f.get(p + "ffn_down.weight");
            layers.push_back(L);
        }
        head_dim = (int) layers[0].q_norm->ne[0];
        n_head    = (int) (layers[0].q->ne[1] / head_dim);
        n_head_kv = (int) (layers[0].k->ne[1] / head_dim);
        n_threads = threads;
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        tp = ggml_threadpool_new(&tpp);
        arena.resize(64u << 20);
        fprintf(stderr, "imi-gen: %zu layers, n_embd %d, %d codebooks x %d codes, %d heads (%d kv) x %d\n",
                layers.size(), n_embd, n_cb, n_vocab, n_head, n_head_kv, head_dim);
        return true;
    }

    ggml_context * new_ctx() {
        ggml_init_params ip = { arena.size(), arena.data(), false };
        return ggml_init(ip);
    }

    void compute(ggml_context * ctx, ggml_cgraph * gf) {
        ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
        if (work.size() < plan.work_size) work.resize(plan.work_size);
        plan.work_data = work.data();
        ggml_graph_compute(gf, &plan);
        (void) ctx;
    }

    // once per frame: the 12 per-codebook inputs before the token / mask embeddings are added
    void begin_frame(const float * h) {
        ggml_context * ctx = new_ctx();
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, proj_in_w->ne[0]);
        memcpy(x->data, h, ggml_nbytes(x));
        ggml_tensor * y = ggml_add(ctx, ggml_mul_mat(ctx, proj_in_w, x), proj_in_b);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        compute(ctx, gf);
        base.assign((float *) y->data, (float *) y->data + (size_t) n_embd * n_cb);
        ggml_free(ctx);
    }

    // one masked pass: codes[i] < 0 = still masked. Writes logits for the codebooks in `need` (n_vocab each).
    void pass(const int32_t * codes, const std::vector<int> & need, std::vector<float> & out) {
        ggml_context * ctx = new_ctx();
        const int E = n_embd, P = n_cb, D = head_dim;

        // depth input: base + (token embedding of a revealed code, or the mask embedding)
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, P);
        float * xd = (float *) x->data;
        const float * me = (const float *) mask_embd->data;
        for (int i = 0; i < P; i++) {
            float * row = xd + (size_t) i * E;
            const float * b = base.data() + (size_t) i * E;
            if (codes[i] >= 0) {
                // the token row in whatever type the table is stored in (16-bit or 8-bit)
                tok.resize(E);
                ggml_get_type_traits(embd_w->type)->to_float((const char *) embd_w->data
                    + (size_t) codes[i] * embd_w->nb[1] + (size_t) i * embd_w->nb[2], tok.data(), E);
                for (int e = 0; e < E; e++) row[e] = b[e] + tok[e];
            } else {
                for (int e = 0; e < E; e++) row[e] = b[e] + me[e];
            }
        }

        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
        for (int i = 0; i < P; i++) ((int32_t *) pos->data)[i] = i;

        ggml_tensor * cur = x;
        for (const auto & L : layers) {
            ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), L.ln1);

            ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.q, h), D, n_head, P);
            ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.k, h), D, n_head_kv, P);
            ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.v, h), D, n_head_kv, P);
            q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), L.q_norm);
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), L.k_norm);
            q = ggml_rope_ext(ctx, q, pos, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, pos, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // bidirectional attention over the 12 positions, each kv head shared by n_head / n_head_kv query heads
            ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));      // [D, P, n_head]
            ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));      // [D, P, n_head_kv]
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));      // [P, D, n_head_kv]
            ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kp, qp), nullptr, 1.0f / sqrtf((float) D), 0.0f);
            ggml_tensor * a = ggml_mul_mat(ctx, vt, kq);                               // [D, P, n_head]
            a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));                      // [D, n_head, P]

            ggml_tensor * g = ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.gate, h));         // [n_head, P]
            a = ggml_mul(ctx, a, ggml_reshape_3d(ctx, g, 1, n_head, P));
            a = ggml_mul_mat(ctx, L.o, ggml_reshape_2d(ctx, a, D * n_head, P));
            cur = ggml_add(ctx, cur, a);

            h = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), L.ln2);
            ggml_tensor * ff = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, L.ffn_gate, h)), ggml_mul_mat(ctx, L.ffn_up, h));
            cur = ggml_add(ctx, cur, ggml_mul_mat(ctx, L.ffn_down, ff));
        }
        cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), out_norm);
        cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), head_norm);   // per codebook weights

        ggml_cgraph * gf = ggml_new_graph(ctx);
        std::vector<ggml_tensor *> outs;
        for (int i : need) {
            ggml_tensor * w = ggml_view_2d(ctx, embd_w, E, n_vocab, embd_w->nb[1], (size_t) i * embd_w->nb[2]);
            ggml_tensor * c = ggml_view_2d(ctx, cur, E, 1, cur->nb[1], (size_t) i * cur->nb[1]);
            outs.push_back(ggml_mul_mat(ctx, w, c));
            ggml_build_forward_expand(gf, outs.back());
        }
        compute(ctx, gf);
        out.resize((size_t) need.size() * n_vocab);
        for (size_t j = 0; j < need.size(); j++) memcpy(out.data() + j * n_vocab, outs[j]->data, (size_t) n_vocab * 4);
        ggml_free(ctx);
    }
};

struct rec {
    int kind;
    std::vector<float> h, out;
    std::vector<int32_t> codes;
};

static std::vector<rec> read_dump(const char * path, int n_cb) {
    std::vector<rec> rs;
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    for (;;) {
        rec r; int32_t ne, no;
        if (fread(&r.kind, 4, 1, f) != 1) break;
        fread(&ne, 4, 1, f); r.h.resize(ne); fread(r.h.data(), 4, ne, f);
        r.codes.resize(n_cb); fread(r.codes.data(), 4, n_cb, f);
        fread(&no, 4, 1, f); r.out.resize(no); fread(r.out.data(), 4, no, f);
        rs.push_back(std::move(r));
    }
    fclose(f);
    return rs;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: imi-gen MMPROJ.gguf DUMP.bin [threads]\n"); return 1; }
    ggml_cpu_init();
    imi_gen g;
    if (!g.load(argv[1], argc > 3 ? atoi(argv[3]) : 4)) return 1;
    std::vector<rec> rs = read_dump(argv[2], g.n_cb);

    // the codebooks each pass picks = the ones revealed in the next record
    std::vector<std::vector<int>> need(rs.size());
    for (size_t r = 0; r < rs.size(); r++) {
        if (rs[r].kind != 0) continue;
        if (r + 1 < rs.size() && rs[r + 1].h == rs[r].h) {
            for (int i = 0; i < g.n_cb; i++) if (rs[r].codes[i] < 0 && rs[r + 1].codes[i] >= 0) need[r].push_back(i);
        }
        if (need[r].empty()) for (int i = 0; i < g.n_cb; i++) if (rs[r].codes[i] < 0) need[r].push_back(i);
    }

    // 1) check: all 12 codebooks' scores against llama.cpp
    std::vector<int> all(g.n_cb);
    for (int i = 0; i < g.n_cb; i++) all[i] = i;
    std::vector<float> out;
    const float * last_h = nullptr;
    double max_rel = 0, sum_rel = 0; long n_pass = 0, n_top = 0, n_top_same = 0;
    for (size_t r = 0; r < rs.size(); r++) {
        if (rs[r].kind != 0) continue;
        if (!last_h || rs[r].h != rs[r - 1].h) g.begin_frame(rs[r].h.data());
        last_h = rs[r].h.data();
        g.pass(rs[r].codes.data(), all, out);
        double num = 0, den = 0;
        for (size_t j = 0; j < out.size(); j++) { double d = out[j] - rs[r].out[j]; num += d * d; den += (double) rs[r].out[j] * rs[r].out[j]; }
        const double rel = sqrt(num / den);
        max_rel = std::max(max_rel, rel); sum_rel += rel; n_pass++;
        for (int i : need[r]) {
            const float * a = out.data() + (size_t) i * g.n_vocab, * b = rs[r].out.data() + (size_t) i * g.n_vocab;
            int ia = 0, ib = 0;
            for (int c = 1; c < g.n_vocab; c++) { if (a[c] > a[ia]) ia = c; if (b[c] > b[ib]) ib = c; }
            n_top++; n_top_same += ia == ib;
        }
    }
    printf("check: %ld passes, score error vs llama.cpp mean %.2e max %.2e, top code same %ld/%ld\n",
           n_pass, sum_rel / n_pass, max_rel, n_top_same, n_top);

    // 2) time: per frame = proj once + 3 passes scoring only the picked codebooks
    for (int rep = 0; rep < 2; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        long frames = 0;
        for (size_t r = 0; r < rs.size(); r++) {
            if (rs[r].kind != 0) continue;
            if (r == 0 || rs[r].h != rs[r - 1].h) { g.begin_frame(rs[r].h.data()); frames++; }
            g.pass(rs[r].codes.data(), need[r], out);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("time (run %d): %ld frames, %.1f ms/frame with %d threads\n", rep + 1, frames, ms / frames, g.n_threads);
    }
    return 0;
}
