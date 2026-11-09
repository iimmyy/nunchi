// Checks imi's brain against llama.cpp's answer key (ref/ref_llama.cpp), position by position, then times a step.
//   imi-brain-check MODEL.gguf REF.bin [threads]
#include "brain.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static double rel(const float * a, const float * b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return sqrt(num / (den + 1e-30));
}
static int top(const float * a, int n) { int t = 0; for (int i = 1; i < n; i++) if (a[i] > a[t]) t = i; return t; }

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: imi-brain-check MODEL.gguf REF.bin [threads]\n"); return 1; }
    ggml_cpu_init();
    imi_brain br;
    if (!br.load(argv[1], argc > 3 ? atoi(argv[3]) : 4, 1024)) return 1;
    FILE * f = fopen(argv[2], "rb");
    int32_t ne, nv, n;
    if (!f || fread(&ne, 4, 1, f) != 1) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    fread(&nv, 4, 1, f); fread(&n, 4, 1, f);
    if (ne != br.n_embd || nv != br.n_vocab) { fprintf(stderr, "answer key is for another model\n"); return 1; }
    std::vector<int32_t> tok(n); std::vector<float> hid((size_t) n * ne), lg((size_t) n * nv);
    for (int i = 0; i < n; i++) { fread(&tok[i], 4, 1, f); fread(&hid[(size_t) i * ne], 4, ne, f); fread(&lg[(size_t) i * nv], 4, nv, f); }
    fclose(f);

    double hmax = 0, hsum = 0, lmax = 0, lsum = 0; int same = 0;
    for (int i = 0; i < n; i++) {
        br.step(tok[i], nullptr, true);
        const double eh = rel(br.hidden.data(), &hid[(size_t) i * ne], ne), el = rel(br.logits.data(), &lg[(size_t) i * nv], nv);
        hmax = std::max(hmax, eh); hsum += eh; lmax = std::max(lmax, el); lsum += el;
        same += top(br.logits.data(), nv) == top(&lg[(size_t) i * nv], nv);
        if (i < 3 || eh > 0.05) printf("  pos %d: hidden error %.2e, score error %.2e\n", i, eh, el);
    }
    printf("check: %d positions, hidden error mean %.2e max %.2e, score error mean %.2e max %.2e, same next word %d/%d\n",
           n, hsum / n, hmax, lsum / n, lmax, same, n);

    // where the road goes: only the big table reads of one step, back to back (no norms, conv, attention, adds)
    {
        std::vector<uint8_t> ar(64u << 20), wk;
        ggml_init_params ip = { ar.size(), ar.data(), false };
        ggml_context * c = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 4096, false);
        size_t bytes = 0;
        const int ncol = getenv("IMI_NCOL") ? atoi(getenv("IMI_NCOL")) : 1;   // jobs riding one step
        auto mm = [&](ggml_tensor * w) {
            if (!w) return;
            ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, w->ne[0], ncol);
            for (int64_t i = 0; i < w->ne[0] * ncol; i++) ((float *) x->data)[i] = 0.01f * (float) (i % 7);
            ggml_build_forward_expand(gf, ggml_mul_mat(c, w, x));
            bytes += ggml_nbytes(w);
        };
        for (auto & L : br.layers) {
            if (L.qkvg) mm(L.qkvg); else { mm(L.q); mm(L.k); mm(L.v); mm(L.gate); }
            if (L.gate_up) mm(L.gate_up); else { mm(L.ffn_gate); mm(L.ffn_up); }
            mm(L.in_proj); mm(L.out_proj); mm(L.o); mm(L.ffn_down);
        }
        double best = 1e9;
        for (int r = 0; r < 20; r++) {
            ggml_cplan plan = ggml_graph_plan(gf, br.n_threads, br.tp);
            if (wk.size() < plan.work_size) wk.resize(plan.work_size);
            plan.work_data = wk.data();
            auto t0 = std::chrono::steady_clock::now();
            ggml_graph_compute(gf, &plan);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("tables only, %d input(s) per step: %d reads, %.0f MB in %.1f ms = %.1f GB/s\n", ncol, ggml_graph_n_nodes(gf), bytes / 1e6, best, bytes / 1e6 / best);
        ggml_free(c);
    }

    // time: hidden state only (an audio frame) and with word scores (a text step)
    for (int want = 0; want < 2; want++) {
        br.reset();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) br.step(tok[i], nullptr, want);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("time: %.1f ms/step %s, %d threads\n", ms / n, want ? "with word scores" : "hidden state only", br.n_threads);
    }
    printf("graph: %d nodes, %d that compute (each ends in a stop)\n", br.last_nodes, br.last_compute_nodes);
    return 0;
}
