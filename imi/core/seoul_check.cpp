// Checks imi's own brain (core/seoul) against llama.cpp's answer key, position by position, then times a step.
//   imi-seoul-check MODEL.gguf REF.bin [threads]
#include "seoul.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double rel(const float * a, const float * b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return sqrt(num / (den + 1e-30));
}
static int top(const float * a, int n) { int t = 0; for (int i = 1; i < n; i++) if (a[i] > a[t]) t = i; return t; }

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: imi-seoul-check MODEL.gguf REF.bin [threads]\n"); return 1; }
    imi_seoul br;
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
    FILE * dump = getenv("IMI_DUMP") ? fopen(getenv("IMI_DUMP"), "wb") : nullptr;   // our answers, same format
    if (dump) { fwrite(&ne, 4, 1, dump); fwrite(&nv, 4, 1, dump); fwrite(&n, 4, 1, dump); }
    for (int i = 0; i < n; i++) {
        br.step(tok[i], nullptr, true);
        if (dump) { fwrite(&tok[i], 4, 1, dump); fwrite(br.hidden.data(), 4, ne, dump); fwrite(br.logits.data(), 4, nv, dump); }
        const double eh = rel(br.hidden.data(), &hid[(size_t) i * ne], ne), el = rel(br.logits.data(), &lg[(size_t) i * nv], nv);
        hmax = std::max(hmax, eh); hsum += eh; lmax = std::max(lmax, el); lsum += el;
        same += top(br.logits.data(), nv) == top(&lg[(size_t) i * nv], nv);
        if (i < 3 || eh > 0.1) printf("  pos %d: hidden error %.2e, score error %.2e\n", i, eh, el);
    }
    if (dump) fclose(dump);
    printf("check: %d positions, hidden error mean %.2e max %.2e, score error mean %.2e max %.2e, same next word %d/%d\n",
           n, hsum / n, hmax, lsum / n, lmax, same, n);

    if (getenv("IMI_BENCH")) {
        for (int nc = 1; nc <= 2; nc++) printf("tables only, %d input(s): %.1f ms\n", nc, br.bench_tables(nc, 20));
        return 0;
    }
    if (getenv("IMI_BATCH")) {   // read all positions in batches, compare every position's hidden state
        std::vector<float> all((size_t) n * ne);
        br.reset();
        auto t0 = std::chrono::steady_clock::now();
        br.step_batch(tok.data(), nullptr, n, true, all.data());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        double hs = 0, hm = 0;
        for (int i = 0; i < n; i++) { const double e = rel(&all[(size_t) i * ne], &hid[(size_t) i * ne], ne); hs += e; hm = std::max(hm, e); }
        printf("batch: %d positions in %.1f ms (%.2f ms each), hidden error mean %.2e max %.2e, last word same %d\n", n, ms, ms / n, hs / n, hm,
               top(br.logits.data(), nv) == top(&lg[(size_t) (n - 1) * nv], nv));
        FILE * bd = getenv("IMI_DUMP_BATCH") ? fopen(getenv("IMI_DUMP_BATCH"), "wb") : nullptr;
        if (bd) {   // hidden states only; scores left at zero except the last
            fwrite(&ne, 4, 1, bd); fwrite(&nv, 4, 1, bd); fwrite(&n, 4, 1, bd);
            std::vector<float> z(nv, 0.0f);
            for (int i = 0; i < n; i++) { fwrite(&tok[i], 4, 1, bd); fwrite(&all[(size_t) i * ne], 4, ne, bd); fwrite(i == n - 1 ? br.logits.data() : z.data(), 4, nv, bd); }
            fclose(bd);
        }
        // a long read: the same words over and over to 512 positions
        std::vector<int32_t> lt(512); for (int i = 0; i < 512; i++) lt[i] = tok[i % n];
        br.reset(); t0 = std::chrono::steady_clock::now();
        br.step_batch(lt.data(), nullptr, 512, false);
        const double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("batch: 512 positions in %.0f ms = %.1f words/s\n", ms2, 512 / ms2 * 1e3);
        return 0;
    }
    const int reps = getenv("IMI_REPS") ? atoi(getenv("IMI_REPS")) : 3;
    for (int want = 0; want < 2; want++) {
        double best = 1e9, tab = 0, tsy = 0, tat = 0;
        for (int r = 0; r < reps; r++) {
            br.reset(); br.t_tables = br.t_sync = br.t_attn = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; i++) br.step(tok[i], nullptr, want);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
            if (ms < best) { best = ms; tab = br.t_tables / n; tsy = br.t_sync / n; tat = br.t_attn / n; }
        }
        printf("time: %.1f ms/step %s (thread 0: layer tables %.1f, waiting at stops %.1f, attention %.2f ms), %d threads\n", best, want ? "with word scores" : "hidden state only", tab, tsy, tat, br.n_threads);
    }
    return 0;
}
