// Checks imi's own voice generator (core/voice) against llama.cpp's dump (LG_GEN_DUMP), every pass, then times it.
//   imi-voice-check MMPROJ.gguf DUMP.bin [threads]
#include "voice.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
    if (argc < 3) { fprintf(stderr, "usage: imi-voice-check MMPROJ.gguf DUMP.bin [threads]\n"); return 1; }
    imi_voice g;
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
    FILE * dump = getenv("IMI_DUMP") ? fopen(getenv("IMI_DUMP"), "wb") : nullptr;   // our scores, dump format
    for (size_t r = 0; r < rs.size(); r++) {
        if (rs[r].kind != 0) continue;
        if (!last_h || rs[r].h != rs[r - 1].h) g.begin_frame(rs[r].h.data());
        last_h = rs[r].h.data();
        g.pass(rs[r].codes.data(), all, out);
        if (dump) { int32_t kd = 0, ne = (int32_t) rs[r].h.size(), no = (int32_t) out.size(); fwrite(&kd, 4, 1, dump); fwrite(&ne, 4, 1, dump); fwrite(rs[r].h.data(), 4, ne, dump); fwrite(rs[r].codes.data(), 4, g.n_cb, dump); fwrite(&no, 4, 1, dump); fwrite(out.data(), 4, no, dump); }
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
    if (dump) fclose(dump);
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
