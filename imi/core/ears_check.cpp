// Runs imi's ears on a 16 kHz clip and compares the result with the 64-bit answer key (ref/ears_exact.py) and with
// any other answers given (e.g. the fork's MTMD_DEBUG_EMBEDDINGS dump); times the clip.
//   imi-ears-check MMPROJ.gguf CLIP.f32 TRUTH.bin [threads] [reps] [NAME=OTHER.bin ...]
//   IMI_DUMP=ours.bin writes our answers; IMI_MEL=TRUTH_MEL.bin also compares the mel frames.
#include "ears.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_bin(const char * path, int & rows, int & cols, std::vector<float> & d) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    int32_t h[2];
    bool ok = fread(h, 4, 2, f) == 2;
    if (ok) { rows = h[0]; cols = h[1]; d.resize((size_t) rows * cols); ok = fread(d.data(), 4, d.size(), f) == d.size(); }
    fclose(f);
    return ok;
}
// relative error per row against the truth: mean, max, first row
static void report(const char * name, const float * a, const std::vector<float> & t, int rows, int cols) {
    double sum = 0, mx = 0, first = 0;
    for (int r = 0; r < rows; r++) {
        double num = 0, den = 0;
        for (int c = 0; c < cols; c++) { const double d = (double) a[(size_t) r * cols + c] - t[(size_t) r * cols + c]; num += d * d; den += (double) t[(size_t) r * cols + c] * t[(size_t) r * cols + c]; }
        const double e = sqrt(num / (den + 1e-30));
        sum += e; mx = std::max(mx, e); if (r == 0) first = e;
    }
    printf("%-10s rel err mean %.2e  max %.2e  pos0 %.2e  (%d x %d)\n", name, sum / rows, mx, first, rows, cols);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: imi-ears-check MMPROJ.gguf CLIP.f32 TRUTH.bin [threads] [reps] [NAME=OTHER.bin ...]\n"); return 1; }
    const int threads = argc > 4 ? atoi(argv[4]) : 4, reps = argc > 5 ? atoi(argv[5]) : 3;
    imi_ears ears;
    if (!ears.load(argv[1], threads)) return 1;
    std::vector<float> pcm;
    {
        FILE * f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "no clip %s\n", argv[2]); return 1; }
        float s; while (fread(&s, 4, 1, f) == 1) pcm.push_back(s);
        fclose(f);
    }
    std::vector<float> out;
    int P = 0;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        ears.t_mel = ears.t_down = ears.t_layers = ears.t_proj = 0;
        auto t0 = std::chrono::steady_clock::now();
        P = ears.encode(pcm.data(), (int) pcm.size(), out);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, s);
        printf("run %d: %.1f ms (mel %.1f, downsampler %.1f, layers %.1f, projector %.1f) for %.2f s of sound -> %d positions\n",
               r, 1e3 * s, 1e3 * ears.t_mel, 1e3 * ears.t_down, 1e3 * ears.t_layers, 1e3 * ears.t_proj, pcm.size() / 16000.0, P);
    }
    printf("best %.1f ms = %.4f CPU-s per sound second on %d threads\n", 1e3 * best, best / (pcm.size() / 16000.0), threads);
    if (const char * mp = getenv("IMI_MEL")) {
        int r, c; std::vector<float> tm;
        if (read_bin(mp, r, c, tm) && r * c == (int) ears.last_mel.size()) report("mel", ears.last_mel.data(), tm, r, c);
        else printf("mel: shape mismatch (truth %s)\n", mp);
    }
    int tr, tc; std::vector<float> truth;
    if (!read_bin(argv[3], tr, tc, truth)) { fprintf(stderr, "no truth %s\n", argv[3]); return 1; }
    if (tr != P || tc != ears.n_out) { printf("shape mismatch: ours %d x %d, truth %d x %d\n", P, ears.n_out, tr, tc); return 1; }
    report("imi", out.data(), truth, tr, tc);
    for (int i = 6; i < argc; i++) {
        std::string a = argv[i];
        const size_t eq = a.find('=');
        int r, c; std::vector<float> o;
        if (eq == std::string::npos || !read_bin(a.c_str() + eq + 1, r, c, o)) { printf("cannot read %s\n", a.c_str()); continue; }
        if (r != tr || c != tc) { printf("%s: shape %d x %d vs truth %d x %d\n", a.substr(0, eq).c_str(), r, c, tr, tc); continue; }
        report(a.substr(0, eq).c_str(), o.data(), truth, tr, tc);
    }
    if (const char * dp = getenv("IMI_DUMP")) {
        FILE * f = fopen(dp, "wb");
        const int32_t h[2] = { P, ears.n_out };
        fwrite(h, 4, 2, f); fwrite(out.data(), 4, out.size(), f); fclose(f);
    }
    return 0;
}
