// imi-ears-causal: research, may the ears' rows be read while the speaker is still talking? Encodes a clip whole, then
// every growing start of it (every STEP seconds), and for each start prints how many of its last rows differ from the
// whole clip's rows (beyond 1e-4) and the largest difference in the rows before those. If only a few last rows ever
// differ, rows that far from the end are final as soon as their audio is in.
//   imi-ears-causal MMPROJ.gguf CLIP.f32 [threads=4] [STEP=0.5]
#include "ears.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: imi-ears-causal MMPROJ.gguf CLIP.f32 [threads] [STEP]\n"); return 1; }
    const int threads = argc > 3 ? atoi(argv[3]) : 4;
    const double step = argc > 4 ? atof(argv[4]) : 0.5;
    imi_ears ears;
    if (!ears.load(argv[1], threads)) return 1;
    std::vector<float> pcm;
    {
        FILE * f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "no clip %s\n", argv[2]); return 1; }
        float s; while (fread(&s, 4, 1, f) == 1) pcm.push_back(s);
        fclose(f);
    }
    std::vector<float> whole;
    const int P = ears.encode(pcm.data(), (int) pcm.size(), whole);
    const int D = (int) (whole.size() / std::max(1, P));
    printf("whole: %.2f s, %d rows of %d\n", pcm.size() / 16000.0, P, D);
    int worst_tail = 0;
    double worst_before = 0;
    for (double t = step; t < pcm.size() / 16000.0; t += step) {
        const int n = (int) (t * 16000);
        std::vector<float> part;
        const int p = ears.encode(pcm.data(), n, part);
        std::vector<double> d(p, 0.0);   // each row's largest difference from the whole clip's row
        for (int r = 0; r < p && r < P; r++)
            for (int c = 0; c < D; c++) d[r] = std::max(d[r], (double) std::fabs(part[(size_t) r * D + c] - whole[(size_t) r * D + c]));
        int tail = 0;                    // the last rows that differ
        while (tail < p && d[p - 1 - tail] > 1e-4) tail++;
        double before = 0;
        for (int r = 0; r < p - tail; r++) before = std::max(before, d[r]);
        worst_tail = std::max(worst_tail, tail);
        worst_before = std::max(worst_before, before);
        printf("  %.2f s: %d rows, last %d differ, rows before them differ at most %.2e\n", t, p, tail, before);
    }
    printf("imi-ears-causal: at most the last %d rows change as audio comes in; the rows before them at most %.2e\n", worst_tail, worst_before);
    return 0;
}
