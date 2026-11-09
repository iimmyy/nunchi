// one 8-bit table (voice generator FFN down, 2816 -> 1024) times 12 inputs, one thread: how fast is our kernel?
#include "tables.h"
#include "../model.h"
#include <chrono>
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    imi_file f; if (!f.load(argv[1], false)) return 1;
    const char * name = argc > 2 ? argv[2] : "a.gen.code.blk.0.ffn_down.weight";
    const ggml_tensor * t = f.get(name);
    std::vector<std::pair<const ggml_tensor *, int>> o; for (int r = 0; r < t->ne[1]; r++) o.push_back({ t, r });
    imi_q8g W; W.build(o);
    std::vector<float> xt((size_t) W.cols * 12, 0.01f), out(12 * 8);
    for (int C : { 12, 1 }) {
        double best = 1e9;
        for (int rep = 0; rep < 20; rep++) {
            auto t0 = std::chrono::steady_clock::now();
            for (int g = 0; g < W.rows_pad / 8; g++) imi_q8g_group(W, xt.data(), C, g, out.data());
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const double macs = (double) W.rows * W.cols * C;
        printf("%s x %d inputs: %.2f ms, %.2f GMAC/s (one thread)\n", name, C, best, macs / best / 1e6);
    }
    return 0;
}
