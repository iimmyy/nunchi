// Runs imi's own codec on the final codes of a pass dump (LG_GEN_DUMP) and compares the sound with the WAV
// llama.cpp wrote in the same run; times one frame.
//   imi-codec-check MMPROJ.gguf DUMP.bin REF.wav [threads] [OUT.wav]
#include "codec.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: imi-codec-check MMPROJ.gguf DUMP.bin REF.wav [threads] [OUT.wav]\n"); return 1; }
    imi_codec cd;
    if (!cd.load(argv[1], argc > 4 ? atoi(argv[4]) : 4)) return 1;
    std::vector<std::vector<int32_t>> frames;
    FILE * f = fopen(argv[2], "rb");
    for (;;) {
        int32_t kind, ne, no;
        if (!f || fread(&kind, 4, 1, f) != 1) break;
        fread(&ne, 4, 1, f); fseek(f, 4L * ne, SEEK_CUR);
        std::vector<int32_t> codes(12); fread(codes.data(), 4, 12, f);
        fread(&no, 4, 1, f); fseek(f, 4L * no, SEEK_CUR);
        if (kind == 1) frames.push_back(codes);    // every code known: the frame the codec turns into sound
    }
    if (f) fclose(f);
    std::vector<float> pcm;
    auto t0 = std::chrono::steady_clock::now();
    for (auto & c : frames) cd.frame(c.data(), pcm);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / frames.size();
    cd.flush(pcm);

    FILE * w = fopen(argv[3], "rb");
    std::vector<int16_t> ref;
    if (w) { fseek(w, 44, SEEK_SET); int16_t s; while (fread(&s, 2, 1, w) == 1) ref.push_back(s); fclose(w); }
    double num = 0, den = 0, mx = 0;
    const size_t n = std::min(ref.size(), pcm.size());
    for (size_t i = 0; i < n; i++) {
        const double r = ref[i] / 32767.0, d = pcm[i] - r;
        num += d * d; den += r * r; mx = std::max(mx, fabs(d));
    }
    printf("codec: %zu frames, %zu samples (reference %zu); vs llama.cpp's sound: SNR %.1f dB, max difference %.5f\n",
           frames.size(), pcm.size(), ref.size(), 10 * log10(den / (num + 1e-30)), mx);
    printf("time: %.2f ms/frame, %d threads\n", ms, cd.n_threads);
    if (argc > 5) {
        FILE * o = fopen(argv[5], "wb");
        const int sr = 24000; const uint32_t nb = (uint32_t) pcm.size() * 2, cs = 36 + nb, br = sr * 2; const uint16_t one = 1, ba = 2, bits = 16;
        const uint32_t s16 = 16, srr = sr;
        fwrite("RIFF", 1, 4, o); fwrite(&cs, 4, 1, o); fwrite("WAVEfmt ", 1, 8, o); fwrite(&s16, 4, 1, o); fwrite(&one, 2, 1, o); fwrite(&one, 2, 1, o);
        fwrite(&srr, 4, 1, o); fwrite(&br, 4, 1, o); fwrite(&ba, 2, 1, o); fwrite(&bits, 2, 1, o); fwrite("data", 1, 4, o); fwrite(&nb, 4, 1, o);
        for (float s : pcm) { const int16_t v = (int16_t) lrintf(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f); fwrite(&v, 2, 1, o); }
        fclose(o);
    }
    return 0;
}
