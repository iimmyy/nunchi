// imi-speak: Seoul speaks, end to end on imi's own engine (brain, voice generator, codec on one thread team).
//   imi-speak MODEL.gguf MMPROJ.gguf PROMPT.ids OUT.wav [threads] [seed]
// PROMPT.ids: the word IDs of the "tts" chat template's prompt, as llama-tokenize --ids prints them.
// Per frame: the brain's hidden state -> the generator picks 12 codes in 2 masked passes (6 + 6, random order,
// temperature 0.4, repeat penalty 1.2 over the last 64 frames per codebook) -> the codec makes 1920 samples ->
// the 12 codes' feedback row sum is the brain's next input. Any end-of-audio code stops.
#include "seoul.h"
#include "voice.h"
#include "codec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

static void write_wav(const char * path, const std::vector<float> & pcm, int sr) {
    FILE * o = fopen(path, "wb");
    const uint32_t nb = (uint32_t) pcm.size() * 2, cs = 36 + nb, br = sr * 2, s16 = 16, srr = sr; const uint16_t one = 1, ba = 2, bits = 16;
    fwrite("RIFF", 1, 4, o); fwrite(&cs, 4, 1, o); fwrite("WAVEfmt ", 1, 8, o); fwrite(&s16, 4, 1, o); fwrite(&one, 2, 1, o); fwrite(&one, 2, 1, o);
    fwrite(&srr, 4, 1, o); fwrite(&br, 4, 1, o); fwrite(&ba, 2, 1, o); fwrite(&bits, 2, 1, o); fwrite("data", 1, 4, o); fwrite(&nb, 4, 1, o);
    for (float s : pcm) { const int16_t v = (int16_t) lrintf(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f); fwrite(&v, 2, 1, o); }
    fclose(o);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: imi-speak MODEL.gguf MMPROJ.gguf PROMPT.ids OUT.wav [threads] [seed]\n"); return 1; }
    const int nt = argc > 5 ? atoi(argv[5]) : 4;
    const unsigned seed = argc > 6 ? (unsigned) atoi(argv[6]) : 11;
    std::vector<int32_t> prompt;
    {
        std::ifstream in(argv[3]); std::stringstream ss; ss << in.rdbuf(); std::string s = ss.str();
        for (char & ch : s) if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
        std::stringstream ns(s); int v; while (ns >> v) prompt.push_back(v);
    }
    const auto t_load = clk::now();
    imi_team team; team.start(nt);
    imi_seoul brain; imi_voice voice; imi_codec codec;
    if (!brain.load(argv[1], nt, 4096, &team) || !voice.load(argv[2], nt, &team) || !codec.load(argv[2], nt, &team)) return 1;
    fprintf(stderr, "imi-speak: loaded in %.1f s, prompt %zu words\n", ms_since(t_load) / 1e3, prompt.size());

    const auto t_prompt = clk::now();
    brain.step_batch(prompt.data(), nullptr, (int) prompt.size(), false);   // the whole prompt, read in batches
    const double prompt_ms = ms_since(t_prompt);

    std::mt19937 rng(seed);
    const int n_cb = voice.n_cb, V = voice.n_vocab, eos = V - 1;
    const std::vector<int> schedule = { 6, 6 };
    const float temp = 0.4f, rep = 1.2f; const int rep_win = 64;
    std::vector<int32_t> hist;        // [frame][n_cb]
    std::vector<float> pcm, scores, fb(voice.n_in);
    double t_gen = 0, t_codec = 0, t_brain = 0; int frames = 0;
    const auto t_speak = clk::now();
    for (;;) {
        auto t0 = clk::now();
        std::vector<int> order(n_cb); for (int i = 0; i < n_cb; i++) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<int32_t> codes(n_cb, -1);
        voice.begin_frame(brain.hidden.data());
        int k = 0;
        for (int n_pick : schedule) {
            std::vector<int> need(order.begin() + k, order.begin() + k + n_pick);
            voice.pass(codes.data(), need, scores);
            for (int j = 0; j < n_pick; j++, k++) {
                const int cb = need[j];
                float * l = scores.data() + (size_t) j * V;
                // repeat penalty over the codes this codebook took in the last rep_win frames
                const size_t nh = hist.size() / n_cb, nw = std::min(nh, (size_t) rep_win);
                std::vector<bool> seen(V, false);
                for (size_t f = nh - nw; f < nh; f++) {
                    const int c = hist[f * n_cb + cb];
                    if (c >= 0 && c < V && !seen[c]) { seen[c] = true; l[c] = l[c] > 0 ? l[c] / rep : l[c] * rep; }
                }
                const float lmax = *std::max_element(l, l + V);
                std::vector<double> p(V); double sum = 0;
                for (int v = 0; v < V; v++) { p[v] = std::exp((double) (l[v] - lmax) / temp); sum += p[v]; }
                double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
                int pick = V - 1;
                for (int v = 0; v < V; v++) { r -= p[v]; if (r <= 0) { pick = v; break; } }
                codes[cb] = pick;
            }
        }
        t_gen += ms_since(t0);
        if (std::find(codes.begin(), codes.end(), eos) != codes.end() || frames >= 2000) break;
        hist.insert(hist.end(), codes.begin(), codes.end());
        t0 = clk::now();
        codec.frame(codes.data(), pcm);
        t_codec += ms_since(t0);
        t0 = clk::now();
        voice.feedback(codes.data(), fb.data());
        brain.step(-1, fb.data(), false);
        t_brain += ms_since(t0);
        frames++;
    }
    codec.flush(pcm);
    const double speak_ms = ms_since(t_speak), audio_s = pcm.size() / 24000.0;
    write_wav(argv[4], pcm, 24000);
    printf("imi-speak: %d frames, %.2f s of speech in %.2f s (%.2fx real time); prompt %zu words in %.2f s\n",
           frames, audio_s, speak_ms / 1e3, audio_s / (speak_ms / 1e3), prompt.size(), prompt_ms / 1e3);
    printf("stages per frame: generator %.1f ms, codec %.1f ms, brain %.1f ms\n", t_gen / std::max(frames, 1), t_codec / std::max(frames, 1), t_brain / std::max(frames, 1));
    return 0;
}
