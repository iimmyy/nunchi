// imi-hear-live: research, does Seoul (LFM3-Omni) hear live? The live hearing layout LFM3-Audio was trained on (the
// fork's mtmd_audio_stream_lfm3a, "trained layout"):
//   <prompt> [A x 12] text<|text_end|> [A x 12] text<|text_end|> ... [A x r] <|mixed_end|> text<|text_end|>
// prompt = <|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n
// 12 audio rows = 0.96 s. A chunk's text is greedy until <|text_end|> (48 words at most, 96 for the last chunk); the
// <|text_end|> goes in before the next chunk. The sequence restarts (brain reset, prompt again) at a chunk after 15 s
// whose last chunk had no text, and always after 30 s. The fork ships this template for LFM3-Audio only, not Omni.
// The ears are causal, so the whole clip is encoded once here and fed 12 rows at a time (the same rows a stream gives).
//   imi-hear-live MODEL.gguf MMPROJ.gguf CLIP.f32 [threads] [OUT.txt]
#include "ears.h"
#include "seoul.h"
#include "words.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: imi-hear-live MODEL.gguf MMPROJ.gguf CLIP.f32 [threads] [OUT.txt]\n"); return 1; }
    const int threads = argc > 4 ? atoi(argv[4]) : 4;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    imi_team team;
    team.start(threads);
    imi_seoul brain;
    if (!brain.load(argv[1], threads, 4096, &team)) return 1;
    imi_words words;
    if (!words.load(brain.f)) return 1;
    imi_ears ears;
    if (!ears.load(argv[2], threads, &team)) return 1;
    auto one = [&](const char * piece) {
        const int32_t id = words.find(piece);
        if (id < 0) { fprintf(stderr, "imi-hear-live: no word %s\n", piece); exit(1); }
        return id;
    };
    const int32_t text_end = one("<|text_end|>"), mixed_end = one("<|mixed_end|>");
    std::vector<char> banned(brain.n_vocab, 0);
    for (const char * p : { "<|mixed_end|>", "<|mixed_start|>", "<|audio_start|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>" }) banned[one(p)] = 1;
    const std::vector<int32_t> prompt = words.encode_prompt("<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n");

    std::vector<float> pcm;
    {
        FILE * f = fopen(argv[3], "rb");
        if (!f) { fprintf(stderr, "no clip %s\n", argv[3]); return 1; }
        float s; while (fread(&s, 4, 1, f) == 1) pcm.push_back(s);
        fclose(f);
    }
    const double sound_s = pcm.size() / 16000.0;
    auto t0 = clk::now();
    std::vector<float> emb;
    const int P = ears.encode(pcm.data(), (int) pcm.size(), emb);
    const double t_ears = secs(t0);

    const int chunk = 12;
    const int soft_rows = (int) std::lround(15.0 * 16000 / 1280), max_rows = (int) std::lround(30.0 * 16000 / 1280);
    std::vector<int32_t> closing, all;
    std::string text;
    int seq_rows = 0, restarts = 0;
    bool started = false, last_had_text = true;
    t0 = clk::now();
    for (int r0 = 0; r0 < P; r0 += chunk) {
        const int n = std::min(chunk, P - r0);
        const bool final = r0 + n >= P;
        if (!started || seq_rows >= max_rows || (seq_rows >= soft_rows && !last_had_text)) {
            if (started) restarts++;
            brain.reset();
            closing = prompt;
            seq_rows = 0;
        }
        const auto tc = clk::now();
        if (!closing.empty()) brain.step_batch(closing.data(), nullptr, (int) closing.size(), false);
        closing.clear();
        if (final) {
            brain.step_batch(nullptr, emb.data() + (size_t) r0 * brain.n_embd, n, false);
            brain.step(mixed_end, nullptr, true);
        } else {
            brain.step_batch(nullptr, emb.data() + (size_t) r0 * brain.n_embd, n, true);
        }
        started = true;
        seq_rows += n;
        std::vector<int32_t> got;
        const int cap = final ? 96 : 48;
        for (;;) {
            int best = -1;
            for (int v = 0; v < brain.n_vocab; v++) if (!banned[v] && (best < 0 || brain.logits[v] > brain.logits[best])) best = v;
            if (best == text_end) break;
            if ((int) got.size() + 1 >= cap) { closing.push_back(best); got.push_back(best); break; }   // a runaway chunk
            got.push_back(best);
            brain.step(best, nullptr, true);
        }
        closing.push_back(text_end);
        last_had_text = !got.empty();
        const std::string piece = words.text(got.data(), (int) got.size());
        text += piece;
        all.insert(all.end(), got.begin(), got.end());
        fprintf(stderr, "%6.2f-%6.2f s  %3d words %5.0f ms  [%s]\n", r0 * 0.08, (r0 + n) * 0.08, (int) got.size(), 1e3 * secs(tc), piece.c_str());
    }
    const double t_live = secs(t0);
    printf("%s\n", text.c_str());
    fprintf(stderr, "imi-hear-live: %.2f s of sound, %d rows, %zu words, %d restarts; ears %.0f ms, live %.0f ms -> %.3f CPU-s per sound second\n",
            sound_s, P, all.size(), restarts, 1e3 * t_ears, 1e3 * t_live, (t_ears + t_live) / sound_s);
    if (argc > 5) { FILE * f = fopen(argv[5], "w"); fputs(text.c_str(), f); fclose(f); }
    return 0;
}
