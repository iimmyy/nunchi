// imi hears: our ears into our brain, then the Korean words one at a time, always the top-scoring word (the hearing
// setting, temperature 0), until <|im_end|>.
//   imi-hear MODEL.gguf MMPROJ.gguf CLIP.f32 [threads] [OUT.txt] [OUT.ids] [LANGUAGE]
// The prompt is the fork's own, word for word (its /apply-template and /tokenize on the same model file):
//   <|startoftext|><|im_start|>system\nPerform ASR in Korean.<|im_end|>\n<|im_start|>user\n<|vision_start|>
//   (the sound positions) <|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>
// LANGUAGE (default Korean) changes only the system line, e.g. English to check what our voice said.
#include "ears.h"
#include "seoul.h"
#include "words.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// plain-text words as the fork's tokenizer splits them; the control words are checked against the word list at start
static const int32_t PRE[]  = { 97357, 97362, 20460, 207, 89141, 10040, 58, 296, 13853, 22, 97363, 207, 97362, 5303, 207, 97472 };
static const int32_t POST[] = { 97473, 97363, 207, 97362, 53502, 207, 97364, 97365 };

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: imi-hear MODEL.gguf MMPROJ.gguf CLIP.f32 [threads] [OUT.txt] [OUT.ids] [LANGUAGE]\n"); return 1; }
    const int threads = argc > 4 ? atoi(argv[4]) : 4;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    imi_team team;
    team.start(threads);
    imi_seoul brain;
    if (!brain.load(argv[1], threads, 4096, &team)) return 1;
    imi_words words;
    if (!words.load(brain.f)) return 1;
    const struct { int32_t id; const char * piece; } specials[] = {
        { 97357, "<|startoftext|>" }, { 97362, "<|im_start|>" }, { 97363, "<|im_end|>" }, { 97472, "<|vision_start|>" },
        { 97473, "<|vision_end|>" }, { 97364, "<think>" }, { 97365, "</think>" } };
    for (auto & s : specials)
        if (words.bytes.size() <= (size_t) s.id || words.bytes[s.id] != s.piece) { fprintf(stderr, "imi-hear: word %d is not %s in this model\n", s.id, s.piece); return 1; }
    imi_ears ears;
    if (!ears.load(argv[2], threads, &team)) return 1;
    if (ears.n_out != brain.n_embd) { fprintf(stderr, "imi-hear: ears give %d wide, the brain reads %d\n", ears.n_out, brain.n_embd); return 1; }

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
    const double t_ears = secs(t0); t0 = clk::now();
    std::vector<int32_t> pre(PRE, PRE + sizeof(PRE) / 4);
    if (argc > 7)
        pre = words.encode_prompt(std::string("<|startoftext|><|im_start|>system\nPerform ASR in ") + argv[7] + ".<|im_end|>\n<|im_start|>user\n<|vision_start|>");
    brain.reset();
    brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
    brain.step_batch(nullptr, emb.data(), P, false);
    brain.step_batch(POST, nullptr, (int) (sizeof(POST) / 4), true);
    const double t_prompt = secs(t0); t0 = clk::now();
    const int n_prompt = (int) pre.size() + P + (int) (sizeof(POST) / 4);

    std::vector<int32_t> ids;
    for (int i = 0; i < 512; i++) {
        int best = 0;
        for (int v = 1; v < brain.n_vocab; v++) if (brain.logits[v] > brain.logits[best]) best = v;
        if (best == words.im_end) break;
        ids.push_back(best);
        brain.step(best, nullptr, true);
    }
    const double t_words = secs(t0);
    const std::string text = words.text(ids.data(), (int) ids.size());
    printf("%s\n", text.c_str());
    fprintf(stderr, "imi-hear: %.2f s of sound -> ears %.0f ms (%d positions), prompt %d positions %.0f ms (%.1f /s), %zu words %.0f ms (%.1f ms each)\n",
            sound_s, 1e3 * t_ears, P, n_prompt, 1e3 * t_prompt, n_prompt / t_prompt, ids.size(), 1e3 * t_words, ids.empty() ? 0.0 : 1e3 * t_words / ids.size());
    fprintf(stderr, "imi-hear: %.3f CPU-s per sound second (ears %.3f, prompt %.3f, words %.3f)\n",
            (t_ears + t_prompt + t_words) / sound_s, t_ears / sound_s, t_prompt / sound_s, t_words / sound_s);
    if (argc > 5) { FILE * f = fopen(argv[5], "w"); fputs(text.c_str(), f); fclose(f); }
    if (argc > 6) { FILE * f = fopen(argv[6], "wb"); fwrite(ids.data(), 4, ids.size(), f); fclose(f); }
    return 0;
}
