// imi-voice-hears: research, can the English voice carry the speaker's feeling if it hears him first? Each case is
// spoken twice with the same seed:
//   A  the plain "tts" prompt: the English text alone (what the app does)
//   B  the same prompt with the sentence's own Korean audio (the ears' rows) in the user turn before the English, so
//      the voice's memory holds how he said it
//   imi-voice-hears MODEL.gguf MMPROJ.gguf CASES.tsv OUT_DIR [threads] [seeds=7]
// CASES.tsv: name<TAB>clip.f32 (16 kHz mono float, his Korean sentence)<TAB>english, one case per line.
// OUT_DIR/name-A-sSEED.wav and name-B-sSEED.wav, 24 kHz; seeds "7,8" makes both for each seed.
#include "ears.h"
#include "talk.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

static std::vector<float> read_f32(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> x(b.size() / 4);
    memcpy(x.data(), b.data(), x.size() * 4);
    return x;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: imi-voice-hears MODEL.gguf MMPROJ.gguf CASES.tsv OUT_DIR [threads] [seeds=7]\n"); return 1; }
    const int nt = argc > 5 ? atoi(argv[5]) : 4;
    std::vector<unsigned> seeds;
    { std::stringstream ss(argc > 6 ? argv[6] : "7"); std::string s; while (std::getline(ss, s, ',')) seeds.push_back((unsigned) atoi(s.c_str())); }
    const std::string out = argv[4];
    imi_team team; team.start(nt);
    imi_seoul brain; imi_voice voice; imi_codec codec; imi_words words; imi_ears ears;
    if (!brain.load(argv[1], nt, 4096, &team) || !words.load(brain.f) || !ears.load(argv[2], nt, &team) ||
        !voice.load(argv[2], nt, &team) || !codec.load(argv[2], nt, &team)) return 1;
    const std::string sys = "<|startoftext|><|im_start|>system\nPerform TTS in American English using a male voice.<|im_end|>\n<|im_start|>user\n";
    const std::vector<int32_t> pre = words.encode_prompt(sys + "<|vision_start|>");
    imi_talker tk(brain, voice, codec);
    std::ifstream cases(argv[3]);
    for (std::string line; std::getline(cases, line);) {
        std::stringstream ls(line);
        std::string name, clip, en;
        if (!std::getline(ls, name, '\t') || !std::getline(ls, clip, '\t') || !std::getline(ls, en)) continue;
        const std::vector<float> pcm16 = read_f32(clip);
        std::vector<float> emb;
        const int P = ears.encode(pcm16.data(), (int) pcm16.size(), emb);
        const std::vector<int32_t> post = words.encode_prompt("<|vision_end|>" + en + "<|im_end|>\n<|im_start|>assistant\n<think></think><|audio_start|>");
        for (unsigned seed : seeds) {
            for (char v : { 'A', 'B' }) {
                const auto t0 = clk::now();
                if (v == 'A') tk.begin_sentences(words, en, seed, 2000, true, true);
                else {
                    brain.reset();
                    brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
                    brain.step_batch(nullptr, emb.data(), P, false);
                    brain.step_batch(post.data(), nullptr, (int) post.size(), false);
                    tk.begin_sentences(words, en, seed, 2000, true, false);
                }
                std::vector<float> pcm, fb(voice.n_in);
                while (tk.next(pcm, fb.data())) brain.step(-1, fb.data(), false);
                const std::string path = out + "/" + name + "-" + v + "-s" + std::to_string(seed) + ".wav";
                imi_write_wav(path, pcm, 24000);
                printf("%s %c seed %u: Korean %.2f s (%d rows), English speech %.2f s, %d frames, %.1f s to make  [%s]\n",
                       name.c_str(), v, seed, pcm16.size() / 16000.0, P, pcm.size() / 24000.0, tk.st.frames, ms_since(t0) / 1e3, en.c_str());
                fflush(stdout);
            }
        }
    }
    return 0;
}
