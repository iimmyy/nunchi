// imi-speak: Seoul speaks, end to end on imi's own engine (brain, voice generator, codec on one thread team).
//   imi-speak MODEL.gguf MMPROJ.gguf PROMPT.ids|ENGLISH.txt OUT.wav [threads] [seed]
// PROMPT.ids: the word IDs of the "tts" chat template's prompt, as llama-tokenize --ids prints them.
// ENGLISH.txt: the English itself; the whole text in one "tts" prompt, or with IMI_SENTENCES set, one sentence at a
// time in one voice conversation (imi_talker::begin_sentences). The frame loop itself is in talk.cpp.
#include "talk.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: imi-speak MODEL.gguf MMPROJ.gguf PROMPT.ids|ENGLISH.txt OUT.wav [threads] [seed]\n"); return 1; }
    const int nt = argc > 5 ? atoi(argv[5]) : 4;
    const unsigned seed = argc > 6 ? (unsigned) atoi(argv[6]) : 11;
    const std::string in_path = argv[3];
    const bool english = in_path.size() > 4 && in_path.compare(in_path.size() - 4, 4, ".txt") == 0;
    const bool by_sentence = english && getenv("IMI_SENTENCES") != nullptr;
    std::string text;
    {
        std::ifstream in(in_path);
        if (!in) { fprintf(stderr, "imi-speak: no %s\n", argv[3]); return 1; }
        std::stringstream ss; ss << in.rdbuf(); text = ss.str();
    }
    const auto t_load = clk::now();
    imi_team team; team.start(nt);
    imi_seoul brain; imi_voice voice; imi_codec codec; imi_words words;
    if (!brain.load(argv[1], nt, 4096, &team) || !voice.load(argv[2], nt, &team) || !codec.load(argv[2], nt, &team) ||
        !words.load(brain.f)) return 1;
    std::vector<int32_t> prompt;
    if (english) prompt = words.encode_prompt(imi_tts_prompt(text));
    else {
        for (char & ch : text) if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
        std::stringstream ns(text); int v; while (ns >> v) prompt.push_back(v);
    }
    fprintf(stderr, "imi-speak: loaded in %.1f s, %s\n", ms_since(t_load) / 1e3, by_sentence ? "sentence by sentence" : "one prompt");
    if (by_sentence) for (const std::string & s : imi_sentences(text)) fprintf(stderr, "  [%s]\n", s.c_str());

    const auto t_all = clk::now();
    imi_talker tk(brain, voice, codec);
    const bool keep = !(getenv("IMI_SENTENCES") && std::string(getenv("IMI_SENTENCES")) == "fresh");   // IMI_SENTENCES=fresh
    if (by_sentence) tk.begin_sentences(words, text, seed, 2000, keep); else tk.begin(prompt, seed);
    std::vector<float> pcm, fb(voice.n_in);
    while (tk.next(pcm, fb.data())) {
        const auto t0 = clk::now();
        brain.step(-1, fb.data(), false);
        tk.st.brain_ms += ms_since(t0);
    }
    const imi_talk_stats & st = tk.st;
    const double audio_s = pcm.size() / 24000.0, speak_ms = ms_since(t_all) - st.prompt_ms;
    imi_write_wav(argv[4], pcm, 24000);
    printf("imi-speak: %d frames, %.2f s of speech in %.2f s (%.2fx real time); prompt %zu words in %.2f s\n",
           st.frames, audio_s, speak_ms / 1e3, audio_s / (speak_ms / 1e3), prompt.size(), st.prompt_ms / 1e3);
    const int fr = st.frames > 0 ? st.frames : 1;
    printf("stages per frame: generator %.1f ms, codec %.1f ms, brain %.1f ms\n", st.gen_ms / fr, st.codec_ms / fr, st.brain_ms / fr);
    if (by_sentence) {
        printf("sentences %d, capped %d, frames each:", st.sentences, st.capped);
        for (int f : st.sent_frames) printf(" %d", f);
        printf("\n");
    }
    return 0;
}
