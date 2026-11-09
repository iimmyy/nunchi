// imi-speak: Seoul speaks, end to end on imi's own engine (brain, voice generator, codec on one thread team).
//   imi-speak MODEL.gguf MMPROJ.gguf PROMPT.ids OUT.wav [threads] [seed]
// PROMPT.ids: the word IDs of the "tts" chat template's prompt, as llama-tokenize --ids prints them. The frame loop
// itself is in talk.cpp (shared with imi-nunchi).
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
    std::vector<float> pcm;
    const imi_talk_stats st = imi_speak(brain, voice, codec, prompt, seed, pcm);
    const double audio_s = pcm.size() / 24000.0, speak_ms = st.total_ms - st.prompt_ms;
    imi_write_wav(argv[4], pcm, 24000);
    printf("imi-speak: %d frames, %.2f s of speech in %.2f s (%.2fx real time); prompt %zu words in %.2f s\n",
           st.frames, audio_s, speak_ms / 1e3, audio_s / (speak_ms / 1e3), prompt.size(), st.prompt_ms / 1e3);
    const int fr = st.frames > 0 ? st.frames : 1;
    printf("stages per frame: generator %.1f ms, codec %.1f ms, brain %.1f ms\n", st.gen_ms / fr, st.codec_ms / fr, st.brain_ms / fr);
    return 0;
}
