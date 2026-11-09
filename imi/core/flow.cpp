// imi-flow: the sermon pipeline as it runs live: while window N is spoken, window N+1 is heard and translated, and
// each hearing or translation word step rides on a voice frame's brain step (step2: one weight read, two
// conversations). Outputs are the same as running every step alone (imi-nunchi); only the time changes.
//   imi-flow MODEL.gguf MMPROJ.gguf OUT_DIR threads RIDE W1.f32 [W2.f32 ...]
// RIDE: 1 = share the brain steps, 0 = the same order with every step alone (to measure what riding saves).
// Writes OUT_DIR/<name>.ko.txt, .en.txt, .wav per window and OUT_DIR/flow.json.
#include "ears.h"
#include "seoul.h"
#include "talk.h"
#include "words.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static std::string chat_head(const std::string & sys) { return "<|startoftext|><|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n"; }
static const char * CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think></think>";

struct text_job {
    std::string name;
    double sound_s = 0;
    int phase = 0;                // 1 hearing words, 2 translation words, 3 done
    int32_t pending = -1;         // the next word to step
    std::vector<int32_t> ko, en;
    std::string ko_text, en_text;
};

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: imi-flow MODEL.gguf MMPROJ.gguf OUT_DIR threads RIDE W1.f32 [W2.f32 ...]\n"); return 1; }
    const std::string dir = argv[3];
    const int threads = atoi(argv[4]);
    const bool ride = atoi(argv[5]) != 0;
    const bool by_sentence = getenv("IMI_SENTENCES") != nullptr;   // research: the voice one sentence at a time
    std::vector<std::string> clips(argv + 6, argv + argc);
    imi_team team;
    team.start(threads);
    imi_seoul brain;
    imi_words words;
    imi_ears ears;
    imi_voice voice;
    imi_codec codec;
    if (!brain.load(argv[1], threads, 4096, &team) || !words.load(brain.f) || !ears.load(argv[2], threads, &team) ||
        !voice.load(argv[2], threads, &team) || !codec.load(argv[2], threads, &team)) return 1;
    imi_seoul_state T = brain.new_state();   // the text conversation (hearing, then translation); the voice keeps the brain's own

    int n_ride = 0, n_text_alone = 0, n_voice_alone = 0;
    double sound_total = 0;
    auto in_T = [&](auto fn) { brain.swap_state(T); fn(); brain.swap_state(T); };
    // after a text step (or a prompt): the next word, or the end of this part
    std::function<void(text_job &)> pick = [&](text_job & j) {
        std::vector<int32_t> & got = j.phase == 1 ? j.ko : j.en;
        const size_t cap = j.phase == 1 ? 512 : 600;
        int best = 0;
        for (int v = 1; v < brain.n_vocab; v++) if (T.logits[v] > T.logits[best]) best = v;
        if (best != words.im_end && got.size() < cap) { got.push_back(best); j.pending = best; return; }
        j.pending = -1;
        if (j.phase == 1) {
            j.ko_text = words.text(j.ko.data(), (int) j.ko.size());
            while (!j.ko_text.empty() && (j.ko_text.back() == '\n' || j.ko_text.back() == ' ')) j.ko_text.pop_back();
            const std::vector<int32_t> tp = words.encode_prompt(chat_head("Please translate the following Korean into English text.") + j.ko_text + CHAT_TAIL);
            in_T([&] { brain.reset(); brain.step_batch(tp.data(), nullptr, (int) tp.size(), true); });
            j.phase = 2;
            pick(j);
        } else {
            j.en_text = words.text(j.en.data(), (int) j.en.size());
            j.phase = 3;
        }
    };
    auto start = [&](text_job & j, const std::string & path) {
        std::vector<float> pcm;
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "no clip %s\n", path.c_str()); exit(1); }
        float s; while (fread(&s, 4, 1, f) == 1) pcm.push_back(s);
        fclose(f);
        j.sound_s = pcm.size() / 16000.0;
        const size_t sl = path.find_last_of('/'), dot = path.find_last_of('.');
        j.name = path.substr(sl == std::string::npos ? 0 : sl + 1, dot - (sl == std::string::npos ? 0 : sl + 1));
        std::vector<float> emb;
        const int P = ears.encode(pcm.data(), (int) pcm.size(), emb);
        const std::vector<int32_t> pre = words.encode_prompt(chat_head("Perform ASR in Korean.") + "<|vision_start|>"),
                                   post = words.encode_prompt(std::string("<|vision_end|>") + CHAT_TAIL);
        in_T([&] {
            brain.reset();
            brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
            brain.step_batch(nullptr, emb.data(), P, false);
            brain.step_batch(post.data(), nullptr, (int) post.size(), true);
        });
        j.phase = 1;
        pick(j);
    };
    auto text_alone = [&](text_job & j) { const int32_t w = j.pending; in_T([&] { brain.step(w, nullptr, true); }); n_text_alone++; pick(j); };

    const auto t_all = clk::now();
    std::vector<text_job> jobs(clips.size());
    start(jobs[0], clips[0]);
    while (jobs[0].phase != 3) text_alone(jobs[0]);    // the first window has no speech to ride on
    std::vector<float> fb(voice.n_in);
    for (size_t i = 0; i < clips.size(); i++) {
        sound_total += jobs[i].sound_s;
        imi_talker tk(brain, voice, codec);
        if (by_sentence) tk.begin_sentences(words, jobs[i].en_text, 11, 2000, std::string(getenv("IMI_SENTENCES")) != "fresh");
        else tk.begin(words.encode_prompt(imi_tts_prompt(jobs[i].en_text)), 11);
        text_job * nx = i + 1 < clips.size() ? &jobs[i + 1] : nullptr;
        if (nx) start(*nx, clips[i + 1]);
        std::vector<float> pcm;
        bool speaking = true;
        while (speaking || (nx && nx->phase != 3)) {
            if (speaking) speaking = tk.next(pcm, fb.data());
            const bool text = nx && nx->phase != 3;
            if (speaking && text && ride) {
                brain.step2(-1, fb.data(), false, T, nx->pending, nullptr, true);
                n_ride++;
                pick(*nx);
                continue;
            }
            if (speaking) { brain.step(-1, fb.data(), false); n_voice_alone++; }
            if (text) text_alone(*nx);
        }
        imi_write_wav(dir + "/" + jobs[i].name + ".wav", pcm, 24000);
        { std::ofstream f(dir + "/" + jobs[i].name + ".ko.txt"); f << jobs[i].ko_text; }
        { std::ofstream f(dir + "/" + jobs[i].name + ".en.txt"); f << jobs[i].en_text; }
        fprintf(stderr, "imi-flow: %s done (%.2f s of sound, %d frames, %d sentences, %d capped), %.1f s in\n", jobs[i].name.c_str(),
                jobs[i].sound_s, tk.st.frames, tk.st.sentences, tk.st.capped, secs(t_all));
    }
    const double total = secs(t_all);
    fprintf(stderr, "imi-flow: %zu windows, %.1f s of sound in %.1f s -> %.3f CPU-s per sound second (ride %d: %d shared steps, %d text alone, %d voice alone)\n",
            clips.size(), sound_total, total, total / sound_total, ride ? 1 : 0, n_ride, n_text_alone, n_voice_alone);
    std::ofstream f(dir + "/flow.json");
    f << "{\"windows\": " << clips.size() << ", \"sound_s\": " << sound_total << ", \"wall_s\": " << total << ", \"per_sound_s\": " << total / sound_total
      << ", \"ride\": " << (ride ? 1 : 0) << ", \"shared_steps\": " << n_ride << ", \"text_alone\": " << n_text_alone << ", \"voice_alone\": " << n_voice_alone << "}\n";
    return 0;
}
