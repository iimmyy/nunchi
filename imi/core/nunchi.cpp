// imi-nunchi: the sermon pipeline on our own engine for one window of sound: hear (Korean words), translate
// (English words), speak (English voice). Hearing and translation are greedy (temperature 0, thinking off) with the
// product's prompts; speech as imi-speak (seed 11). One thread team, one brain.
//   imi-nunchi MODEL.gguf MMPROJ.gguf CLIP.f32 [threads] [OUT_PREFIX] [KO.txt]
// MMPROJ: the file with the ears and the 8-bit voice (genq8b). KO.txt: translate this Korean instead of what we heard.
// Writes OUT_PREFIX.ko.txt, .en.txt, .wav, .times.json.
#include "ears.h"
#include "seoul.h"
#include "talk.h"
#include "words.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static const char * HEAR_SYS = "Perform ASR in Korean.";
static const char * TRANSLATE_SYS = "Please translate the following Korean into English text.";
// the chat template's shape with thinking off (the fork's /apply-template output for these messages)
static std::string chat_head(const std::string & sys) { return "<|startoftext|><|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n"; }
static const char * CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think></think>";

struct greedy_out { std::vector<int32_t> ids; double s = 0; };
static greedy_out greedy(imi_seoul & brain, const imi_words & words, int max_new) {
    greedy_out o;
    auto t0 = clk::now();
    for (int i = 0; i < max_new; i++) {
        int best = 0;
        for (int v = 1; v < brain.n_vocab; v++) if (brain.logits[v] > brain.logits[best]) best = v;
        if (best == words.im_end) break;
        o.ids.push_back(best);
        brain.step(best, nullptr, true);
    }
    o.s = secs(t0);
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: imi-nunchi MODEL.gguf MMPROJ.gguf CLIP.f32 [threads] [OUT_PREFIX] [KO.txt]\n"); return 1; }
    const int threads = argc > 4 ? atoi(argv[4]) : 4;
    const std::string out = argc > 5 ? argv[5] : "nunchi";
    imi_team team;
    team.start(threads);
    imi_seoul brain;
    if (!brain.load(argv[1], threads, 4096, &team)) return 1;
    imi_words words;
    if (!words.load(brain.f)) return 1;
    imi_ears ears;
    imi_voice voice;
    imi_codec codec;
    if (!ears.load(argv[2], threads, &team) || !voice.load(argv[2], threads, &team) || !codec.load(argv[2], threads, &team)) return 1;
    std::vector<float> pcm;
    {
        FILE * f = fopen(argv[3], "rb");
        if (!f) { fprintf(stderr, "no clip %s\n", argv[3]); return 1; }
        float s; while (fread(&s, 4, 1, f) == 1) pcm.push_back(s);
        fclose(f);
    }
    const double sound_s = pcm.size() / 16000.0;

    // ---- hear
    auto t0 = clk::now();
    std::vector<float> emb;
    const int P = ears.encode(pcm.data(), (int) pcm.size(), emb);
    const double t_ears = secs(t0); t0 = clk::now();
    const std::vector<int32_t> pre = words.encode_prompt(chat_head(HEAR_SYS) + "<|vision_start|>"), post = words.encode_prompt(std::string("<|vision_end|>") + CHAT_TAIL);
    brain.reset();
    brain.step_batch(pre.data(), nullptr, (int) pre.size(), false);
    brain.step_batch(nullptr, emb.data(), P, false);
    brain.step_batch(post.data(), nullptr, (int) post.size(), true);
    const double t_hear_prompt = secs(t0);
    const int n_hear_prompt = (int) (pre.size() + P + post.size());
    const greedy_out ko = greedy(brain, words, 512);
    std::string ko_text = words.text(ko.ids.data(), (int) ko.ids.size());
    if (argc > 6) { std::ifstream k(argv[6]); std::stringstream ss; ss << k.rdbuf(); ko_text = ss.str(); }
    while (!ko_text.empty() && (ko_text.back() == '\n' || ko_text.back() == ' ')) ko_text.pop_back();

    // ---- translate (the Korean as text, as the two-step product hands it over)
    t0 = clk::now();
    const std::vector<int32_t> tp = words.encode_prompt(chat_head(TRANSLATE_SYS) + ko_text + CHAT_TAIL);
    brain.reset();
    brain.step_batch(tp.data(), nullptr, (int) tp.size(), true);
    const double t_tr_prompt = secs(t0);
    const greedy_out en = greedy(brain, words, 600);
    const std::string en_text = words.text(en.ids.data(), (int) en.ids.size());

    // ---- speak
    const std::vector<int32_t> sp = words.encode_prompt(imi_tts_prompt(en_text));
    std::vector<float> pcm_out;
    const imi_talk_stats st = imi_speak(brain, voice, codec, sp, 11, pcm_out);
    const double t_speak = st.total_ms / 1e3, speech_s = pcm_out.size() / 24000.0;
    imi_write_wav(out + ".wav", pcm_out, 24000);
    { std::ofstream f(out + ".speak.ids"); f << "["; for (size_t i = 0; i < sp.size(); i++) f << (i ? ", " : "") << sp[i]; f << "]\n"; }

    printf("KO: %s\nEN: %s\n", ko_text.c_str(), en_text.c_str());
    const double hear = t_ears + t_hear_prompt + ko.s, tr = t_tr_prompt + en.s, total = hear + tr + t_speak;
    const int fr = st.frames > 0 ? st.frames : 1;
    fprintf(stderr, "imi-nunchi: %.2f s of sound\n  hear: ears %.0f ms, prompt %d in %.0f ms, %zu words in %.0f ms\n"
                    "  translate: prompt %zu in %.0f ms, %zu words in %.0f ms\n"
                    "  speak: prompt %zu in %.0f ms, %d frames (%.2f s of speech) in %.0f ms (per frame: generator %.1f, codec %.1f, brain %.1f ms)\n"
                    "  %.3f CPU-s per sound second (hear %.3f, translate %.3f, speak %.3f)\n",
            sound_s, 1e3 * t_ears, n_hear_prompt, 1e3 * t_hear_prompt, ko.ids.size(), 1e3 * ko.s, tp.size(), 1e3 * t_tr_prompt, en.ids.size(), 1e3 * en.s,
            sp.size(), st.prompt_ms, st.frames, speech_s, st.total_ms - st.prompt_ms, st.gen_ms / fr, st.codec_ms / fr, st.brain_ms / fr,
            total / sound_s, hear / sound_s, tr / sound_s, t_speak / sound_s);
    { std::ofstream f(out + ".ko.txt"); f << ko_text; }
    { std::ofstream f(out + ".en.txt"); f << en_text; }
    {
        std::ofstream f(out + ".times.json");
        f << "{\"sound_s\": " << sound_s << ", \"ears_s\": " << t_ears << ", \"hear_prompt_n\": " << n_hear_prompt << ", \"hear_prompt_s\": " << t_hear_prompt
          << ", \"hear_words\": " << ko.ids.size() << ", \"hear_words_s\": " << ko.s << ", \"translate_prompt_n\": " << tp.size()
          << ", \"translate_prompt_s\": " << t_tr_prompt << ", \"en_words\": " << en.ids.size() << ", \"en_words_s\": " << en.s
          << ", \"speak_prompt_n\": " << sp.size() << ", \"speak_prompt_s\": " << st.prompt_ms / 1e3 << ", \"frames\": " << st.frames
          << ", \"speech_s\": " << speech_s << ", \"speak_s\": " << t_speak << ", \"gen_s\": " << st.gen_ms / 1e3 << ", \"codec_s\": " << st.codec_ms / 1e3
          << ", \"brain_frames_s\": " << st.brain_ms / 1e3 << ", \"per_sound_s\": " << total / sound_s
          << ", \"hear_prompt_ids\": [";
        for (size_t i = 0; i < pre.size(); i++) f << (i ? ", " : "") << pre[i];
        f << ", \"audio\"";
        for (int32_t id : post) f << ", " << id;
        f << "]}\n";
    }
    return 0;
}
