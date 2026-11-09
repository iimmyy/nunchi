#include "talk.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

std::string imi_tts_prompt(const std::string & english) {
    return "<|startoftext|><|im_start|>system\nPerform TTS in American English using a male voice.<|im_end|>\n<|im_start|>user\n" + english +
           "<|im_end|>\n<|im_start|>assistant\n<think></think><|audio_start|>";
}

void imi_talker::begin(const std::vector<int32_t> & prompt, unsigned seed, int max_fr) {
    st = imi_talk_stats();
    max_frames = max_fr;
    brain.reset();
    codec.reset();
    const auto t0 = clk::now();
    brain.step_batch(prompt.data(), nullptr, (int) prompt.size(), false);
    st.prompt_ms = ms_since(t0);
    rng.seed(seed);
    hist.clear();
    p.assign(voice.n_vocab, 0.0);
    seen.assign(voice.n_vocab, 0);
    words = nullptr;
    sents.clear();
    si = 0;
    sent_n = sent_cap = 0;
}

std::vector<std::string> imi_sentences(const std::string & t) {
    std::vector<std::string> out;
    std::string cur;
    auto push = [&] {
        const size_t a = cur.find_first_not_of(" \t\n"), b = cur.find_last_not_of(" \t\n");
        if (a != std::string::npos) out.push_back(cur.substr(a, b - a + 1));
        cur.clear();
    };
    for (size_t i = 0; i < t.size(); i++) {
        cur += t[i];
        if (t[i] != '.' && t[i] != '?' && t[i] != '!') continue;
        for (;;) {   // closing quotes and brackets stay with their sentence
            if (i + 1 < t.size() && t[i + 1] != 0 && strchr("\"')]", t[i + 1])) { cur += t[++i]; continue; }
            if (i + 3 < t.size() && t.compare(i + 1, 2, "\xE2\x80") == 0 && (t[i + 3] == '\x9D' || t[i + 3] == '\x99')) {
                cur += t.substr(i + 1, 3); i += 3; continue;
            }
            break;
        }
        if (i + 1 >= t.size() || t[i + 1] == ' ' || t[i + 1] == '\n' || t[i + 1] == '\t') push();
    }
    push();
    return out;
}

static int sentence_cap(const std::string & s) {
    int n = 0;
    bool in = false;
    for (char c : s) { const bool w = c != ' ' && c != '\n' && c != '\t'; if (w && !in) n++; in = w; }
    return 25 + 10 * n;
}

void imi_talker::begin_sentences(const imi_words & w, const std::string & english, unsigned seed, int max_fr, bool keep) {
    std::vector<std::string> s = imi_sentences(english);
    if (s.empty()) s.push_back(english);
    begin(w.encode_prompt(imi_tts_prompt(s[0])), seed, max_fr);
    words = &w;
    sents = std::move(s);
    keep_memory = keep;
    sent_cap = sentence_cap(sents[0]);
    st.sentences = 1;
}

void imi_talker::next_sentence() {
    const auto t0 = clk::now();
    // keeping memory, the answer ends the way the fork continues a conversation: the end-of-audio frame goes back in,
    // then the chat template's turn change brings the next sentence as a new user turn. Fresh: a new conversation.
    si++;
    std::vector<int32_t> turn;
    if (keep_memory) {
        const std::vector<int32_t> e(voice.n_cb, voice.n_vocab - 1);
        efb.resize(voice.n_in);
        voice.feedback(e.data(), efb.data());
        brain.step(-1, efb.data(), false);
        turn = words->encode_prompt("<|im_end|>\n<|im_start|>user\n" + sents[si] + "<|im_end|>\n<|im_start|>assistant\n<think></think><|audio_start|>");
    } else {
        brain.reset();
        turn = words->encode_prompt(imi_tts_prompt(sents[si]));
    }
    brain.step_batch(turn.data(), nullptr, (int) turn.size(), false);
    hist.clear();   // the repeat penalty is per answer, as the fork's is per response
    sent_n = 0;
    sent_cap = sentence_cap(sents[si]);
    st.sentences++;
    st.prompt_ms += ms_since(t0);
}

bool imi_talker::next(std::vector<float> & pcm, float * fb) {
    const int n_cb = voice.n_cb, V = voice.n_vocab, eos = V - 1;
    static const int schedule[2] = { 6, 6 };
    // research switches (unset = the model file's own settings): IMI_TEMP, IMI_REP, IMI_REP_WIN
    static const float temp = getenv("IMI_TEMP") ? (float) atof(getenv("IMI_TEMP")) : 0.4f;
    static const float rep = getenv("IMI_REP") ? (float) atof(getenv("IMI_REP")) : 1.2f;
    static const int rep_win = getenv("IMI_REP_WIN") ? atoi(getenv("IMI_REP_WIN")) : 64;
    for (;;) {
        auto t0 = clk::now();
        std::vector<int> order(n_cb);
        for (int i = 0; i < n_cb; i++) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<int32_t> codes(n_cb, -1);
        voice.begin_frame(brain.hidden.data());
        int k = 0;
        double stay = 1.0;   // the chance that no codebook picks end-of-audio this frame
        static const bool eos_log = getenv("IMI_EOS_LOG") != nullptr;
        for (int n_pick : schedule) {
            std::vector<int> need(order.begin() + k, order.begin() + k + n_pick);
            voice.pass(codes.data(), need, scores);
            for (int j = 0; j < n_pick; j++, k++) {
                const int cb = need[j];
                float * l = scores.data() + (size_t) j * V;
                // repeat penalty over the codes this codebook took in the last rep_win frames
                const size_t nh = hist.size() / n_cb, nw = std::min(nh, (size_t) rep_win);
                std::fill(seen.begin(), seen.end(), 0);
                for (size_t f = nh - nw; f < nh; f++) {
                    const int c = hist[f * n_cb + cb];
                    if (c >= 0 && c < V && !seen[c]) { seen[c] = 1; l[c] = l[c] > 0 ? l[c] / rep : l[c] * rep; }
                }
                const float lmax = *std::max_element(l, l + V);
                double sum = 0;
                for (int v = 0; v < V; v++) { p[v] = std::exp((double) (l[v] - lmax) / temp); sum += p[v]; }
                stay *= 1.0 - p[eos] / sum;
                double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
                int pick = V - 1;
                for (int v = 0; v < V; v++) { r -= p[v]; if (r <= 0) { pick = v; break; } }
                codes[cb] = pick;
            }
        }
        st.gen_ms += ms_since(t0);
        if (eos_log) fprintf(stderr, "imi-eos %d %.6f\n", st.frames, 1.0 - stay);   // research: the chance speech ended here
        const bool end = std::find(codes.begin(), codes.end(), eos) != codes.end();
        const bool capped = !end && !sents.empty() && sent_n >= sent_cap;
        if (end || capped || st.frames >= max_frames) {
            if (capped) st.capped++;
            if (!sents.empty()) st.sent_frames.push_back(sent_n);
            if (si + 1 < sents.size() && st.frames < max_frames) { next_sentence(); continue; }   // the next one's first frame
            codec.flush(pcm);
            return false;
        }
        hist.insert(hist.end(), codes.begin(), codes.end());
        t0 = clk::now();
        codec.frame(codes.data(), pcm);
        st.codec_ms += ms_since(t0);
        voice.feedback(codes.data(), fb);
        st.frames++;
        sent_n++;
        return true;
    }
}

imi_talk_stats imi_speak(imi_seoul & brain, imi_voice & voice, imi_codec & codec, const std::vector<int32_t> & prompt,
                         unsigned seed, std::vector<float> & pcm, int max_frames) {
    const auto t_all = clk::now();
    imi_talker tk(brain, voice, codec);
    tk.begin(prompt, seed, max_frames);
    std::vector<float> fb(voice.n_in);
    while (tk.next(pcm, fb.data())) {
        const auto t0 = clk::now();
        brain.step(-1, fb.data(), false);
        tk.st.brain_ms += ms_since(t0);
    }
    tk.st.total_ms = ms_since(t_all);
    return tk.st;
}

void imi_write_wav(const std::string & path, const std::vector<float> & pcm, int sr) {
    FILE * o = fopen(path.c_str(), "wb");
    if (!o) return;
    const uint32_t nb = (uint32_t) pcm.size() * 2, cs = 36 + nb, br = sr * 2, s16 = 16, srr = sr;
    const uint16_t one = 1, ba = 2, bits = 16;
    fwrite("RIFF", 1, 4, o); fwrite(&cs, 4, 1, o); fwrite("WAVEfmt ", 1, 8, o); fwrite(&s16, 4, 1, o); fwrite(&one, 2, 1, o); fwrite(&one, 2, 1, o);
    fwrite(&srr, 4, 1, o); fwrite(&br, 4, 1, o); fwrite(&ba, 2, 1, o); fwrite(&bits, 2, 1, o); fwrite("data", 1, 4, o); fwrite(&nb, 4, 1, o);
    for (float s : pcm) { const int16_t v = (int16_t) lrintf(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f); fwrite(&v, 2, 1, o); }
    fclose(o);
}
