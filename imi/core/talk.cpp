#include "talk.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
}

bool imi_talker::next(std::vector<float> & pcm, float * fb) {
    const int n_cb = voice.n_cb, V = voice.n_vocab, eos = V - 1;
    static const int schedule[2] = { 6, 6 };
    // research switches (unset = the model file's own settings): IMI_TEMP, IMI_REP, IMI_REP_WIN
    static const float temp = getenv("IMI_TEMP") ? (float) atof(getenv("IMI_TEMP")) : 0.4f;
    static const float rep = getenv("IMI_REP") ? (float) atof(getenv("IMI_REP")) : 1.2f;
    static const int rep_win = getenv("IMI_REP_WIN") ? atoi(getenv("IMI_REP_WIN")) : 64;
    auto t0 = clk::now();
    std::vector<int> order(n_cb);
    for (int i = 0; i < n_cb; i++) order[i] = i;
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
            std::fill(seen.begin(), seen.end(), 0);
            for (size_t f = nh - nw; f < nh; f++) {
                const int c = hist[f * n_cb + cb];
                if (c >= 0 && c < V && !seen[c]) { seen[c] = 1; l[c] = l[c] > 0 ? l[c] / rep : l[c] * rep; }
            }
            const float lmax = *std::max_element(l, l + V);
            double sum = 0;
            for (int v = 0; v < V; v++) { p[v] = std::exp((double) (l[v] - lmax) / temp); sum += p[v]; }
            double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
            int pick = V - 1;
            for (int v = 0; v < V; v++) { r -= p[v]; if (r <= 0) { pick = v; break; } }
            codes[cb] = pick;
        }
    }
    st.gen_ms += ms_since(t0);
    if (std::find(codes.begin(), codes.end(), eos) != codes.end() || st.frames >= max_frames) {
        codec.flush(pcm);
        return false;
    }
    hist.insert(hist.end(), codes.begin(), codes.end());
    t0 = clk::now();
    codec.frame(codes.data(), pcm);
    st.codec_ms += ms_since(t0);
    voice.feedback(codes.data(), fb);
    st.frames++;
    return true;
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
