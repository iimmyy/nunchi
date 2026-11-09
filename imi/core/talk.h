// imi.cpp: Seoul speaks one utterance: the brain reads the "tts" prompt, then frame by frame
//   brain hidden state -> the generator picks 12 codes in masked passes (6 + 6, random order, temperature 0.4,
//   repeat penalty 1.2 over the last 64 frames per codebook) -> the codec makes 1920 samples at 24 kHz ->
//   the 12 codes' feedback row sum is the brain's next input. Any end-of-audio code stops.
#pragma once
#include "seoul.h"
#include "voice.h"
#include "codec.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct imi_talk_stats {
    int frames = 0;
    double prompt_ms = 0, gen_ms = 0, codec_ms = 0, brain_ms = 0, total_ms = 0;
};

// the prompt for English speech in the male voice (the "tts" chat template)
std::string imi_tts_prompt(const std::string & english);

// one utterance, a frame at a time, so the caller owns the brain step after each frame (and can share it with
// another conversation's step, see imi_seoul::step2):
//   begin(prompt); while (next(pcm, fb)) brain.step(-1, fb, false);
struct imi_talker {
    imi_seoul & brain; imi_voice & voice; imi_codec & codec;
    imi_talk_stats st;
    imi_talker(imi_seoul & b, imi_voice & v, imi_codec & c) : brain(b), voice(v), codec(c) {}
    // resets the brain's own conversation and the codec, reads the prompt
    void begin(const std::vector<int32_t> & prompt, unsigned seed, int max_frames = 2000);
    // the next frame's codes from brain.hidden; true: its samples were appended and fb (voice.n_in wide) is the brain's
    // next input; false: speech is over (the codec has been flushed)
    bool next(std::vector<float> & pcm, float * fb);
private:
    std::mt19937 rng;
    int max_frames = 2000;
    std::vector<int32_t> hist;
    std::vector<float> scores;
    std::vector<double> p;
    std::vector<char> seen;
};

// resets the brain, reads `prompt`, speaks; appends 24 kHz samples to pcm
imi_talk_stats imi_speak(imi_seoul & brain, imi_voice & voice, imi_codec & codec, const std::vector<int32_t> & prompt,
                         unsigned seed, std::vector<float> & pcm, int max_frames = 2000);

void imi_write_wav(const std::string & path, const std::vector<float> & pcm, int sr);
