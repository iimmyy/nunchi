// imi.cpp: Seoul speaks one utterance: the brain reads the "tts" prompt, then frame by frame
//   brain hidden state -> the generator picks 12 codes in masked passes (6 + 6, random order, temperature 0.4,
//   repeat penalty 1.2 over the last 64 frames per codebook) -> the codec makes 1920 samples at 24 kHz ->
//   the 12 codes' feedback row sum is the brain's next input. Any end-of-audio code stops.
#pragma once
#include "seoul.h"
#include "voice.h"
#include "codec.h"

#include <cstdint>
#include <string>
#include <vector>

struct imi_talk_stats {
    int frames = 0;
    double prompt_ms = 0, gen_ms = 0, codec_ms = 0, brain_ms = 0, total_ms = 0;
};

// the prompt for English speech in the male voice (the "tts" chat template)
std::string imi_tts_prompt(const std::string & english);

// resets the brain, reads `prompt`, speaks; appends 24 kHz samples to pcm
imi_talk_stats imi_speak(imi_seoul & brain, imi_voice & voice, imi_codec & codec, const std::vector<int32_t> & prompt,
                         unsigned seed, std::vector<float> & pcm, int max_frames = 2000);

void imi_write_wav(const std::string & path, const std::vector<float> & pcm, int sr);
