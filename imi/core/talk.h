// imi.cpp: Seoul speaks one utterance: the brain reads the "tts" prompt, then frame by frame
//   brain hidden state -> the generator picks 12 codes in masked passes (6 + 6, random order, temperature 0.4,
//   repeat penalty 1.2 over the last 64 frames per codebook) -> the codec makes 1920 samples at 24 kHz ->
//   the 12 codes' feedback row sum is the brain's next input. Any end-of-audio code stops.
#pragma once
#include "seoul.h"
#include "voice.h"
#include "codec.h"
#include "words.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct imi_talk_stats {
    int frames = 0;
    double prompt_ms = 0, gen_ms = 0, codec_ms = 0, brain_ms = 0, total_ms = 0;
    // sentence by sentence (begin_sentences): sentences spoken, how many hit their cap, frames of each
    int sentences = 0, capped = 0;
    std::vector<int> sent_frames;
};

// the English cut after . ? ! (with any closing quote or bracket) where a space or the end follows
std::vector<std::string> imi_sentences(const std::string & english);

// the prompt for English speech in the male voice (the "tts" chat template): head + english + tail
std::string imi_tts_prompt(const std::string & english);
// its fixed start (system turn and the user turn's opening), the same for every sentence
std::string imi_tts_head();

// one utterance, a frame at a time, so the caller owns the brain step after each frame (and can share it with
// another conversation's step, see imi_seoul::step2):
//   begin(prompt); while (next(pcm, fb)) brain.step(-1, fb, false);
struct imi_talker {
    imi_seoul & brain; imi_voice & voice; imi_codec & codec;
    imi_talk_stats st;
    imi_talker(imi_seoul & b, imi_voice & v, imi_codec & c) : brain(b), voice(v), codec(c) {}
    // resets the brain's own conversation and the codec, reads the prompt
    void begin(const std::vector<int32_t> & prompt, unsigned seed, int max_frames = 2000);
    // the English one sentence at a time in one voice conversation: the first sentence gets the "tts" prompt, each
    // next one comes in as a new user turn after the voice's own audio (the chat template's turn change, as the fork
    // continues a conversation), so the voice keeps its memory of how it was speaking but has only one sentence in
    // front of it. Each sentence stops at its end-of-audio or at 25 + 10 frames per word (about 2x the slowest
    // normal pace), whichever comes first.
    // keep_memory false: each sentence starts a fresh voice conversation with its own "tts" prompt instead (the
    // codec keeps streaming either way)
    // read false: the brain's own memory already holds the first sentence's whole "tts" prompt (read ahead by the
    // caller), so nothing is reset or read here
    void begin_sentences(const imi_words & words, const std::string & english, unsigned seed, int max_frames = 2000,
                         bool keep_memory = true, bool read = true);
    // the head (imi_tts_head) is already in the brain's own memory at positions 0..len-1, with the conv memory `conv`
    // right after it: a fresh sentence goes back there and reads only the rest of its prompt (the same numbers as
    // reading the whole prompt again, without the head's reading time)
    void set_head(const std::vector<int32_t> & ids, const std::vector<std::vector<float>> & conv);
    // the next frame's codes from brain.hidden; true: its samples were appended and fb (voice.n_in wide) is the brain's
    // next input; false: speech is over (the codec has been flushed)
    bool next(std::vector<float> & pcm, float * fb);
private:
    void start(unsigned seed, int max_frames);
    // a fresh conversation reads `prompt`: from the head if it starts with it, else from the start
    void read_fresh(const std::vector<int32_t> & prompt);
    std::vector<int32_t> head_ids;
    std::vector<std::vector<float>> head_conv;
    std::mt19937 rng;
    int max_frames = 2000;
    std::vector<int32_t> hist;
    std::vector<float> scores;
    std::vector<double> p;
    std::vector<char> seen;
    // sentence by sentence
    const imi_words * words = nullptr;
    std::vector<std::string> sents;
    size_t si = 0;
    int sent_n = 0, sent_cap = 0;
    bool keep_memory = true;
    std::vector<float> efb;
    void next_sentence();
};

// resets the brain, reads `prompt`, speaks; appends 24 kHz samples to pcm
imi_talk_stats imi_speak(imi_seoul & brain, imi_voice & voice, imi_codec & codec, const std::vector<int32_t> & prompt,
                         unsigned seed, std::vector<float> & pcm, int max_frames = 2000);

void imi_write_wav(const std::string & path, const std::vector<float> & pcm, int sr);
