// nunchi: the live interpreter. The speaker's sound comes in as he speaks (push), the live pause finder cuts it at his
// pauses, and one thread does the rest the way imi-instant does it on a clock (core/instant.cpp, where every step was
// measured): hearing that keeps its memory and hears ahead while he talks, translating at sentence ends with the
// sentences before kept, and the voice speaking each English sentence as soon as it is translated, its prompt read
// while the sentence is being translated.
#pragma once
#include "core/codec.h"
#include "core/ears.h"
#include "core/seoul.h"
#include "core/talk.h"
#include "core/voice.h"
#include "core/words.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// what the interpreter tells the outside; called on its own thread, so they must be quick
struct nunchi_events {
    std::function<void(const std::string & ko)> heard;                                  // a piece's Korean
    std::function<void(int id, const std::string & en)> writing;                        // a sentence's English so far
    std::function<void(int id, const std::string & ko, const std::string & en)> said;   // a sentence translated
    std::function<void(int id, const float * pcm, size_t n)> speech;                    // its voice, 24 kHz, to play
                                                                                         // right after what came before
    std::function<void(const std::string & line)> log;
};

struct nunchi_live {
    // the 4-bit Seoul brain and its ears + voice file
    bool load(const std::string & model, const std::string & mmproj, int threads);
    // the speaker's sound, 16 kHz mono, as it comes in (any thread)
    void push(const float * pcm, size_t n);
    // the interpreter: runs until stop is set
    void run(const nunchi_events & ev, const std::atomic<bool> & stop);
    // the week's terms, "Korean = English" one per line: where the heard Korean has one, its English goes right after it
    // in brackets for the translation (hints in place)
    void set_terms(const std::string & lines);
    std::string terms_text();

    std::atomic<bool> listening{ true };   // off: sound is dropped (songs, breaks); work already in finishes
    std::atomic<bool> voice_on{ true };    // off: captions only
    std::atomic<float> level_db{ -120 };   // the loudest 20 ms of the latest sound, for a meter
    std::atomic<int> heard_pieces{ 0 }, sentences{ 0 };

private:
    imi_team team;
    imi_seoul brain;
    imi_words words;
    imi_ears ears;
    imi_voice voice;
    imi_codec codec;
    std::mutex mu;
    std::vector<float> inbox;
    std::vector<std::pair<std::string, std::string>> terms;
    std::string terms_raw;
};
