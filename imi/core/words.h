// imi.cpp: the brain's word list, both ways.
//   text -> ids: Llama 3's split rule (letters, numbers up to 3 digits, punctuation runs, spaces), each piece as one word
//     if the list has it, else GPT-2 byte-level merges by rank. Control words are never made from text: a prompt
//     puts them in by id.
//   ids -> text: each word is stored as byte-level text (every byte written as one printable character), so decoding
//     maps those characters back to bytes. Control words (<|im_end|> and the like) print nothing.
#pragma once
#include "../model.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct imi_words {
    std::vector<std::string> bytes;   // each word's raw bytes
    std::vector<int32_t> type;        // 1 normal, 3 control, 4 user-defined
    int32_t im_end = -1;
    bool load(const imi_file & f);
    int32_t find(const std::string & piece) const;   // a word by its stored text, -1 if none
    std::string text(const int32_t * ids, int n) const;
    std::vector<int32_t> encode(const std::string & text) const;
    // a prompt: control and user-defined words written as text (<|im_start|> ...) become their ids, the rest is encoded
    std::vector<int32_t> encode_prompt(const std::string & text) const;

private:
    std::unordered_map<std::string, int32_t> normal;            // byte-level text -> id, normal words
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges;   // (left id, right id) -> (rank, merged id)
    std::string byte_char[256];                                  // byte -> its printable character (UTF-8)
};
