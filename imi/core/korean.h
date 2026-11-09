// imi.cpp: where a Korean sentence ends, from its last syllables. Korean puts the verb last, so a sentence is known
// once its final verb ending lands: -다 -요 -죠 -라 -오, -까 as a question (합니까, 할까; 하니까 "because" joins), or ? !.
// Joining endings (-고 -서 -며 -면 -데 -니 -도록, 하지만) carry a sentence on; a long sentence can be cut there.
#pragma once
#include <string>
#include <vector>

// the last Hangul syllables (up to 3, spaces and punctuation skipped) as code points; ? or ! at the end noted
struct imi_ko_tail { std::vector<int> syl; bool mark = false; };

inline imi_ko_tail imi_ko_tail_of(const std::string & ko) {
    imi_ko_tail t;
    size_t e = ko.size();
    while (e > 0 && (ko[e - 1] == ' ' || ko[e - 1] == '.' || ko[e - 1] == ',' || ko[e - 1] == '?' || ko[e - 1] == '!' || ko[e - 1] == '\n')) {
        if (ko[e - 1] == '?' || ko[e - 1] == '!') t.mark = true;
        e--;
    }
    while (e >= 3 && t.syl.size() < 3) {   // Hangul syllables are 3 bytes in UTF-8
        const unsigned char a = ko[e - 3], b = ko[e - 2], c = ko[e - 1];
        if ((a & 0xF0) != 0xE0) break;
        const int cp = ((a & 0x0F) << 12) | ((b & 0x3F) << 6) | (c & 0x3F);
        if (cp < 0xAC00 || cp > 0xD7A3) break;
        t.syl.insert(t.syl.begin(), cp);
        e -= 3;
    }
    return t;
}
inline int imi_ko_syl(const char * s) { const unsigned char * u = (const unsigned char *) s; return ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F); }
inline int imi_ko_batchim(int cp) { return (cp - 0xAC00) % 28; }   // final consonant: 0 none, 8 ㄹ, 17 ㅂ

inline bool imi_ko_sentence_end(const std::string & ko) {
    const imi_ko_tail t = imi_ko_tail_of(ko);
    if (t.syl.empty()) return false;
    const int last = t.syl.back(), prev = t.syl.size() > 1 ? t.syl[t.syl.size() - 2] : 0;
    for (const char * e : { "다", "요", "죠", "라", "오" }) if (last == imi_ko_syl(e)) return true;
    if (last == imi_ko_syl("까")) {   // 합니까 / 할까 ask; 하니까 ("because") joins
        if (prev == imi_ko_syl("니")) { const int p2 = t.syl.size() > 2 ? t.syl[t.syl.size() - 3] : 0; return p2 && imi_ko_batchim(p2) == 17; }
        return true;
    }
    if (!t.mark) return false;
    for (const char * e : { "니", "냐", "나", "가", "래" }) if (last == imi_ko_syl(e)) return true;
    if (last == imi_ko_syl("지"))   // 하지? 있지? ask; 하는지 / 좋은지 / 인지 / 할지 are inside a sentence
        return !(prev == imi_ko_syl("는") || prev == imi_ko_syl("은") || (prev && (imi_ko_batchim(prev) == 4 || imi_ko_batchim(prev) == 8)));
    return false;
}

inline bool imi_ko_joining_end(const std::string & ko) {
    const imi_ko_tail t = imi_ko_tail_of(ko);
    if (t.syl.empty()) return false;
    const int last = t.syl.back(), prev = t.syl.size() > 1 ? t.syl[t.syl.size() - 2] : 0;
    for (const char * e : { "고", "서", "며", "면", "데", "니", "록" }) if (last == imi_ko_syl(e)) return true;   // not -게: an adverb (일관성 있게)
    if (last == imi_ko_syl("만") && prev == imi_ko_syl("지")) return true;   // 하지만
    if (last == imi_ko_syl("까") && prev == imi_ko_syl("니")) return !imi_ko_sentence_end(ko);
    return false;
}

// a heard piece's Korean, as it joins the sentence so far: the hearing writes a full stop (or ?) at the end of every
// piece, also where the speaker only paused mid-sentence; that mark goes unless the piece really ends a sentence
inline std::string imi_ko_piece(std::string ko) {
    while (!ko.empty() && (ko.back() == ' ' || ko.back() == '\n')) ko.pop_back();
    while (!ko.empty() && ko.front() == ' ') ko.erase(ko.begin());
    while (!ko.empty() && (ko.back() == '.' || ko.back() == '?' || ko.back() == '!') && !imi_ko_sentence_end(ko)) ko.pop_back();
    return ko;
}

// a sentence end inside the Korean so far, with more words after it (the hearing sometimes writes nothing for a piece
// and its words come with the next one, so a sentence end can land mid-text): the byte position just after the first
// such . ? ! (with any repeats or closing quote), or npos
inline size_t imi_ko_inner_end(const std::string & ko) {
    for (size_t i = 0; i < ko.size(); i++) {
        if (ko[i] != '.' && ko[i] != '?' && ko[i] != '!') continue;
        size_t j = i + 1;
        while (j < ko.size() && (ko[j] == '.' || ko[j] == '?' || ko[j] == '!' || ko[j] == '"' || ko[j] == '\'')) j++;
        if (j >= ko.size() || ko[j] != ' ') continue;
        size_t k = j;
        while (k < ko.size() && ko[k] == ' ') k++;
        if (k >= ko.size()) continue;
        if (imi_ko_sentence_end(ko.substr(0, j))) return j;
        i = j - 1;
    }
    return std::string::npos;
}

// translate now? (the sentence so far, its seconds of speech): a sentence end; past 8 s a joining ending; past 12 s anywhere
inline bool imi_ko_translate_now(const std::string & ko, double secs) {
    return imi_ko_sentence_end(ko) || (secs >= 8 && imi_ko_joining_end(ko)) || secs >= 12;
}
