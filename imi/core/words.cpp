#include "words.h"
#include "unicode_tables.h"

#include <climits>
#include <cstdio>

namespace {
// GPT-2's byte -> character table, turned around: character (as a code point) -> byte
std::vector<int> char_to_byte() {
    std::vector<int> back(512, -1);
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        const bool kept = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        back[kept ? b : 256 + extra++] = b;
    }
    return back;
}
std::string utf8(uint32_t cp) {
    std::string o;
    if (cp < 0x80) o += (char) cp;
    else if (cp < 0x800) { o += (char) (0xC0 | (cp >> 6)); o += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char) (0xE0 | (cp >> 12)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
    else { o += (char) (0xF0 | (cp >> 18)); o += (char) (0x80 | ((cp >> 12) & 0x3F)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
    return o;
}
template <size_t N> bool in_ranges(const uint32_t (&r)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) { const size_t m = (lo + hi) / 2; if (cp < r[m][0]) hi = m; else if (cp > r[m][1]) lo = m + 1; else return true; }
    return false;
}
bool is_letter(uint32_t cp) { return in_ranges(imi_letters, cp); }
bool is_number(uint32_t cp) { return in_ranges(imi_numbers, cp); }
bool is_space(uint32_t cp) {   // Unicode White_Space
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
// one UTF-8 code point from s at i (advances i); -1 if broken
int next_cp(const std::string & s, size_t & i) {
    const unsigned char c = s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!n || i + n > s.size()) { i++; return -1; }
    int cp = n == 1 ? c : c & (0x7F >> n);
    for (int k = 1; k < n; k++) cp = (cp << 6) | (s[i + k] & 0x3F);
    i += n;
    return cp;
}
}

bool imi_words::load(const imi_file & f) {
    const int kt = gguf_find_key(f.meta, "tokenizer.ggml.tokens"), ky = gguf_find_key(f.meta, "tokenizer.ggml.token_type");
    if (kt < 0) { fprintf(stderr, "imi-words: no word list\n"); return false; }
    const int n = (int) gguf_get_arr_n(f.meta, kt);
    const std::vector<int> back = char_to_byte();
    bytes.resize(n); type.assign(n, 1);
    const int32_t * ty = ky >= 0 ? (const int32_t *) gguf_get_arr_data(f.meta, ky) : nullptr;
    for (int i = 0; i < n; i++) {
        const std::string s = gguf_get_arr_str(f.meta, kt, i);
        if (ty) type[i] = ty[i];
        if (type[i] != 1) { bytes[i] = s; continue; }   // control and user-defined words are stored as plain text
        std::string o;
        for (size_t j = 0; j < s.size();) {
            const int cp = next_cp(s, j);
            if (cp >= 0 && cp < (int) back.size() && back[cp] >= 0) o += (char) back[cp];
        }
        bytes[i] = o;
    }
    im_end = find("<|im_end|>");
    // byte -> character, the forward table
    {
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            const bool kept = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            byte_char[b] = utf8(kept ? b : 256 + extra++);
        }
    }
    for (int i = 0; i < n; i++) if (type[i] == 1) normal[gguf_get_arr_str(f.meta, kt, i)] = i;
    const int km = gguf_find_key(f.meta, "tokenizer.ggml.merges");
    if (km >= 0) {
        const int nm = (int) gguf_get_arr_n(f.meta, km);
        for (int r = 0; r < nm; r++) {
            const std::string m = gguf_get_arr_str(f.meta, km, r);
            const size_t sp = m.find(' ', 1);
            if (sp == std::string::npos) continue;
            const std::string a = m.substr(0, sp), b = m.substr(sp + 1);
            auto ia = normal.find(a), ib = normal.find(b), ic = normal.find(a + b);
            if (ia == normal.end() || ib == normal.end() || ic == normal.end()) continue;
            const uint64_t key = ((uint64_t) (uint32_t) ia->second << 32) | (uint32_t) ib->second;
            if (!merges.count(key)) merges[key] = { r, ic->second };
        }
    }
    return true;
}

// Llama 3's split rule, as code (the same rule llama.cpp hand-writes for this regex):
//   's 't 're 've 'm 'll 'd | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3} | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<int32_t> imi_words::encode(const std::string & text) const {
    std::vector<uint32_t> cp;
    for (size_t i = 0; i < text.size();) { const int c = next_cp(text, i); cp.push_back(c < 0 ? 0xFFFD : (uint32_t) c); }
    const size_t n = cp.size();
    std::vector<std::pair<size_t, size_t>> pieces;   // [begin, end) in code points
    size_t prev = 0;
    auto cut = [&](size_t end) { if (end > prev) pieces.push_back({ prev, end }); prev = end; };
    auto L = [&](size_t p) { return p < n && is_letter(cp[p]); };
    auto N = [&](size_t p) { return p < n && is_number(cp[p]); };
    auto S = [&](size_t p) { return p < n && is_space(cp[p]); };
    auto lower = [&](size_t p) -> uint32_t { const uint32_t c = p < n ? cp[p] : 0; return c >= 'A' && c <= 'Z' ? c + 32 : c; };
    for (size_t p = 0; p < n;) {
        const uint32_t c = cp[p];
        if (c == '\'' && p + 1 < n) {
            const uint32_t c1 = lower(p + 1);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { p += 2; cut(p); continue; }
            if (p + 2 < n) {
                const uint32_t c2 = lower(p + 2);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { p += 3; cut(p); continue; }
            }
        }
        if (!(c == '\r' || c == '\n' || N(p)) && (L(p) || L(p + 1))) {
            p++;
            while (L(p)) p++;
            cut(p); continue;
        }
        if (N(p)) {
            size_t ini = p;
            while (N(p)) { if (++p - ini >= 3) { cut(p); ini = p; } }
            cut(p); continue;
        }
        {
            const size_t q = c == ' ' ? p + 1 : p;
            if (q < n && !(S(q) || L(q) || N(q))) {
                p = q;
                while (p < n && !(S(p) || L(p) || N(p))) p++;
                while (p < n && (cp[p] == '\r' || cp[p] == '\n')) p++;
                cut(p); continue;
            }
        }
        size_t ns = 0, last_nl = 0;
        while (S(p + ns)) { if (cp[p + ns] == '\r' || cp[p + ns] == '\n') last_nl = p + ns + 1; ns++; }
        if (last_nl > 0) { p = last_nl; cut(p); continue; }
        if (ns > 1 && p + ns < n) { p += ns - 1; cut(p); continue; }
        if (ns > 0) { p += ns; cut(p); continue; }
        cut(++p);
    }
    std::vector<int32_t> ids;
    for (const auto & pc : pieces) {
        std::string bl;   // the piece in byte-level characters
        for (size_t i = pc.first; i < pc.second; i++) for (char ch : utf8(cp[i])) bl += byte_char[(unsigned char) ch];
        auto whole = normal.find(bl);
        if (whole != normal.end()) { ids.push_back(whole->second); continue; }   // a whole piece in the list wins
        std::vector<int32_t> sym;
        for (size_t i = 0; i < bl.size();) {
            const size_t s0 = i; next_cp(bl, i);
            auto it = normal.find(bl.substr(s0, i - s0));
            sym.push_back(it == normal.end() ? -1 : it->second);
        }
        for (;;) {
            int at = -1; int32_t rank = INT_MAX, merged = -1;
            for (size_t i = 0; i + 1 < sym.size(); i++) {
                if (sym[i] < 0 || sym[i + 1] < 0) continue;
                auto it = merges.find(((uint64_t) (uint32_t) sym[i] << 32) | (uint32_t) sym[i + 1]);
                if (it != merges.end() && it->second.first < rank) { rank = it->second.first; merged = it->second.second; at = (int) i; }
            }
            if (at < 0) break;
            sym[at] = merged; sym.erase(sym.begin() + at + 1);
        }
        for (int32_t s : sym) if (s >= 0) ids.push_back(s);
    }
    return ids;
}

std::vector<int32_t> imi_words::encode_prompt(const std::string & text) const {
    std::vector<int32_t> ids;
    size_t i = 0, plain = 0;
    auto flush = [&](size_t end) {
        if (end > plain) { const std::vector<int32_t> p = encode(text.substr(plain, end - plain)); ids.insert(ids.end(), p.begin(), p.end()); }
    };
    while (i < text.size()) {
        int32_t best = -1; size_t len = 0;
        if (text[i] == '<' || text[i] == '[')   // every special word here starts with one of these; longest match wins
            for (size_t w = 0; w < bytes.size(); w++)
                if (type[w] != 1 && bytes[w].size() > len && text.compare(i, bytes[w].size(), bytes[w]) == 0) { best = (int32_t) w; len = bytes[w].size(); }
        if (best < 0) { i++; continue; }
        flush(i); ids.push_back(best); i += len; plain = i;
    }
    flush(text.size());
    return ids;
}

int32_t imi_words::find(const std::string & piece) const {
    for (size_t i = 0; i < bytes.size(); i++) if (type[i] != 1 && bytes[i] == piece) return (int32_t) i;
    for (size_t i = 0; i < bytes.size(); i++) if (bytes[i] == piece) return (int32_t) i;
    return -1;
}

std::string imi_words::text(const int32_t * ids, int n) const {
    std::string o;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= (int) bytes.size() || type[ids[i]] == 3) continue;
        o += bytes[ids[i]];
    }
    return o;
}
